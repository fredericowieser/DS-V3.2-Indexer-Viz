import os
import torch
import torch.distributed as dist
import tilelang
import tilelang.language as T
from typing import Tuple, Optional
import math

_index_call_count = 0

def _dev_mode_enabled() -> bool:
    return os.environ.get("DS_DEV_MODE", "0") == "1" or os.environ.get("DS_DISABLE_TILELANG", "0") == "1"

_FP8_MIN = -448.0
_FP8_MAX = 448.0

def _round_scale_pow2_up(scale: torch.Tensor) -> torch.Tensor:
    """
    Snap positive scale to the next power-of-two (>= scale).
    Matches the spirit of the kernel's round_scale path.
    """
    # scale is positive
    eps = torch.finfo(scale.dtype).tiny if scale.is_floating_point() else 1e-38
    scale = torch.clamp(scale, min=eps)
    return torch.pow(2.0, torch.ceil(torch.log2(scale)))

def _act_quant_torch(
    x: torch.Tensor, block_size: int = 128, round_scale: bool = False
) -> tuple[torch.Tensor, torch.Tensor]:
    """
    Blockwise FP8 quantization along last dimension.
    Returns:
      y: float8_e4m3fn same shape as x
      s: float32 scales shape x[..., N/block_size]
    """
    assert x.is_contiguous(), "Input tensor must be contiguous"
    assert x.size(-1) % block_size == 0, "Last dim must be divisible by block_size"

    N = x.size(-1)
    g = N // block_size

    x_view = x.view(*x.shape[:-1], g, block_size)  # (..., g, bs)
    amax = x_view.abs().amax(dim=-1)               # (..., g)
    amax = torch.clamp(amax, min=1e-4)

    scale = (amax / _FP8_MAX).to(torch.float32)    # (..., g) float32

    if round_scale:
        scale = _round_scale_pow2_up(scale)

    scale_expand = scale.to(x.dtype).unsqueeze(-1)  # (..., g, 1) in x dtype for division
    y = (x_view / scale_expand).clamp(_FP8_MIN, _FP8_MAX).to(torch.float8_e4m3fn).view_as(x)

    return y, scale

def _dequant_fp8_activation(a_fp8: torch.Tensor, a_s: torch.Tensor, block_size: int = 128) -> torch.Tensor:
    """
    a_fp8: (..., K) float8
    a_s:   (..., K/block) float32
    Returns: (..., K) float16/bf16/float32 (torch default dtype)
    """
    K = a_fp8.size(-1)
    assert K % block_size == 0
    g = K // block_size
    a_view = a_fp8.float().view(*a_fp8.shape[:-1], g, block_size)
    s_view = a_s.float().view(*a_fp8.shape[:-1], g).unsqueeze(-1)
    out = (a_view * s_view).to(torch.get_default_dtype()).view_as(a_fp8)
    return out

def _dequant_fp8_weight(b_fp8: torch.Tensor, b_s: torch.Tensor, block_size: int = 128) -> torch.Tensor:
    """
    b_fp8: (N, K) float8
    b_s:   (N/block, K/block) float32
    Returns: (N, K) dequant (torch default dtype)
    """
    assert b_fp8.dim() == 2
    N, K = b_fp8.shape
    assert N % block_size == 0 and K % block_size == 0

    nb = N // block_size
    kb = K // block_size

    # (nb, bs, kb, bs) -> (nb, kb, bs, bs)
    w = b_fp8.float().view(nb, block_size, kb, block_size).transpose(1, 2).contiguous()
    s = b_s.float().view(nb, kb).unsqueeze(-1).unsqueeze(-1)  # (nb,kb,1,1)
    w = (w * s).to(torch.get_default_dtype())
    w = w.transpose(1, 2).contiguous().view(N, K)
    return w

def _fp8_index_torch(q: torch.Tensor, q_s: torch.Tensor, k: torch.Tensor, k_s: torch.Tensor) -> torch.Tensor:
    """
    Equivalent to the kernel:
      logits = relu(k @ q^T) * q_s
      sum over heads -> logits_sum
      logits_sum *= k_s
    Shapes:
      q:   (b,m,h,d) float8
      q_s: (b,m,h)   float32
      k:   (b,n,d)   float8
      k_s: (b,n)     float32
    Returns:
      (b,m,n) float32
    """
    qf = q.float()
    kf = k.float()

    # (b,n,d) dot (b,m,h,d) -> (b,m,n,h)
    logits = torch.einsum("bnd,bmhd->bmnh", kf, qf)
    logits = torch.relu(logits)

    logits = logits * q_s.float().unsqueeze(2)      # (b,m,1,h)
    out = logits.sum(dim=-1)                        # (b,m,n)
    out = out * k_s.float().unsqueeze(1)            # (b,1,n)
    return out


tilelang.set_log_level("WARNING")

pass_configs = {
    tilelang.PassConfigKey.TL_DISABLE_WARP_SPECIALIZED: True,
    tilelang.PassConfigKey.TL_DISABLE_TMA_LOWER: True,
    tilelang.PassConfigKey.TL_DISABLE_FAST_MATH: True,
}

FP8 = "float8_e4m3"
BF16 = "bfloat16"
FP32 = "float32"


# fast_log2_ceil:
# Computes an approximate log2 of `x` by reinterpreting the float32 bit representation as an integer.
# It extracts the exponent (`exp_x`) and mantissa (`man_bits`) using bitwise shifts and masks.
# The result is derived from the exponent bias (127) and a correction if the mantissa is non-zero, returning an integer approximation.
def fast_log2_ceil(x):
    bits_x = T.reinterpret("uint32", x)
    exp_x = (bits_x >> 23) & 0xFF
    man_bits = bits_x & ((1 << 23) - 1)
    return T.Cast("int32", exp_x - 127 + T.if_then_else(man_bits != 0, 1, 0))


# fast_pow2:
# Computes an approximate power of 2 for input `x` using bitwise manipulation.
# It shifts the input `x` (offset by the bias 127) into the exponent position of a float32 representation.
# The resulting integer bit pattern is then reinterpreted back as a float32, effectively calculating 2^x.
def fast_pow2(x):
    bits_x = (x + 127) << 23
    return T.reinterpret("float32", bits_x)


# fast_round_scale:
# Calculates a power-of-2 scaling factor for quantization based on the maximum absolute value `amax`.
# It multiplies `amax` by the inverse of the FP8 max value to normalize the range.
# Then calls `fast_log2_ceil` and `fast_pow2` to snap this normalized scale to the nearest upper power of 2.
def fast_round_scale(amax, fp8_max_inv):
    return fast_pow2(fast_log2_ceil(amax * fp8_max_inv))


# act_quant_kernel:
# Defines a TileLang JIT kernel for activation quantization that divides input `X` into blocks of `blk_m` x `group_size`.
# It loads data to shared memory, computes the row-wise max (`amax_local`), and determines scales (`s_local`) (optionally rounding them).
# Finally, it normalizes elements by `s_local`, clamps them to the FP8 range, and stores the quantized `Y` and scales `S`.
@tilelang.jit(pass_configs=pass_configs, target=_TL_TARGET)
def act_quant_kernel(
    N, in_dtype=BF16, out_dtype=FP8, scale_dtype=FP32, round_scale=False
):
    M = T.symbolic("M")
    fp8_min = -448.0
    fp8_max = 448.0
    fp8_max_inv = 1 / fp8_max
    num_stages = 0 if round_scale else 2
    blk_m = 32
    group_size = 128

    @T.prim_func
    def act_quant_kernel_(
        X: T.Tensor[(M, N), in_dtype],
        Y: T.Tensor[(M, N), out_dtype],
        S: T.Tensor[(M, T.ceildiv(N, group_size)), scale_dtype],
    ):
        with T.Kernel(T.ceildiv(M, blk_m), T.ceildiv(N, group_size), threads=128) as (
            pid_m,
            pid_n,
        ):
            x_shared = T.alloc_shared((blk_m, group_size), in_dtype)
            x_local = T.alloc_fragment((blk_m, group_size), in_dtype)
            amax_local = T.alloc_fragment((blk_m,), scale_dtype)
            s_local = T.alloc_fragment((blk_m,), scale_dtype)
            y_local = T.alloc_fragment((blk_m, group_size), out_dtype)
            y_shared = T.alloc_shared((blk_m, group_size), out_dtype)

            for _ in T.Pipelined(1, num_stages=num_stages):
                T.copy(X[pid_m * blk_m, pid_n * group_size], x_shared)
                T.copy(x_shared, x_local)
                T.reduce_absmax(x_local, amax_local, dim=1)
                for i in T.Parallel(blk_m):
                    amax_local[i] = T.max(amax_local[i], 1e-4)
                    if round_scale:
                        s_local[i] = fast_round_scale(amax_local[i], fp8_max_inv)
                    else:
                        s_local[i] = amax_local[i] * fp8_max_inv
                for i, j in T.Parallel(blk_m, group_size):
                    y_local[i, j] = T.clamp(
                        x_local[i, j] / s_local[i], fp8_min, fp8_max
                    )
                for i in T.Parallel(blk_m):
                    S[pid_m * blk_m + i, pid_n] = s_local[i]
                T.copy(y_local, y_shared)
                T.copy(y_shared, Y[pid_m * blk_m, pid_n * group_size])

    return act_quant_kernel_


# act_quant:
# Serves as the high-level entry point for activation quantization, verifying input contiguousness and block alignment.
# It instantiates output tensors `y` (FP8) and `s` (scales), then invokes the JIT-compiled `act_quant_kernel`.
# The function flattens inputs to manage dynamic shapes before returning the quantized tensor and its scales.
def act_quant(
    x: torch.Tensor, block_size: int = 128, scale_fmt: Optional[str] = None
) -> Tuple[torch.Tensor, torch.Tensor]:
    """
    Quantizes the input tensor `x` using block-wise quantization.

    Args:
        x (torch.Tensor): The input tensor to be quantized. Must be contiguous and its last dimension size must be divisible by `block_size`.
        block_size (int, optional): The size of the blocks to be used for quantization. Default is 128.
        scale_fmt (Optional[str], optional): The format of the scale. Default is None.
    Returns:
        Tuple[torch.Tensor, torch.Tensor]: A tuple containing:
            - The quantized tensor with dtype `torch.float8_e4m3fn`.
            - A tensor of scaling factors with dtype `torch.float32`.
    """
    assert x.is_contiguous(), "Input tensor must be contiguous"
    assert x.size(-1) % block_size == 0, (
        f"Last dimension size must be divisible by block_size (block_size={block_size})"
    )

    if _dev_mode_enabled():
        return _act_quant_torch(x, block_size=block_size, round_scale=scale_fmt is not None)

    N = x.size(-1)
    y = torch.empty_like(x, dtype=torch.float8_e4m3fn)
    s = x.new_empty(*x.size()[:-1], N // block_size, dtype=torch.float32)
    kernel = act_quant_kernel(N, round_scale=scale_fmt is not None)
    kernel(x.view(-1, N), y.view(-1, N), s.view(-1, N // block_size))
    return y, s


# fp8_gemm_kernel:
# Defines a pipelined FP8 General Matrix Multiplication (GEMM) kernel that computes C = A * B.
# It manages shared memory loads for tiles of A, B, and their scales, fusing the scale multiplication into the accumulation loop.
# The kernel utilizes swizzling for L2 cache efficiency and accumulates in `accum_dtype` (float32) before storing the final result.
@tilelang.jit(pass_configs=pass_configs, target=_TL_TARGET)
def fp8_gemm_kernel(N, K, out_dtype=BF16, accum_dtype="float32"):
    assert out_dtype in [BF16, "float32"]

    M = T.symbolic("M")
    group_size = 128
    block_M = 32
    block_N = 128
    block_K = 128

    @T.prim_func
    def fp8_gemm_kernel_(
        A: T.Tensor[(M, K), FP8],
        B: T.Tensor[(N, K), FP8],
        C: T.Tensor[(M, N), out_dtype],
        scales_a: T.Tensor[(M, T.ceildiv(K, group_size)), FP32],
        scales_b: T.Tensor[(T.ceildiv(N, group_size), T.ceildiv(K, group_size)), FP32],
    ):
        with T.Kernel(T.ceildiv(N, block_N), T.ceildiv(M, block_M), threads=128) as (
            bx,
            by,
        ):
            A_shared = T.alloc_shared((block_M, block_K), FP8)
            B_shared = T.alloc_shared((block_N, block_K), FP8)
            C_shared = T.alloc_shared((block_M, block_N), out_dtype)
            Scale_C_shared = T.alloc_shared((block_M), FP32)
            C_local = T.alloc_fragment((block_M, block_N), accum_dtype)
            C_local_accum = T.alloc_fragment((block_M, block_N), accum_dtype)

            # Improve L2 Cache
            T.use_swizzle(panel_size=10)

            T.clear(C_local)
            T.clear(C_local_accum)
            K_iters = T.ceildiv(K, block_K)
            for k in T.Pipelined(K_iters, num_stages=4):
                # Load A into shared memory
                T.copy(A[by * block_M, k * block_K], A_shared)
                # Load B into shared memory
                T.copy(B[bx * block_N, k * block_K], B_shared)
                # Load scale into shared memory
                Scale_B = scales_b[bx * block_N // group_size, k]
                for i in T.Parallel(block_M):
                    Scale_C_shared[i] = scales_a[by * block_M + i, k] * Scale_B

                T.gemm(A_shared, B_shared, C_local, transpose_B=True)
                # Promote to enable 2xAcc
                for i, j in T.Parallel(block_M, block_N):
                    C_local_accum[i, j] += C_local[i, j] * Scale_C_shared[i]
                T.clear(C_local)
            # TMA store
            T.copy(C_local_accum, C_shared)
            T.copy(C_shared, C[by * block_M, bx * block_N])

    return fp8_gemm_kernel_


# fp8_gemm:
# Orchestrates the FP8 matrix multiplication by checking input contiguousness and flattening tensors for the kernel.
# It allocates the result tensor `c` and dispatches the `fp8_gemm_kernel` with the appropriate problem sizes.
# The inputs include matrices `a` and `b` along with their corresponding block-wise scaling factors `a_s` and `b_s`.
def fp8_gemm(
    a: torch.Tensor, a_s: torch.Tensor, b: torch.Tensor, b_s: torch.Tensor
) -> torch.Tensor:
    """
    Perform a matrix multiplication using FP8 precision.

    Args:
        a (torch.Tensor): The first input matrix, must be contiguous.
        a_s (torch.Tensor): The scaling factor for the first input matrix, must be contiguous.
        b (torch.Tensor): The second input matrix, must be contiguous.
        b_s (torch.Tensor): The scaling factor for the second input matrix, must be contiguous.

    Returns:
        torch.Tensor: The result of the matrix multiplication.
    """
    assert a.is_contiguous() and b.is_contiguous(), "Input tensors must be contiguous"
    assert a_s.is_contiguous() and b_s.is_contiguous(), (
        "Scaling factor tensors must be contiguous"
    )

    if _dev_mode_enabled():
        # Dequant + matmul (slow but correct)
        K = a.size(-1)
        M = a.numel() // K
        N = b.size(0)

        a2 = _dequant_fp8_activation(a.view(M, K), a_s.view(M, -1))
        b2 = _dequant_fp8_weight(b, b_s)

        c2 = a2 @ b2.t()  # (M,N)
        return c2.view(*a.size()[:-1], N)

    K = a.size(-1)
    M = a.numel() // K
    N = b.size(0)
    c = a.new_empty(*a.size()[:-1], N, dtype=torch.get_default_dtype())
    kernel = fp8_gemm_kernel(N, K)
    kernel(a.view(M, K), b, c.view(M, N), a_s.view(M, -1), b_s)
    return c


# fp8_index_kernel:
# Defines a specialized kernel for computing index scores using FP8 vector-matrix operations.
# It loads a query vector `q` and iterates over key vectors `k`, performing a GEMM to get logits.
# It applies ReLU activation, scales the logits using `q_s` and `k_s`, and sums them to produce the final `o` scores.
@tilelang.jit(out_idx=[4], pass_configs=pass_configs, target=_TL_TARGET)
def fp8_index_kernel(h: int, d: int):
    b = T.symbolic("b")
    m = T.symbolic("m")
    n = T.symbolic("n")

    blk_n1 = 512
    blk_n2 = 128

    @T.prim_func
    def fp8_index_kernel_(
        q: T.Tensor[(b, m, h, d), FP8],
        q_s: T.Tensor[(b, m, h), FP32],
        k: T.Tensor[(b, n, d), FP8],
        k_s: T.Tensor[(b, n), FP32],
        o: T.Tensor[(b, m, n), FP32],
    ) -> None:
        with T.Kernel(b, m, T.ceildiv(n, blk_n1)) as (i_b, i_m, i1_n):
            q_smem = T.alloc_shared((h, d), FP8)
            T.copy(q[i_b, i_m, 0, 0], q_smem)

            q_s_frag = T.alloc_fragment(h, FP32)
            T.copy(q_s[i_b, i_m, 0], q_s_frag)

            for i2_n in T.Pipelined(blk_n1 // blk_n2, num_stages=2):
                k_smem = T.alloc_shared((blk_n2, d), FP8)
                T.copy(k[i_b, i1_n * blk_n1 + i2_n * blk_n2, 0], k_smem)

                k_s_frag = T.alloc_fragment(blk_n2, FP32)
                T.copy(k_s[i_b, i1_n * blk_n1 + i2_n * blk_n2], k_s_frag)

                logits = T.alloc_fragment((blk_n2, h), FP32)
                T.gemm(
                    k_smem,
                    q_smem,
                    logits,
                    transpose_A=False,
                    transpose_B=True,
                    clear_accum=True,
                )

                for i_h, i3_n in T.Parallel(h, blk_n2):
                    logits[i3_n, i_h] = T.max(logits[i3_n, i_h], 0) * q_s_frag[i_h]

                logits_sum = T.alloc_fragment(blk_n2, FP32)
                T.reduce_sum(logits, logits_sum, dim=1)

                for i3_n in T.Parallel(blk_n2):
                    logits_sum[i3_n] *= k_s_frag[i3_n]

                T.copy(logits_sum, o[i_b, i_m, i1_n * blk_n1 + i2_n * blk_n2])

    return fp8_index_kernel_


# fp8_index:
# High-level wrapper for the `fp8_index_kernel` that computes expert routing scores using FP8 precision.
# It invokes the JIT-compiled kernel with query `q` and key `k` tensors and their respective scales.
# The calculation involves `ReLU(q @ k.T) * scales` to effectively score the relevance of experts.
def fp8_index(
    q: torch.Tensor,
    q_s: torch.Tensor,
    k: torch.Tensor,
    k_s: torch.Tensor,
) -> torch.Tensor:
    """
    Perform index score using FP8 precision.

    Args:
        q (torch.Tensor): The Q tensor, must be contiguous.
        q_s (torch.Tensor): The scaling factor for Q (float), must be contiguous.
        k (torch.Tensor): The K tensor, must be contiguous.
        k_s (torch.Tensor): The scaling factor for K (e8m0 here), must be contiguous.

        fp8 q @ fp8 k -> fp32 logits
        relu(fp32 logits) * q_s (weights) -> fp32 logits
        fp32 logits -> fp32 logits_sum
        fp32 logits_sum * k_s (e8m0) -> fp32 index_score
    """
    global _index_call_count
    
    if _dev_mode_enabled():
        o = _fp8_index_torch(q, q_s, k, k_s)
    else:
        o = fp8_index_kernel(q.shape[2], q.shape[3])(q, q_s, k, k_s)
    
    if os.environ.get("RECORD_INDEX") == "1":
        rank = dist.get_rank() if dist.is_initialized() else 0
        if rank == 0:
            if _index_call_count == 0:
                os.makedirs("logs", exist_ok=True)
            torch.save(o.detach().cpu(), f"logs/scores_{_index_call_count}.pt")
            _index_call_count += 1
            
    return o
