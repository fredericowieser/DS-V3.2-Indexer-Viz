import os
import json
import csv
import torch
import torch.distributed as dist
from argparse import ArgumentParser
from typing import Optional
from transformers import AutoTokenizer
from safetensors.torch import load_model as st_load_model

from inference.model import Transformer, ModelArgs, Indexer, apply_rotary_emb
from inference.kernel import act_quant, fp8_index, rotate_activation


TEXT_SNIPPET = """
Hey there! I'm testing the indexer attention mechanism in this transformer model.
"""

CAPTURED_DATA = []


def patched_indexer_forward(self, x: torch.Tensor, qr: torch.Tensor, start_pos: int, freqs_cis: torch.Tensor, mask: Optional[torch.Tensor]):
    """
    A patched version of Indexer.forward that intercepts and records 
    the calculated index scores before they are discarded.
    """
    bsz, seqlen, _ = x.size()
    end_pos = start_pos + seqlen

    # Prepare Queries
    q = self.wq_b(qr)
    q = q.view(bsz, seqlen, self.n_heads, self.head_dim)
    q_pe, q_nope = torch.split(q, [self.rope_head_dim, self.head_dim - self.rope_head_dim], dim=-1)
    
    # Note: RoPE in indexer is not interleaved
    q_pe = apply_rotary_emb(q_pe, freqs_cis, False)
    q = torch.cat([q_pe, q_nope], dim=-1)

    # Prepare Keys
    k = self.wk(x)
    k = self.k_norm(k)
    k_pe, k_nope = torch.split(k, [self.rope_head_dim, self.head_dim - self.rope_head_dim], dim=-1)
    k_pe = apply_rotary_emb(k_pe.unsqueeze(2), freqs_cis, False).squeeze(2)
    k = torch.cat([k_pe, k_nope], dim=-1)

    # Quantization & Rotation
    q = rotate_activation(q)
    k = rotate_activation(k)
    q_fp8, q_scale = act_quant(q, block_size=128, scale_fmt=self.scale_fmt)
    k_fp8, k_scale = act_quant(k, block_size=128, scale_fmt=self.scale_fmt)

    # Update Cache
    self.k_cache[:bsz, start_pos:end_pos] = k_fp8
    self.k_scale_cache[:bsz, start_pos:end_pos] = k_scale

    # Score Calculation
    weights = self.weights_proj(x.float()) * self.n_heads ** -0.5
    weights = weights.unsqueeze(-1) * q_scale * self.softmax_scale
    
    # index_score shape: (batch_size, query_len, total_key_len)
    index_score = fp8_index(
        q_fp8.contiguous(), 
        weights, 
        self.k_cache[:bsz, :end_pos].contiguous(), 
        self.k_scale_cache[:bsz, :end_pos].contiguous()
    )
    if mask is not None:
        index_score += mask

    # Top-K Selection
    # We capture BOTH values and indices here
    topk_values, topk_indices = index_score.topk(min(self.index_topk, end_pos), dim=-1)
    try:
        # We capture data only for the LAST token in the input sequence.
        # This represents the "prediction step" context.
        # Shape of topk_values: (batch, seqlen, k) -> we take [:, -1, :]
        last_token_scores = topk_values[:, -1, :].float().cpu().numpy()
        last_token_indices = topk_indices[:, -1, :].long().cpu().numpy()
        current_layer = getattr(self, 'layer_id', -1)
        # Iterate over batch (usually 1)
        for b in range(bsz):
            # Iterate over the top-k selections
            for k_rank in range(last_token_scores.shape[1]):
                CAPTURED_DATA.append({
                    'layer_id': current_layer,
                    'batch_idx': b,
                    'query_idx': start_pos + seqlen - 1, # Absolute position of the query token
                    'rank': k_rank, # 0 = highest score
                    'key_idx': int(last_token_indices[b, k_rank]), # The token being attended to
                    'score': float(last_token_scores[b, k_rank])
                })
    except Exception as e:
        print(f"[Rank {dist.get_rank()}] Error capturing data: {e}")
    # Broadcast indices as required by distributed setup
    topk_indices_ = topk_indices.clone()
    dist.broadcast(topk_indices_, src=0)
    return topk_indices


def main():
    parser = ArgumentParser()
    parser.add_argument("--ckpt-path", type=str, required=True, help="Path to checkpoint folder")
    parser.add_argument("--config", type=str, required=True, help="Path to JSON config")
    args_cli = parser.parse_args()

    # Distributed Init
    world_size = int(os.getenv("WORLD_SIZE", "1"))
    rank = int(os.getenv("RANK", "0"))
    local_rank = int(os.getenv("LOCAL_RANK", "0"))
    
    if world_size > 1:
        dist.init_process_group("nccl")
    
    torch.cuda.set_device(local_rank)
    torch.set_default_dtype(torch.bfloat16)
    torch.manual_seed(42)

    # Load Configuration
    with open(args_cli.config) as f:
        model_args = ModelArgs(**json.load(f))

    # APPLY MONKEY PATCH
    if rank == 0:
        print(f"Patching Indexer.forward to capture scores...")
    Indexer.forward = patched_indexer_forward

    # Model Init
    with torch.device("cuda"):
        model = Transformer(model_args)

    # Inject Layer IDs so the patch knows which layer is running
    for i, layer in enumerate(model.layers):
        layer.attn.indexer.layer_id = i

    # Load Weights
    ckpt_file = os.path.join(args_cli.ckpt_path, f"model{rank}-mp{world_size}.safetensors")
    if rank == 0:
        print(f"Loading weights...")
    st_load_model(model, ckpt_file)

    # Load tokenizer from the checkpoint path
    if rank == 0:
        print(f"Tokenizing input snippet...")
        
    tokenizer = AutoTokenizer.from_pretrained(args_cli.ckpt_path, trust_remote_code=True)
    tokens = tokenizer.encode(TEXT_SNIPPET, return_tensors="pt").cuda()
    
    if rank == 0:
        print(f"Input Length: {tokens.size(1)} tokens")
        print(f"Input Text: {TEXT_SNIPPET.strip()}")

    # Run Inference
    if rank == 0:
        print(f"Running inference pass...")

    with torch.inference_mode():
        model(tokens)

    # Save Data to CSV
    csv_filename = f"index_scores_rank_{rank}.csv"
    if rank == 0:
        print(f"Saving {len(CAPTURED_DATA)} rows to {csv_filename}...")
    
    keys = ['layer_id', 'batch_idx', 'query_idx', 'rank', 'key_idx', 'score']
    
    try:
        with open(csv_filename, 'w', newline='') as f:
            writer = csv.DictWriter(f, fieldnames=keys)
            writer.writeheader()
            writer.writerows(CAPTURED_DATA)
        if rank == 0:
            print(f"Success. Data saved.")
    except Exception as e:
        print(f"[Rank {rank}] Failed to write CSV: {e}")

    if world_size > 1:
        dist.destroy_process_group()

if __name__ == "__main__":
    main()