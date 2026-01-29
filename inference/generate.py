import os
import json
import inspect
import csv
from argparse import ArgumentParser
from copy import deepcopy
from typing import List, Any
import numpy as np
from accelerate.utils import offload
import accelerate.utils.modeling
import torch.nn.functional as F

# Monkeypatch accelerate to support FP8 offloading via int8 view
_orig_offload_weight = offload.offload_weight
_orig_load_offloaded_weight = offload.load_offloaded_weight

def _patched_offload_weight(weight, name, save_folder, index=None):
    if weight.dtype == torch.float8_e4m3fn:
        weight = weight.view(torch.int8)
    return _orig_offload_weight(weight, name, save_folder, index)

def _patched_load_offloaded_weight(file, index=None):
    weight = _orig_load_offloaded_weight(file, index)
    if isinstance(weight, torch.Tensor) and weight.dtype == torch.int8:
        return weight.view(torch.float8_e4m3fn)
    return weight

# Apply to definition
offload.offload_weight = _patched_offload_weight
offload.load_offloaded_weight = _patched_load_offloaded_weight

# Apply to usage in modeling (CRITICAL: this is where the call comes from)
accelerate.utils.modeling.offload_weight = _patched_offload_weight
accelerate.utils.modeling.load_offloaded_weight = _patched_load_offloaded_weight

import torch
import torch.distributed as dist
from transformers import AutoTokenizer
from accelerate import init_empty_weights, load_checkpoint_and_dispatch
from accelerate.utils import set_module_tensor_to_device

from model import Transformer, ModelArgs, precompute_freqs_cis


def _default_system_prompt() -> str:
    # Highest priority: explicit prompt override
    sp = os.getenv("DS_SYSTEM_PROMPT")
    if sp and sp.strip():
        return sp.strip()

    lang = os.getenv("DS_LANG", "en").strip().lower()
    if lang in {"en", "english"}:
        return (
            "You are a helpful assistant. "
            "Respond in English unless the user explicitly asks you to use another language. "
            "If the user writes in another language but asks for English, respond in English."
        )
    if lang in {"zh", "zh-cn", "zh-hans", "chinese"}:
        return (
            "你是一个有帮助的助手。"
            "除非用户明确要求切换语言，否则请始终用中文回答。"
        )
    # Fallback: be explicit but neutral
    return (
        f"You are a helpful assistant. Respond in {lang} unless the user explicitly asks otherwise."
    )


def _init_messages_with_system() -> List[dict]:
    return [{"role": "system", "content": _default_system_prompt()}]


def _set_language_system_message(messages: List[dict], lang: str) -> None:
    os.environ["DS_LANG"] = lang
    sys_msg = {"role": "system", "content": _default_system_prompt()}
    if messages and messages[0].get("role") == "system":
        messages[0] = sys_msg
    else:
        messages.insert(0, sys_msg)


def _endswith(seq: List[int], suffix: List[int]) -> bool:
    if len(suffix) == 0 or len(seq) < len(suffix):
        return False
    return seq[-len(suffix):] == suffix


# sample:
# Samples a single token from the probability distribution derived from logits.
# It applies temperature scaling to smooth or sharpen the distribution.
# Uses a sampling trick equivalent to Gumbel-Max: `argmax(probs / Exp(1))` to sample efficiently without explicit log-softmax computation.
def sample(logits, temperature: float = 1.0):
    """
    Samples a token from the logits using temperature scaling.

    Args:
        logits (torch.Tensor): The logits tensor for token predictions.
        temperature (float, optional): Temperature for scaling logits. Defaults to 1.0.

    Returns:
        torch.Tensor: The sampled token.
    """
    logits = logits / max(temperature, 1e-5)
    probs = torch.softmax(logits, dim=-1, dtype=torch.float32)
    return probs.div_(torch.empty_like(probs).exponential_(1)).argmax(dim=-1)


# generate:
# The core autoregressive generation loop.
# It initializes the token buffer with the prompt and iterates `max_new_tokens` times.
# In each step, it runs the model forward, samples the next token (or greedily selects if temp=0), and updates the buffer.
# Handles EOS tokens and updates the generation status for each sequence in the batch.
@torch.inference_mode()
def generate(
    model: Transformer,
    prompt_tokens: List[List[int]],
    max_new_tokens: int,
    eos_id: int,
    temperature: float = 1.0,
    stop_token_seqs: List[List[int]] | None = None,
    tokenizer: Any = None,
) -> List[List[int]]:
    """
    Generates new tokens based on the given prompt tokens using the specified model.

    Args:
        model (Transformer): The transformer model used for token generation.
        prompt_tokens (List[List[int]]): A list of lists containing the prompt tokens for each sequence.
        max_new_tokens (int): The maximum number of new tokens to generate.
        eos_id (int): The end-of-sequence token ID.
        temperature (float, optional): The temperature value for sampling. Defaults to 1.0.
        stop_token_seqs (List[List[int]] | None, optional): List of token sequences that stop generation. Defaults to None.
        tokenizer (Any, optional): Tokenizer for decoding tokens (used for CSV logging). Defaults to None.

    Returns:
        List[List[int]]: A list of lists containing the generated tokens for each sequence.
    """
    prompt_lens = [len(t) for t in prompt_tokens]
    assert max(prompt_lens) <= model.max_seq_len, f"Prompt length exceeds model maximum sequence length (max_seq_len={model.max_seq_len})"
    total_len = min(model.max_seq_len, max_new_tokens + max(prompt_lens))
    tokens = torch.full((len(prompt_tokens), total_len), -1, dtype=torch.long, device="cuda")
    for i, t in enumerate(prompt_tokens):
        tokens[i, :len(t)] = torch.tensor(t, dtype=torch.long, device="cuda")

    prev_pos = 0
    finished = torch.tensor([False] * len(prompt_tokens), device="cuda")
    prompt_mask = tokens != -1

    # Track generated tokens per sequence to do suffix checks without decoding
    generated: List[List[int]] = [[] for _ in range(len(prompt_tokens))]
    stop_token_seqs = stop_token_seqs or []

    # List to store surprise data: (batch_idx, pos, token_id, token_raw, token_str, surprise)
    surprise_data = []

    for cur_pos in range(min(prompt_lens), total_len):
        if prev_pos == 0:
            # First step (Prefill): request full logits to calculate surprise for prompt prefix
            logits = model.forward(tokens[:, prev_pos:cur_pos], prev_pos, return_full_logits=True)
            
            # Calculate surprise for the prompt prefix: tokens[:, 1:cur_pos]
            # logits[:, :-1, :] predicts tokens[:, 1:cur_pos]
            
            # Record the first token (index 0) for each batch element
            for b in range(len(prompt_tokens)):
                token_id_0 = int(tokens[b, 0].item())
                token_raw_0 = tokenizer.convert_ids_to_tokens([token_id_0])[0] if tokenizer else str(token_id_0)
                token_str_0 = tokenizer.decode([token_id_0]) if tokenizer else str(token_id_0)
                surprise_data.append((b, 0, token_id_0, token_raw_0, token_str_0, 0.0))

            if cur_pos > 1:
                # Targets: tokens starting from index 1 up to cur_pos-1
                targets_prefix = tokens[:, 1:cur_pos] # Shape [B, L-1]
                logits_prefix = logits[:, :-1, :] # Shape [B, L-1, V]
                
                # Calculate negative log likelihood (surprise)
                # Flatten for cross_entropy
                B, L_minus_1, V = logits_prefix.shape
                flat_logits = logits_prefix.reshape(-1, V)
                flat_targets = targets_prefix.reshape(-1)
                
                # Using reduction='none' to get per-token loss
                prefix_surprises = F.cross_entropy(flat_logits, flat_targets, reduction='none').view(B, L_minus_1)
                
                # Store
                for b in range(B):
                    for t_idx in range(L_minus_1):
                        token_id = int(targets_prefix[b, t_idx].item())
                        pos = t_idx + 1 # 1-based index
                        surp = float(prefix_surprises[b, t_idx].item())
                        token_raw = tokenizer.convert_ids_to_tokens([token_id])[0] if tokenizer else str(token_id)
                        token_str = tokenizer.decode([token_id]) if tokenizer else str(token_id)
                        surprise_data.append((b, pos, token_id, token_raw, token_str, surp))
            
            # For the continuation of the loop, use the last logit
            logits = logits[:, -1, :] # [B, V]
        else:
            logits = model.forward(tokens[:, prev_pos:cur_pos], prev_pos)
        
        # Now logits is [B, V], predicting token at cur_pos
        
        if temperature > 0:
            next_token = sample(logits, temperature)
        else:
            next_token = logits.argmax(dim=-1)

        # Respect prompt tokens where prompt_mask is True
        next_token = torch.where(prompt_mask[:, cur_pos], tokens[:, cur_pos], next_token)
        tokens[:, cur_pos] = next_token

        # Calculate surprise for next_token
        # S(t) = -log P(x_t | x_<t)
        probs = torch.softmax(logits, dim=-1)
        token_probs = probs.gather(1, next_token.unsqueeze(1)).squeeze(1) # [B]
        token_surprises = -torch.log(token_probs + 1e-10)
        
        for b in range(len(prompt_tokens)):
            if finished[b].item() and not prompt_mask[b, cur_pos].item():
                continue
            
            tok_id = int(next_token[b].item())
            surp = float(token_surprises[b].item())
            token_raw = tokenizer.convert_ids_to_tokens([tok_id])[0] if tokenizer else str(tok_id)
            token_str = tokenizer.decode([tok_id]) if tokenizer else str(tok_id)
            surprise_data.append((b, cur_pos, tok_id, token_raw, token_str, surp))

        # Update per-sample generated lists & stop conditions
        for i in range(len(prompt_tokens)):
            if finished[i].item():
                continue
            if prompt_mask[i, cur_pos].item():
                # Still inside the original prompt; don't treat as generated
                continue

            tok = int(next_token[i].item())
            generated[i].append(tok)

            # EOS stop
            if tok == eos_id:
                finished[i] = True
                continue

            # Stop-sequence stop (e.g. "\n\nUser:")
            if stop_token_seqs:
                for stop_seq in stop_token_seqs:
                    if _endswith(generated[i], stop_seq):
                        # Mark finished and remove the stop sequence from the output
                        finished[i] = True
                        del generated[i][-len(stop_seq):]
                        break

        prev_pos = cur_pos
        if finished.all():
            break

    # Write surprise data to CSV (Rank 0 only)
    if not dist.is_initialized() or dist.get_rank() == 0:
        csv_file = "surprise_values.csv"
        file_exists = os.path.isfile(csv_file)
        with open(csv_file, 'a', newline='', encoding='utf-8') as f:
            writer = csv.writer(f)
            if not file_exists:
                writer.writerow(["BatchID", "Pos", "TokenID", "RawToken", "DecodedToken", "Surprise"])
            writer.writerows(surprise_data)
        if surprise_data:
            print(f"[SurpriseRecorder] Saved {len(surprise_data)} records to {csv_file}")

    # Return generated (already stripped prompt + stop seq), clipped to max_new_tokens
    completion_tokens: List[List[int]] = []
    for i in range(len(prompt_tokens)):
        out = generated[i][:max_new_tokens]
        # If EOS somehow included (should be stripped by logic above, but double check), strip it
        if eos_id in out:
            out = out[:out.index(eos_id)]
        completion_tokens.append(out)

    return completion_tokens


def _get_env_int(name: str, default: int) -> int:
    val = os.getenv(name)
    if val is None:
        return default
    try:
        return int(val)
    except ValueError:
        raise ValueError(f"Invalid int for {name}={val!r}")


def _build_max_memory_int_keys(gpu_gib: int, cpu_gib: int, disk_gib: int, device_count: int) -> dict:
    mm: dict = {"cpu": f"{cpu_gib}GiB"}
    # Note: 'disk' key causes RuntimeError in some torch versions during validation.
    # Accelerate will automatically spill to 'offload_folder' if CPU is full.
    for i in range(device_count):
        mm[i] = f"{gpu_gib}GiB"
    return mm


def _load_with_accelerate_offload(**kwargs):
    """
    Call load_checkpoint_and_dispatch, enabling offload_state_dict if supported.
    This reduces peak GPU memory during load for very large checkpoints.
    Retries with different max_memory key formats if accelerate complains.
    """
    sig = inspect.signature(load_checkpoint_and_dispatch)
    if "offload_state_dict" in sig.parameters:
        kwargs.setdefault("offload_state_dict", True)

    # Some accelerate versions only accept integer device keys for max_memory,
    # others accept/expect "cuda:0" strings. We'll retry with the other format.
    try:
        return load_checkpoint_and_dispatch(**kwargs)
    except ValueError as e:
        msg = str(e)
        mm = kwargs.get("max_memory")
        if not isinstance(mm, dict):
            raise

        # If we used cuda:* keys and this accelerate requires integers.
        if "available devices are integers" in msg or "Device cuda:" in msg:
            # Retry by stripping cuda:* keys and mapping them to int keys.
            mm2: dict[str | int, Any] = {}
            for k, v in mm.items():
                if isinstance(k, str) and k.startswith("cuda:"):
                    idx = int(k.split(":", 1)[1])
                    mm2[idx] = v
                elif k == "cpu" or k == "disk" or isinstance(k, int):
                    mm2[k] = v
            kwargs2 = deepcopy(kwargs)
            kwargs2["max_memory"] = mm2
            return load_checkpoint_and_dispatch(**kwargs2)

        # If we used int keys and this accelerate doesn't recognize them (rare).
        if "Device 0 is not recognized" in msg or "Device 1 is not recognized" in msg:
            mm2 = {}
            for k, v in mm.items():
                if isinstance(k, int):
                    mm2[f"cuda:{k}"] = v
                elif k == "cpu" or k == "disk" or (isinstance(k, str) and k.startswith("cuda:")):
                    mm2[k] = v
            kwargs2 = deepcopy(kwargs)
            kwargs2["max_memory"] = mm2
            return load_checkpoint_and_dispatch(**kwargs2)

        raise


# main:
# Entry point for the distributed generation script.
# It initializes the distributed process group, loads model arguments and the sharded checkpoint for the local rank.
# Supports two modes: "interactive" (chat loop with user input) and "batch" (processing prompts from a file).
# Handles synchronization of inputs across ranks to ensure all processes generate for the same prompt.
def main(
    ckpt_path: str,
    config: str,
    input_file: str = "Lorem ipsum",
    interactive: bool = True,
    max_new_tokens: int = 100,
    temperature: float = 1.0,
) -> None:
    """
    Main function to load the model and perform interactive or batch text generation.

    Args:
        ckpt_path (str): Path to the model checkpoint directory.
        config (str): Path to the model configuration file.
        input_file (str, optional): Path to a file containing input prompts. Defaults to "".
        interactive (bool, optional): Whether to run in interactive mode. Defaults to True.
        max_new_tokens (int, optional): Maximum number of new tokens to generate. Defaults to 100.
        temperature (float, optional): Temperature for sampling. Defaults to 1.0.
    """
    world_size = int(os.getenv("WORLD_SIZE", "1"))
    rank = int(os.getenv("RANK", "0"))
    local_rank = int(os.getenv("LOCAL_RANK", "0"))
    if world_size > 1:
        dist.init_process_group("nccl")
    global print
    if rank != 0:
        print = lambda *_, **__: None
    torch.cuda.set_device(local_rank)

    # Some clusters still inject the deprecated env var; drop it to avoid warnings/confusion.
    os.environ.pop("PYTORCH_CUDA_ALLOC_CONF", None)

    torch.set_default_dtype(torch.bfloat16)
    torch.set_num_threads(8)
    torch.manual_seed(33377335)
    with open(config) as f:
        args = ModelArgs(**json.load(f))
    print(args)

    # Offloading logic
    # IMPORTANT: leave headroom on the GPU; "auto" likes to pack tight and can OOM on tiny transient allocations.
    # Tune via env vars if needed (defaults are conservative for 141GiB H200s).
    gpu_gib = _get_env_int("DS_MAX_GPU_GIB", 105)
    cpu_gib = _get_env_int("DS_MAX_CPU_GIB", 450)
    disk_gib = _get_env_int("DS_MAX_DISK_GIB", 2000)
    max_memory = _build_max_memory_int_keys(gpu_gib, cpu_gib, disk_gib, torch.cuda.device_count())

    with init_empty_weights():
        model = Transformer(args)

    # Materialize buffers that are not in the checkpoint (caches, constants)
    # and initialize all other meta buffers to zero
    set_module_tensor_to_device(model, "freqs_cis", "cpu", precompute_freqs_cis(args))
    for name, buf in model.named_buffers():
        if buf.device.type == "meta" and name != "freqs_cis":
            set_module_tensor_to_device(model, name, "cpu", torch.zeros(buf.size(), dtype=buf.dtype))

    os.makedirs("offload", exist_ok=True)
    model = _load_with_accelerate_offload(
        model=model,
        checkpoint=ckpt_path,
        device_map="auto",
        max_memory=max_memory,
        no_split_module_classes=["Block"],
        strict=False,
        offload_folder="offload",
        offload_buffers=True,
    )

    tokenizer = AutoTokenizer.from_pretrained(ckpt_path)
    # Fallback for missing chat template in converted weights
    if tokenizer.chat_template is None:
        print("Warning: No chat_template found in tokenizer config. Using default DeepSeek-style template.")
        tokenizer.chat_template = (
            "{% if messages[0]['role'] == 'system' %}{{ messages[0]['content'] + '\n' }}{% endif %}"
            "{% for message in messages %}"
            "{% if message['role'] == 'user' %}{{ 'User: ' + message['content'] + '\n\n' }}"
            "{% elif message['role'] == 'assistant' %}{{ 'Assistant: ' + message['content'] + eos_token + '\n\n' }}"
            "{% endif %}"
            "{% endfor %}"
            "{% if add_generation_prompt %}{{ 'Assistant: ' }}{% endif %}"
        )
    print("I'm DeepSeek 👋")

    if interactive:
        # Precompute stop sequences for "User:" turns
        stop_strs = ["\n\nUser:", "\nUser:"]
        stop_token_seqs = [tokenizer.encode(s, add_special_tokens=False) for s in stop_strs]
        
        messages = _init_messages_with_system()
        while True:
            if world_size == 1:
                prompt = input(">>> ")
            elif rank == 0:
                prompt = input(">>> ")
                objects = [prompt]
                dist.broadcast_object_list(objects, 0)
            else:
                objects = [None]
                dist.broadcast_object_list(objects, 0)
                prompt = objects[0]
            
            if prompt == "/exit":
                break
            elif prompt == "/clear":
                messages = _init_messages_with_system()
                continue
            elif prompt.startswith("/lang "):
                # Usage: /lang en  OR  /lang zh
                _, _, new_lang = prompt.partition(" ")
                new_lang = new_lang.strip()
                if not new_lang:
                    print("Usage: /lang en|zh|...")
                    continue
                _set_language_system_message(messages, new_lang)
                print(f"(system) Language set to: {new_lang}")
                continue

            messages.append({"role": "user", "content": prompt})
            prompt_tokens = tokenizer.apply_chat_template(messages, add_generation_prompt=True)
            completion_tokens = generate(
                model, 
                [prompt_tokens], 
                max_new_tokens, 
                tokenizer.eos_token_id, 
                temperature,
                stop_token_seqs=stop_token_seqs,
                tokenizer=tokenizer
            )
            completion = tokenizer.decode(completion_tokens[0], skip_special_tokens=True)
            
            # Extra safety: strip trailing "User:" if it leaked through
            for cut in stop_strs:
                if cut in completion:
                    completion = completion.split(cut, 1)[0].rstrip()
            
            print(completion)
            messages.append({"role": "assistant", "content": completion})
    else:
        with open(input_file) as f:
            prompts = f.read().split("\n\n")
        assert len(prompts) <= args.max_batch_size, f"Number of prompts exceeds maximum batch size ({args.max_batch_size})"
        print(f"DEBUG: Processing {len(prompts)} prompts.")
        prompt_tokens = [tokenizer.apply_chat_template([{"role": "user", "content": prompt}], add_generation_prompt=True, tokenize=True) for prompt in prompts]
        
        if len(prompt_tokens) > 0:
            print(f"DEBUG: Type of prompt_tokens[0]: {type(prompt_tokens[0])}")
            print(f"DEBUG: First element of prompt_tokens[0]: {prompt_tokens[0][0] if len(prompt_tokens[0]) > 0 else 'EMPTY'}")
            print(f"DEBUG: Type of first element: {type(prompt_tokens[0][0]) if len(prompt_tokens[0]) > 0 else 'N/A'}")

        completion_tokens = generate(model, prompt_tokens, max_new_tokens, tokenizer.eos_token_id, temperature, tokenizer=tokenizer)
        completions = tokenizer.batch_decode(completion_tokens, skip_special_tokens=True)
        for prompt, completion in zip(prompts, completions):
            print("Prompt:", prompt)
            print("Completion:", completion)
            print()

    if world_size > 1:
        dist.destroy_process_group()


if __name__ == "__main__":
    """
    Command-line interface for distributed text generation.

    Arguments:
        --ckpt-path (str): Path to the model checkpoint directory.
        --config (str): Path to the model configuration file.
        --input-file (str, optional): File containing prompts for batch processing.
        --interactive (bool, optional): Enable interactive mode for generating text.
        --max-new-tokens (int, optional): Maximum number of new tokens to generate. Defaults to 200.
        --temperature (float, optional): Temperature for sampling. Defaults to 0.2.

    Raises:
        AssertionError: If neither input-file nor interactive mode is specified.
    """
    parser = ArgumentParser()
    parser.add_argument("--ckpt-path", type=str, required=True)
    parser.add_argument("--config", type=str, required=True)
    parser.add_argument("--input-file", type=str, default="Lorem ipsum")
    parser.add_argument("--interactive", action="store_true")
    parser.add_argument("--max-new-tokens", type=int, default=200)
    parser.add_argument("--temperature", type=float, default=0.6)
    args = parser.parse_args()
    assert args.input_file or args.interactive, "Either input-file or interactive mode must be specified"
    main(args.ckpt_path, args.config, args.input_file, args.interactive, args.max_new_tokens, args.temperature)
