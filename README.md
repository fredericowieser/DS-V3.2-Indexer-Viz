# DeepSeek-V3.2 Indexer Head Visualiser

The goal of this repo is to edit the original DeepSeek code in order to add a hook into the model's forward pass such that we can record how the trained model retrieves tokens.

## Update

- 2025.11.17: **We have identified that previous versions of the inference demo code contained an implementation discrepancy in Rotary Position Embedding (RoPE) within the indexer module, potentially leading to degraded model performance.** Specifically, the input tensor to RoPE in the indexer module requires a non-interleaved layout, whereas RoPE in the MLA module expects an interleaved layout. This issue has now been resolved. Please refer to the updated version of the inference demo code and take note of this implementation detail.

## Open-Source Kernels

For TileLang kernels with **better readability and research-purpose design**, please refer to [TileLang](https://github.com/tile-ai/tilelang/tree/main/examples/deepseek_v32).

For **high-performance CUDA kernels**, indexer logit kernels (including paged versions) are available in [DeepGEMM](https://github.com/deepseek-ai/DeepGEMM/pull/200). Sparse attention kernels are released in [FlashMLA](https://github.com/deepseek-ai/FlashMLA/pull/98).



## How to Run Locally

### HuggingFace
We provide an updated inference demo code in the [inference](https://huggingface.co/deepseek-ai/DeepSeek-V3.2-Exp/tree/main/inference) folder to help the community quickly get started with our model and understand its architectural details.

First convert huggingface model weights to the the format required by our inference demo. Set `MP` to match your available GPU count:
```bash
cd inference
export EXPERTS=256
python convert.py --hf-ckpt-path ${HF_CKPT_PATH} --save-path ${SAVE_PATH} --n-experts ${EXPERTS} --model-parallel ${MP}
```

Launch the interactive chat interface and start exploring DeepSeek's capabilities:
```bash
export CONFIG=config_671B_v3.2.json
torchrun --nproc-per-node ${MP} generate.py --ckpt-path ${SAVE_PATH} --config ${CONFIG} --interactive
```

### SGLang

#### Installation with Docker

```
# H200
docker pull lmsysorg/sglang:dsv32

# MI350
docker pull lmsysorg/sglang:dsv32-rocm

# NPUs
docker pull lmsysorg/sglang:dsv32-a2
docker pull lmsysorg/sglang:dsv32-a3
```

#### Launch Command
```bash
python -m sglang.launch_server --model deepseek-ai/DeepSeek-V3.2-Exp --tp 8 --dp 8 --enable-dp-attention
```

### vLLM

vLLM provides day-0 support of DeepSeek-V3.2-Exp. See the [recipes](https://docs.vllm.ai/projects/recipes/en/latest/DeepSeek/DeepSeek-V3_2-Exp.html) for up-to-date details.

## License

This repository and the model weights are licensed under the [MIT License](LICENSE).

## Citation

```
@misc{deepseekai2024deepseekv32,
      title={DeepSeek-V3.2-Exp: Boosting Long-Context Efficiency with DeepSeek Sparse Attention}, 
      author={DeepSeek-AI},
      year={2025},
}
```

## Contact

If you have any questions, please raise an issue or contact us at [service@deepseek.com](service@deepseek.com).
