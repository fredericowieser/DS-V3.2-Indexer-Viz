#!/bin/bash --login
#SBATCH --partition=agentS-xlong
#SBATCH --gres=gpu:h200:4
#SBATCH --job-name=fred_dev
#SBATCH --time=5-00:00:00

echo "Setting up environment for GPU training..."

# Set OMP_NUM_THREADS to 1 for efficiency with torch
export OMP_NUM_THREADS=1

# Install uv (if not already installed)
if ! command -v uv &> /dev/null
then
    echo "uv could not be found, installing it now..."
    curl -LsSf https://astral.sh/uv/install.sh | sh
    # Source the environment to add uv to the PATH for the current session
    source "$HOME/.local/bin/env"
fi

# Create a .venv local virtual environment (if it doesn't exist)
if [ ! -d ".venv" ]
then
    echo "Creating virtual environment..."
    # creates a virtual environment in the current directory.
    uv venv
fi

echo "Installing dependencies with uv..."
export UV_HTTP_TIMEOUT=600
uv sync
source .venv/bin/activate


# Configuration
HF_MODEL_REPO="deepseek-ai/DeepSeek-V3.2"  # The repo you requested
RAW_WEIGHTS_DIR="checkpoints/raw_weights"
CONVERTED_WEIGHTS_DIR="checkpoints/converted_weights_tp4"

echo "Model Setup Stage..."

# A. Download from Hugging Face
if [ ! -d "$RAW_WEIGHTS_DIR" ] || [ -z "$(ls -A $RAW_WEIGHTS_DIR)" ]; then
    echo "Downloading $HF_MODEL_REPO from Hugging Face..."
    echo "Saving to: $RAW_WEIGHTS_DIR"
    mkdir -p "$RAW_WEIGHTS_DIR"
    
    # Download the repository (excluding optimizer states if any to save space)
    huggingface-cli download "$HF_MODEL_REPO" \
        --local-dir "$RAW_WEIGHTS_DIR" \
        --local-dir-use-symlinks False \
        --exclude "*.pt" "*.bin" "optimizer*"
else
    echo "Raw weights found in $RAW_WEIGHTS_DIR. Skipping download."
fi

# B. Convert Weights for 4 GPUs
# The inference code requires the model to be sharded specifically for the GPU count.
if [ ! -d "$CONVERTED_WEIGHTS_DIR" ] || [ -z "$(ls -A $CONVERTED_WEIGHTS_DIR)" ]; then
    echo "Converting weights for 4-GPU Tensor Parallelism..."
    echo "Saving to: $CONVERTED_WEIGHTS_DIR"
    mkdir -p "$CONVERTED_WEIGHTS_DIR"

    # Run the conversion script included in the repo
    # --n-experts 256 is standard for V3.2 architectures
    # --model-parallel 4 matches your 4-GPU allocation
    python inference/convert.py \
        --hf-ckpt-path "$RAW_WEIGHTS_DIR" \
        --save-path "$CONVERTED_WEIGHTS_DIR" \
        --n-experts 256 \
        --model-parallel 4
else
    echo "Converted weights found in $CONVERTED_WEIGHTS_DIR. Skipping conversion."
fi

echo "Starting GPU run..."

CONFIG_PATH="inference/config_671B_v3.2.json"

# Run with torchrun on 4 GPUs
torchrun --nproc_per_node=4 main.py \
    --ckpt-path "$CONVERTED_CKPT_DIR" \
    --config "$CONFIG_PATH"