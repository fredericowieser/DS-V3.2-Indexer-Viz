#!/bin/bash --login
#SBATCH --partition=agentS-xlong
#SBATCH --gres=gpu:h200:4
#SBATCH --job-name=deepseek_viz
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=16
#SBATCH --time=5-00:00:00

echo "Setting up environment for DeepSeek-V3.2 Offloading..."

# We set this to 1 to prevent MKL/OpenMP contention with PyTorch
export OMP_NUM_THREADS=1
# Enable the recording logic we added to model.py
export RECORD_INDEX=1

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
HF_MODEL_REPO="deepseek-ai/DeepSeek-V3.2" 
WEIGHTS_DIR="checkpoints/raw_weights"
CONFIG_PATH="inference/config_671B_v3.2.json"

# A. Download Weights
# Note: We prioritize .safetensors for use with load_checkpoint_and_dispatch
if [ ! -d "$WEIGHTS_DIR" ] || [ -z "$(ls -A $WEIGHTS_DIR)" ]; then
    echo "Downloading model weights..."
    mkdir -p "$WEIGHTS_DIR"
    huggingface-cli download "$HF_MODEL_REPO" \
        --local-dir "$WEIGHTS_DIR" \
        --local-dir-use-symlinks False \
        --include "*.safetensors" "*.json"
else
    echo "Weights found. Skipping download."
fi

# IMPORTANT: We use 'python' NOT 'torchrun'. 
# Accelerate's device_map="auto" handles all 4 GPUs within this single process.
# This allows us to shard the 671B model across 4xVRAM + System RAM.

echo "Running inference..."
python inference/generate.py \
    --ckpt-path "$WEIGHTS_DIR" \
    --config "$CONFIG_PATH" \
    --interactive \
    --max-new-tokens 1 \
    --temperature 0.0