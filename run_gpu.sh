#!/bin/bash --login
#SBATCH --partition=agentS-xlong
#SBATCH --gres=gpu:h200:4
#SBATCH --job-name=fred_dev
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=16
#SBATCH --mem=0
#SBATCH --time=5-00:00:00

# Set up CUDA environment for JIT compilation (TileLang/TVM)
export PATH=/usr/local/cuda-12.8/bin:$PATH
export LD_LIBRARY_PATH=/usr/local/cuda-12.8/lib64:$LD_LIBRARY_PATH
export CUDA_HOME=/usr/local/cuda-12.8
# Force TileLang to use sm_90 to avoid auto-detection crash on H200 (sm_90a)
export TILELANG_TARGET="cuda -arch=sm_90"
# Option to disable TileLang kernels completely and fallback to PyTorch
export DS_DEV_MODE=1

echo "Setting up environment for DeepSeek-V3.2 Offloading..."

# We set this to 1 to prevent MKL/OpenMP contention with PyTorch
export OMP_NUM_THREADS=1
# Enable Recording
export RECORD_VECTORS=1
export RECORD_SCORES=1
export RECORD_OUT_DIR="logs/vectors_and_scores"
# Optimize PyTorch memory allocation
export PYTORCH_ALLOC_CONF=expandable_segments:True
# Limit CPU memory (now increased since we requested full node memory)
export DS_MAX_CPU_GIB=250
# Safe GPU memory limit (leaves headroom for buffers)
export DS_MAX_GPU_GIB=115

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

# A. Check for converted weights (the final product we need)
CONVERTED_WEIGHTS_DIR="checkpoints/converted_weights"
# Check if directory exists and contains at least one .safetensors file
if [ ! -d "$CONVERTED_WEIGHTS_DIR" ] || [ -z "$(find "$CONVERTED_WEIGHTS_DIR" -maxdepth 1 -name "*.safetensors" -print -quit)" ]; then
    # Converted weights missing, check if raw weights exist
    if [ ! -d "$WEIGHTS_DIR" ] || [ -z "$(find "$WEIGHTS_DIR" -maxdepth 1 -name "*.safetensors" -print -quit)" ]; then
        echo "Raw weights not found. Downloading model weights..."
        mkdir -p "$WEIGHTS_DIR"
        python -c "from huggingface_hub import snapshot_download; snapshot_download(repo_id='$HF_MODEL_REPO', local_dir='$WEIGHTS_DIR', local_dir_use_symlinks=False, allow_patterns=['*.safetensors', '*.json'])" || { echo "Download failed"; exit 1; }
    else
        echo "Raw weights found. Skipping download."
    fi
    
    # B. Convert weights
    echo "Converting weights to local format..."
    mkdir -p "$CONVERTED_WEIGHTS_DIR"
    python inference/convert.py \
        --hf-ckpt-path "$WEIGHTS_DIR" \
        --save-path "$CONVERTED_WEIGHTS_DIR" \
        --n-experts 256 \
        --model-parallel 1
else
    echo "Converted weights found. Ready to run inference."
fi

# IMPORTANT: We use 'python' NOT 'torchrun'. 
# Accelerate's device_map="auto" handles all 4 GPUs within this single process.
# This allows us to shard the 671B model across 4xVRAM + System RAM.

echo "Running inference..."
python inference/generate.py \
    --ckpt-path "$CONVERTED_WEIGHTS_DIR" \
    --config "$CONFIG_PATH" \
    --input-file "EM-LLM.txt" \
    --max-new-tokens 1 \
    --temperature 0.0