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
echo "Starting GPU run..."

# Define paths (Update these if your paths differ!)
CKPT_PATH="/path/to/your/checkpoints" # REPLACE THIS
CONFIG_PATH="inference/config_671B_v3.2.json"

# Run with torchrun on 4 GPUs
torchrun --nproc_per_node=4 main.py \
    --ckpt-path "$CKPT_PATH" \
    --config "$CONFIG_PATH"