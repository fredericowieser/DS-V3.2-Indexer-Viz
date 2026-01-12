# Source this file inside your interactive session:
# source dev_setup.sh

echo "Setting up Dev Environment..."
export OMP_NUM_THREADS=1
export RECORD_INDEX=1
export UV_HTTP_TIMEOUT=600

# venv setup
if ! command -v uv &> /dev/null;
    echo "Installing uv..."
    curl -LsSf https://astral.sh/uv/install.sh | sh
    source "$HOME/.local/bin/env"
fi

if [ ! -d ".venv" ]; then
    echo "Creating virtual environment..."
    uv venv
fi

echo "Syncing dependencies..."
uv sync
source .venv/bin/activate

# Weights Check
HF_MODEL_REPO="deepseek-ai/DeepSeek-V3.2" 
WEIGHTS_DIR="checkpoints/raw_weights"
CONFIG_PATH="inference/config_671B_v3.2.json"

if [ ! -d "$WEIGHTS_DIR" ] || [ -z "$(ls -A $WEIGHTS_DIR)" ]; then
    echo "Weights missing. Downloading..."
    mkdir -p "$WEIGHTS_DIR"
    huggingface-cli download "$HF_MODEL_REPO" \
        --local-dir "$WEIGHTS_DIR" \
        --local-dir-use-symlinks False \
        --include "*.safetensors" "*.json"
else
    echo "Weights found at $WEIGHTS_DIR"
fi

echo "---------------------------------------------------"
echo "Environment Ready!"
echo ""
echo "To run your model:"
echo "python inference/generate.py --ckpt-path $WEIGHTS_DIR --config $CONFIG_PATH --interactive --max-new-tokens 1 --temperature 0.0"
echo "---------------------------------------------------"
