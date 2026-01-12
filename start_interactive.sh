#!/bin/bash
# Usage: ./start_interactive.sh

echo "Requesting Interactive Session on 'agentS-xlong' ભા"
echo "Specs: 4x H200 GPUs, 16 CPUs, 12 Hours."
echo "Waiting for allocation..."

srun --partition=agentS-xlong \
     --gres=gpu:h200:4 \
     --nodes=1 \
     --ntasks-per-node=1 \
     --cpus-per-task=16 \
     --time=12:00:00 \
     --job-name=ds-dev \
     --pty /bin/bash --login
