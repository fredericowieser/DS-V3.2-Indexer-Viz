#!/bin/bash
#SBATCH --job-name=deepseek_dev
#SBATCH --partition=agentS-xlong
#SBATCH --gres=gpu:h200:4
#SBATCH --output=slurm_out/%j.out
#SBATCH --time=5-00:00:00

set -e

# Ensure the output directory exists
mkdir -p slurm_out

CLI_PATH="${HOME}/vscode_cli"
echo "CLI_PATH: ${CLI_PATH}"

# Install the VS Code CLI command if it doesn't exist
if [[ ! -e ${CLI_PATH}/code ]]; then
  echo "Downloading and installing the VS Code CLI command"
  mkdir -p "${HOME}/vscode_cli"
  pushd "${HOME}/vscode_cli"
  # Process from: https://code.visualstudio.com/docs/remote/tunnels#_using-the-code-cli
  curl -Lk 'https://code.visualstudio.com/sha/download?build=stable&os=cli-alpine-x64' --output vscode_cli.tar.gz
  # unpack the code binary file
  tar -xf vscode_cli.tar.gz
  # clean-up
  rm vscode_cli.tar.gz
  popd
fi

echo "Starting VS Code Tunnel..."
echo "Check the output file in slurm_out/ for the login code/link."

# run the code tunnel command and accept the licence
srun ${CLI_PATH}/code tunnel --accept-server-license-terms
