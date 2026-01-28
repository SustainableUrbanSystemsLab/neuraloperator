#!/bin/bash
#SBATCH -J fno_test
#SBATCH -N 1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=12
#SBATCH --mem=128G
#SBATCH -t 01:00:00
#SBATCH -o logs/fno_test_%j.out
#SBATCH -e logs/fno_test_%j.err

# Note: --gres=gpu:xxx:N is passed from the deploy_ice.sh script
# CHECKPOINT_DIR is passed via --export
# CONFIG_FILE can be passed

echo "Test Job started on $(date)"
echo "Running on node $(hostname)"
echo "Number of GPUs: ${NUM_GPUS:-1}"
echo "Config file: ${CONFIG_FILE:-config.toml}"
echo "Checkpoint: ${CHECKPOINT_DIR}"

# Go to the directory where sbatch was submitted from
cd $SLURM_SUBMIT_DIR

# Ensure we are at the project root (where pyproject.toml exists)
if [ ! -f "pyproject.toml" ]; then
    echo "pyproject.toml not found in current directory ($PWD). Checking parent..."
    if [ -f "../pyproject.toml" ]; then
        cd ..
        echo "Changed directory to project root: $PWD"
    else
        echo "Error: Could not find pyproject.toml. Dependency sync might fail."
    fi
fi

# Ensure logs directory exists
mkdir -p logs

# CRITICAL: Detect and remove Windows virtual environment if uploaded
if [ -d ".venv/Scripts" ]; then
    echo "Detected Windows .venv (Scripts folder exists). Removing it to create Linux compatible environment..."
    rm -rf .venv
fi

# Install uv if not available
if ! command -v uv &> /dev/null; then
    echo "Installing uv..."
    curl -LsSf https://astral.sh/uv/install.sh | sh
    export PATH="$HOME/.local/bin:$PATH"
fi

# Sync dependencies
echo "Syncing dependencies with uv..."
uv sync --frozen --no-cache

# Config flag
CONFIG_FLAG=""
if [ -n "${CONFIG_FILE}" ]; then
    CONFIG_FLAG="--config ${CONFIG_FILE}"
fi

# Checkpoint flag
if [ -z "${CHECKPOINT_DIR}" ]; then
    echo "Error: CHECKPOINT_DIR environment variable is not set."
    exit 1
fi

# Execute evaluation
# We use single GPU for evaluation usually, unless dataset is huge. 
# But the script supports distributed if needed. 
# Defaults to 1 GPU for testing to keep it simple unless specified.

if [ "${NUM_GPUS:-1}" -gt 1 ]; then
    MASTER_PORT=$((29500 + RANDOM % 1000))
    echo "Starting DISTRIBUTED evaluation with ${NUM_GPUS} GPUs..."
    uv run torchrun --nproc_per_node=${NUM_GPUS} --master_port=${MASTER_PORT} scripts/evaluate_navier_stokes.py ${CONFIG_FLAG} --checkpoint "${CHECKPOINT_DIR}"
else
    echo "Starting SINGLE-GPU evaluation..."
    uv run python scripts/evaluate_navier_stokes.py ${CONFIG_FLAG} --checkpoint "${CHECKPOINT_DIR}"
fi

echo "Test Job finished on $(date)"
