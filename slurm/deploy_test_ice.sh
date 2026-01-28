#!/bin/bash

# ICE Evaluation Deployment Wrapper
# Usage: bash slurm/deploy_test_ice.sh --checkpoint ./ckpt [--gpu H100] [--config config.toml]

GPU_TYPE="h200"
NUM_GPUS=1
CONFIG_FILE="config/navier_stokes.toml"
CHECKPOINT_DIR=""

# Parse arguments
while [[ "$#" -gt 0 ]]; do
    case $1 in
        --gpu) GPU_TYPE="$2"; shift ;;
        --ngpus) NUM_GPUS="$2"; shift ;;
        --config) CONFIG_FILE="$2"; shift ;;
        --checkpoint) CHECKPOINT_DIR="$2"; shift ;;
        *) echo "Unknown parameter passed: $1"; exit 1 ;;
    esac
    shift
done

if [ -z "$CHECKPOINT_DIR" ]; then
    echo "Error: --checkpoint argument is required."
    echo "Usage: bash slurm/deploy_test_ice.sh --checkpoint path/to/ckpt"
    exit 1
fi

# Convert to lowercase
GPU_TYPE_LOWER=$(echo "$GPU_TYPE" | tr '[:upper:]' '[:lower:]')

# Map to correct ICE gres names
case $GPU_TYPE_LOWER in
    h200) SLURM_GPU="H200:${NUM_GPUS}" ;;
    h100) SLURM_GPU="H100:${NUM_GPUS}" ;;
    a100) SLURM_GPU="A100:${NUM_GPUS}" ;;
    v100) SLURM_GPU="V100:${NUM_GPUS}" ;;
    *) SLURM_GPU="${GPU_TYPE}:${NUM_GPUS}" ;;
esac

# Read ICE config from config.toml
if [ -f "$CONFIG_FILE" ]; then
    ICE_ACCOUNT=$(sed -n '/^\[ice\]/,/^\[/p' "$CONFIG_FILE" | grep -E "^account\s*=" | sed 's/.*=\s*"\(.*\)".*/\1/' | tr -d ' ')
    ICE_PARTITION=$(sed -n '/^\[ice\]/,/^\[/p' "$CONFIG_FILE" | grep -E "^partition\s*=" | sed 's/.*=\s*"\(.*\)".*/\1/' | tr -d ' ')
else
    ICE_ACCOUNT="coa"
    ICE_PARTITION=""
fi

if [ -z "$ICE_ACCOUNT" ]; then ICE_ACCOUNT="coa"; fi

echo "=========================================="
echo " Preparing ICE Evaluation"
echo " GPU Requested: $GPU_TYPE x $NUM_GPUS ($SLURM_GPU)"
echo " Config file: $CONFIG_FILE"
echo " Checkpoint: $CHECKPOINT_DIR"
echo " Account: $ICE_ACCOUNT"
echo "=========================================="

mkdir -p logs

# Export variables so sbatch can see them
export NUM_GPUS
export CONFIG_FILE
export CHECKPOINT_DIR

SBATCH_CMD="sbatch --gres=gpu:$SLURM_GPU --account=$ICE_ACCOUNT --export=ALL,NUM_GPUS=$NUM_GPUS,CONFIG_FILE=$CONFIG_FILE,CHECKPOINT_DIR=$CHECKPOINT_DIR"

if [ -n "$ICE_PARTITION" ]; then
    SBATCH_CMD="$SBATCH_CMD --partition=$ICE_PARTITION"
fi

SBATCH_CMD="$SBATCH_CMD slurm/test_ice.sbatch"

# Submit
$SBATCH_CMD

echo "------------------------------------------"
echo "Evaluation job submitted. Check status with: squeue -u $USER"
echo "Output will be in logs/fno_test_JOBID.out"
