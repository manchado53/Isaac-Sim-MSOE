#!/bin/bash
# submit_finetune_job_dgx.sh — V100 fallback for Qwen2-VL LoRA fine-tuning.
#
# Same pipeline as submit_finetune_job.sh but targets the dgx partition
# (V100, 32GB) and writes to a distinct output dir + log name so it can
# run in parallel with the H100 job without colliding.
#
# Outputs:
#   data/qwen2vl_door_lora_dgx/   (adapter)
#   logs/finetune_dgx_<JOBID>.out (training log)
#
# Usage:
#   sbatch submit_finetune_job_dgx.sh
#   sbatch submit_finetune_job_dgx.sh --epochs 1   # args forwarded to finetune_qwen2vl.py

#SBATCH --job-name=qwen2vl-ft-dgx
#SBATCH --partition=dgx
#SBATCH --gres=gpu:1
#SBATCH --mem=32G
#SBATCH --cpus-per-task=4
#SBATCH --time=21:00:00
#SBATCH --output=/home/ad.msoe.edu/celliot/Digital-twin-v2/isaac-sim-msoe/logs/finetune_dgx_%j.out
#SBATCH --error=/home/ad.msoe.edu/celliot/Digital-twin-v2/isaac-sim-msoe/logs/finetune_dgx_%j.out

set -euo pipefail

cd "/home/ad.msoe.edu/celliot/Digital-twin-v2/isaac-sim-msoe"

echo "=== Qwen2-VL Fine-tune (DGX / V100 fallback) ==="
echo "Node:    $(hostname)"
echo "Started: $(date)"
echo "GPU:     $(nvidia-smi --query-gpu=name --format=csv,noheader 2>/dev/null || echo 'unknown')"
echo ""

PYTHON="/home/ad.msoe.edu/celliot/.conda/envs/vlm/bin/python"
if [ ! -f "$PYTHON" ]; then
    echo "ERROR: vlm conda env not found at $PYTHON"
    exit 1
fi
echo "Python: $($PYTHON --version)"
echo ""

# Check production shards exist
SHARD_COUNT=$(ls data/door_production_Doors_*.npz 2>/dev/null | wc -l)
if [ "$SHARD_COUNT" -eq 0 ]; then
    echo "ERROR: no production shards found (data/door_production_Doors_*.npz)."
    echo "  Run submit_production_collection.sh first."
    exit 1
fi
echo "Found $SHARD_COUNT production shards"

# Regenerate train/val splits on this compute node to avoid NFS cache issues.
echo "--- Generating data splits ---"
$PYTHON simulations/robot/prepare_finetune_data.py
echo ""

TRAIN_N=$(wc -l < data/train.jsonl)
VAL_N=$(wc -l < data/val.jsonl)
echo "Dataset:  train=$TRAIN_N  val=$VAL_N"
echo ""

echo "--- Fine-tuning (output: data/qwen2vl_door_lora_dgx) ---"
$PYTHON simulations/robot/finetune_qwen2vl.py \
    --output_dir data/qwen2vl_door_lora_dgx \
    "$@"
FT_EXIT=$?

echo ""
echo "Fine-tune exit code: $FT_EXIT"
echo "Finished: $(date)"
