#!/bin/bash
# submit_prepare_data.sh — SLURM job to generate JPEGs + train/val JSONL
# from the 8 training-role production-collection shards (v4).
#
# Runs on a compute node (dgx) to avoid NFS cache issues with JSONL writes.
# After this completes, data/finetune_images_v4/ and data/train_v4.jsonl +
# data/val_v4.jsonl are ready for baseline eval and fine-tuning. Holdout
# shards (door_production_v4_holdout_*.npz) are excluded by the default
# shard glob and handled separately by the eval pipeline.
#
# Usage:  sbatch submit_prepare_data.sh

#SBATCH --job-name=prep-v4
#SBATCH --partition=dgx
#SBATCH --gres=gpu:1
#SBATCH --mem=32G
#SBATCH --cpus-per-task=4
#SBATCH --time=1:00:00
#SBATCH --output=/home/ad.msoe.edu/celliot/Digital-twin-v2/isaac-sim-msoe/logs/prep_data_%j.out
#SBATCH --error=/home/ad.msoe.edu/celliot/Digital-twin-v2/isaac-sim-msoe/logs/prep_data_%j.out

set -euo pipefail

cd "/home/ad.msoe.edu/celliot/Digital-twin-v2/isaac-sim-msoe"

echo "=== Prepare Fine-tune Data (v4, 8 training doors) ==="
echo "Started: $(date)"
echo ""

SHARD_COUNT=$(ls data/door_production_v4_train_Doors_*.npz 2>/dev/null | wc -l)
echo "Production shards found: $SHARD_COUNT"
if [ "$SHARD_COUNT" -eq 0 ]; then
    echo "ERROR: no shards. Run submit_production_collection.sh first."
    exit 1
fi

PYTHON="/home/ad.msoe.edu/celliot/.conda/envs/vlm/bin/python"
echo "Python: $($PYTHON --version)"
echo ""

$PYTHON simulations/robot/prepare_finetune_data.py
EXIT=$?

echo ""
echo "Exit code: $EXIT"
echo "Finished: $(date)"
exit $EXIT
