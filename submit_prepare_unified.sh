#!/bin/bash
# Build data/train_unified.jsonl + data/val_unified.jsonl for the
# 4-class door+glass adapter. Runs on a GPU node only so the JSONL
# writes land on the same NFS client that the fine-tune job will read
# from.
#
# Usage:
#   sbatch submit_prepare_unified.sh
#   sbatch submit_prepare_unified.sh --door-target 12000

#SBATCH --job-name=prep-unified
#SBATCH --partition=teaching
#SBATCH --gres=gpu:1
#SBATCH --mem=8G
#SBATCH --cpus-per-task=2
#SBATCH --time=0:20:00
#SBATCH --output=/home/ad.msoe.edu/celliot/Digital-twin-v2/isaac-sim-msoe/logs/prep_unified_%j.out
#SBATCH --error=/home/ad.msoe.edu/celliot/Digital-twin-v2/isaac-sim-msoe/logs/prep_unified_%j.out

set -euo pipefail
cd /home/ad.msoe.edu/celliot/Digital-twin-v2/isaac-sim-msoe

PYTHON="/home/ad.msoe.edu/celliot/.conda/envs/vlm/bin/python"
echo "Started: $(date)  on $(hostname)"
echo ""

$PYTHON simulations/robot/prepare_unified_data.py "$@"

echo ""
echo "Finished: $(date)"
