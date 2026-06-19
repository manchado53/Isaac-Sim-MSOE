#!/bin/bash
# 4-class eval for the unified door+glass adapter.
#
# Usage:
#   sbatch submit_eval_unified.sh                                   # val_unified
#   sbatch submit_eval_unified.sh --test_jsonl data/other.jsonl     # custom set
#   sbatch submit_eval_unified.sh --adapter ""                      # zero-shot baseline

#SBATCH --job-name=qwen2vl-eval-u
#SBATCH --partition=dgx
#SBATCH --gres=gpu:1
#SBATCH --mem=32G
#SBATCH --cpus-per-task=4
#SBATCH --time=4:00:00
#SBATCH --output=/home/ad.msoe.edu/celliot/Digital-twin-v2/isaac-sim-msoe/logs/eval_unified_%j.out
#SBATCH --error=/home/ad.msoe.edu/celliot/Digital-twin-v2/isaac-sim-msoe/logs/eval_unified_%j.out

set -euo pipefail
cd /home/ad.msoe.edu/celliot/Digital-twin-v2/isaac-sim-msoe

PYTHON="/home/ad.msoe.edu/celliot/.conda/envs/vlm/bin/python"
echo "Started: $(date)  on $(hostname)"
echo ""

$PYTHON simulations/robot/eval_unified.py \
    --test_jsonl data/val_unified.jsonl \
    --adapter    data/qwen3vl_unified_lora_v4 \
    "$@"

EXIT=$?
echo ""
echo "Exit: $EXIT    Finished: $(date)"
exit $EXIT
