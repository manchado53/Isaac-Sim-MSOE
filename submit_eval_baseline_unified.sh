#!/bin/bash
# Zero-shot Qwen3-VL-2B eval on the unified 4-class splits (no adapter).
# Pair with submit_eval_unified.sh results to measure adapter lift.
#
# Usage:
#   sbatch submit_eval_baseline_unified.sh                                        # full val (4-class)
#   sbatch submit_eval_baseline_unified.sh --test_jsonl data/val_unified_door.jsonl  --output data/eval_baseline_door.json
#   sbatch submit_eval_baseline_unified.sh --test_jsonl data/val_unified_glass.jsonl --output data/eval_baseline_glass.json

#SBATCH --job-name=qwen3vl-base-u
#SBATCH --partition=dgx
#SBATCH --gres=gpu:1
#SBATCH --mem=32G
#SBATCH --cpus-per-task=4
#SBATCH --time=4:00:00
#SBATCH --output=/home/ad.msoe.edu/celliot/Digital-twin-v2/isaac-sim-msoe/logs/eval_baseline_unified_%j.out
#SBATCH --error=/home/ad.msoe.edu/celliot/Digital-twin-v2/isaac-sim-msoe/logs/eval_baseline_unified_%j.out

set -euo pipefail
cd /home/ad.msoe.edu/celliot/Digital-twin-v2/isaac-sim-msoe

PYTHON="/home/ad.msoe.edu/celliot/.conda/envs/vlm/bin/python"
echo "Started: $(date)  on $(hostname)"
echo ""

# --adapter "" forces BASELINE mode in eval_unified.py; "$@" lets caller
# override --test_jsonl / --output to point at per-domain splits.
$PYTHON simulations/robot/eval_unified.py \
    --test_jsonl data/val_unified.jsonl \
    --adapter    "" \
    --output     data/eval_baseline_unified.json \
    "$@"

EXIT=$?
echo ""
echo "Exit: $EXIT    Finished: $(date)"
exit $EXIT
