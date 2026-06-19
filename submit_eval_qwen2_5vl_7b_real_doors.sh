#!/bin/bash
# submit_eval_qwen2_5vl_7b_real_doors.sh — off-the-shelf Qwen2.5-VL-7B on real doors.
#
# Mirror of submit_eval_qwen2vl_real_doors.sh but evaluates the larger
# Qwen2.5-VL-7B-Instruct (no adapter) on the same 400-frame real-door set.
# Same unified 4-class prompt is used so results are directly comparable to
# the Qwen3-VL-2B and Qwen2-VL-2B zero-shot runs.
#
# Output:
#   data/real_doors_eval_qwen2_5vl_7b_zeroshot.json
#
# Usage:
#   sbatch submit_eval_qwen2_5vl_7b_real_doors.sh

#SBATCH --job-name=qwen2_5vl-7b-rd
#SBATCH --partition=dgx
#SBATCH --gres=gpu:1
#SBATCH --mem=48G
#SBATCH --cpus-per-task=4
#SBATCH --time=2:00:00
#SBATCH --output=/home/ad.msoe.edu/celliot/Digital-twin-v2/isaac-sim-msoe/logs/eval_qwen2_5vl_7b_real_doors_%j.out
#SBATCH --error=/home/ad.msoe.edu/celliot/Digital-twin-v2/isaac-sim-msoe/logs/eval_qwen2_5vl_7b_real_doors_%j.out

set -euo pipefail

SCRIPT_DIR="/home/ad.msoe.edu/celliot/Digital-twin-v2/isaac-sim-msoe"
cd "$SCRIPT_DIR"

mkdir -p "$SCRIPT_DIR/logs"

PYTHON="/home/ad.msoe.edu/celliot/.conda/envs/vlm/bin/python"
JSONL="data/val_real_doors.jsonl"
OUT_ZS="data/real_doors_eval_qwen2_5vl_7b_zeroshot.json"

UNIFIED_PROMPT="What is shown in this image? Options: door open, door closed, glass cracked, glass intact."

echo "=== Qwen2.5-VL-7B (off-the-shelf) — real_doors (400-frame) eval ==="
echo "Node:     $(hostname)"
echo "Started:  $(date)"
echo "GPU:      $(nvidia-smi --query-gpu=name --format=csv,noheader 2>/dev/null || echo 'unknown')"
echo "Python:   $($PYTHON --version)"
echo "Model:    Qwen/Qwen2.5-VL-7B-Instruct (no adapter)"
echo "JSONL:    $JSONL  ($(wc -l < $JSONL) frames)"
echo "Prompt:   $UNIFIED_PROMPT"
echo ""

if [ ! -f "$JSONL" ]; then
    echo "ERROR: $JSONL missing"; exit 1
fi

$PYTHON simulations/robot/eval_vlm_baseline.py \
    --model       Qwen/Qwen2.5-VL-7B-Instruct \
    --model_class qwen2_5_vl \
    --val_jsonl   "$JSONL" \
    --prompt      "$UNIFIED_PROMPT" \
    --output      "$OUT_ZS"

echo ""
echo "Finished: $(date)"
echo "  Zero-shot Qwen2.5-VL-7B : $OUT_ZS"
