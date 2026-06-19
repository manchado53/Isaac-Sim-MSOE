#!/bin/bash
# submit_eval_qwen3vl_real_doors.sh — Sim-to-real transfer eval.
#
# Runs both the fine-tuned Qwen3-VL adapter AND the off-the-shelf zero-shot
# baseline on val_real_doors/ (200 Closed + 200 Open real-world door photos),
# producing two directly-comparable JSON outputs.
#
# Steps:
#   1. Build data/val_real_doors.jsonl (paths point to ../val_real_doors/...)
#   2. Eval fine-tuned adapter   → data/real_doors_eval_qwen3vl_2b_finetuned.json
#   3. Eval zero-shot Qwen3-VL   → data/real_doors_eval_qwen3vl_2b_zeroshot.json
#
# Usage:
#   sbatch submit_eval_qwen3vl_real_doors.sh
#
# Output:
#   logs/eval_real_doors_<JOBID>.out

#SBATCH --job-name=qwen3vl-real
#SBATCH --partition=dgx
#SBATCH --gres=gpu:1
#SBATCH --mem=32G
#SBATCH --cpus-per-task=4
#SBATCH --time=2:00:00
#SBATCH --output=/home/ad.msoe.edu/celliot/Digital-twin-v2/isaac-sim-msoe/logs/eval_real_doors_%j.out
#SBATCH --error=/home/ad.msoe.edu/celliot/Digital-twin-v2/isaac-sim-msoe/logs/eval_real_doors_%j.out

set -euo pipefail

SCRIPT_DIR="/home/ad.msoe.edu/celliot/Digital-twin-v2/isaac-sim-msoe"
cd "$SCRIPT_DIR"

mkdir -p "$SCRIPT_DIR/logs"

PYTHON="/home/ad.msoe.edu/celliot/.conda/envs/vlm/bin/python"
JSONL="data/val_real_doors.jsonl"
ADAPTER="data/qwen3vl_door_lora_v4_curated"
OUT_FT="data/real_doors_eval_qwen3vl_2b_finetuned.json"
OUT_ZS="data/real_doors_eval_qwen3vl_2b_zeroshot.json"

echo "=== Qwen3-VL-2B — Real Door Sim-to-Real Eval ==="
echo "Node:     $(hostname)"
echo "Started:  $(date)"
echo "GPU:      $(nvidia-smi --query-gpu=name --format=csv,noheader 2>/dev/null || echo 'unknown')"
echo "Python:   $($PYTHON --version)"
echo ""

# ---- Step 1: build JSONL -------------------------------------------------
echo "--- Building real-door JSONL ---"
$PYTHON simulations/robot/prepare_real_doors_data.py
echo ""

if [ ! -f "$JSONL" ]; then
    echo "ERROR: $JSONL missing"; exit 1
fi
echo "Real-door eval set: $JSONL  ($(wc -l < $JSONL) frames)"
echo ""

# ---- Step 2: fine-tuned eval ---------------------------------------------
echo "--- [1/2] Fine-tuned Qwen3-VL adapter ---"
$PYTHON simulations/robot/eval_vlm_baseline.py \
    --model       Qwen/Qwen3-VL-2B-Instruct \
    --model_class qwen3_vl \
    --val_jsonl   "$JSONL" \
    --adapter     "$ADAPTER" \
    --output      "$OUT_FT"

echo ""
echo "--- [2/2] Zero-shot Qwen3-VL (no adapter) ---"
$PYTHON simulations/robot/eval_vlm_baseline.py \
    --model       Qwen/Qwen3-VL-2B-Instruct \
    --model_class qwen3_vl \
    --val_jsonl   "$JSONL" \
    --output      "$OUT_ZS"

echo ""
echo "Finished: $(date)"
echo "  Fine-tuned : $OUT_FT"
echo "  Zero-shot  : $OUT_ZS"
