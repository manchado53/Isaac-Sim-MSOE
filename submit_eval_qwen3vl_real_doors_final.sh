#!/bin/bash
# submit_eval_qwen3vl_real_doors_final.sh — Sim-to-real eval on real_doors_final.
#
# Runs both the unified-V4 fine-tuned Qwen3-VL adapter AND the off-the-shelf
# Qwen3-VL-2B baseline on real_doors_final/RGB/test/{Closed,Open} (200 frames,
# Semi excluded — no semi-open class in the unified 4-way training set).
#
# Both runs use the unified 4-class training prompt so the fine-tuned adapter
# sees its native instruction format. The eval parser falls back to substring
# matches on "OPEN"/"CLOSED", so the unified [DOOR_OPEN]/[DOOR_CLOSED] response
# format and the baseline's free-form responses both parse correctly.
#
# Steps:
#   1. Build data/val_real_doors_final.jsonl
#   2. Eval unified-V4 fine-tuned adapter → data/real_doors_final_eval_qwen3vl_2b_finetuned.json
#   3. Eval zero-shot Qwen3-VL              → data/real_doors_final_eval_qwen3vl_2b_zeroshot.json
#
# Usage:
#   sbatch submit_eval_qwen3vl_real_doors_final.sh
#
# Output:
#   logs/eval_real_doors_final_<JOBID>.out

#SBATCH --job-name=qwen3vl-rd-final
#SBATCH --partition=dgx
#SBATCH --gres=gpu:1
#SBATCH --mem=32G
#SBATCH --cpus-per-task=4
#SBATCH --time=2:00:00
#SBATCH --output=/home/ad.msoe.edu/celliot/Digital-twin-v2/isaac-sim-msoe/logs/eval_real_doors_final_%j.out
#SBATCH --error=/home/ad.msoe.edu/celliot/Digital-twin-v2/isaac-sim-msoe/logs/eval_real_doors_final_%j.out

set -euo pipefail

SCRIPT_DIR="/home/ad.msoe.edu/celliot/Digital-twin-v2/isaac-sim-msoe"
cd "$SCRIPT_DIR"

mkdir -p "$SCRIPT_DIR/logs"

PYTHON="/home/ad.msoe.edu/celliot/.conda/envs/vlm/bin/python"
JSONL="data/val_real_doors_final.jsonl"
ADAPTER="data/qwen3vl_unified_lora_v4"
OUT_FT="data/real_doors_final_eval_qwen3vl_2b_finetuned.json"
OUT_ZS="data/real_doors_final_eval_qwen3vl_2b_zeroshot.json"

UNIFIED_PROMPT="What is shown in this image? Options: door open, door closed, glass cracked, glass intact."

echo "=== Qwen3-VL-2B — real_doors_final Sim-to-Real Eval ==="
echo "Node:     $(hostname)"
echo "Started:  $(date)"
echo "GPU:      $(nvidia-smi --query-gpu=name --format=csv,noheader 2>/dev/null || echo 'unknown')"
echo "Python:   $($PYTHON --version)"
echo "Prompt:   $UNIFIED_PROMPT"
echo ""

# ---- Step 1: build JSONL -------------------------------------------------
echo "--- Building real_doors_final JSONL ---"
$PYTHON simulations/robot/prepare_real_doors_final.py
echo ""

if [ ! -f "$JSONL" ]; then
    echo "ERROR: $JSONL missing"; exit 1
fi
echo "Eval set: $JSONL  ($(wc -l < $JSONL) frames)"
echo ""

# ---- Step 2: fine-tuned eval ---------------------------------------------
echo "--- [1/2] Fine-tuned Qwen3-VL adapter (unified V4) ---"
$PYTHON simulations/robot/eval_vlm_baseline.py \
    --model       Qwen/Qwen3-VL-2B-Instruct \
    --model_class qwen3_vl \
    --val_jsonl   "$JSONL" \
    --adapter     "$ADAPTER" \
    --prompt      "$UNIFIED_PROMPT" \
    --output      "$OUT_FT"

echo ""
echo "--- [2/2] Zero-shot Qwen3-VL (no adapter) ---"
$PYTHON simulations/robot/eval_vlm_baseline.py \
    --model       Qwen/Qwen3-VL-2B-Instruct \
    --model_class qwen3_vl \
    --val_jsonl   "$JSONL" \
    --prompt      "$UNIFIED_PROMPT" \
    --output      "$OUT_ZS"

echo ""
echo "Finished: $(date)"
echo "  Fine-tuned : $OUT_FT"
echo "  Zero-shot  : $OUT_ZS"
