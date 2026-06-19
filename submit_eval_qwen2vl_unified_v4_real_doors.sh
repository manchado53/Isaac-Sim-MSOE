#!/bin/bash
# submit_eval_qwen2vl_unified_v4_real_doors.sh — Qwen2-VL unified V4 vs baseline.
#
# Runs the new qwen2vl_unified_lora_v4 adapter (4-class: door open/closed +
# glass cracked/intact) AND the off-the-shelf Qwen2-VL-2B baseline on the
# 400-frame real-door eval set (data/val_real_doors.jsonl).
#
# Both runs use the unified 4-class training prompt so the adapter sees its
# native instruction format. Eval parser substring-matches "OPEN"/"CLOSED"
# which catches both [DOOR_OPEN]/[DOOR_CLOSED] (fine-tuned) and free-form
# (baseline) responses.
#
# Outputs:
#   data/real_doors_eval_qwen2vl_2b_unified_v4_finetuned.json
#   data/real_doors_eval_qwen2vl_2b_unified_v4_zeroshot.json
#
# Usage:
#   sbatch submit_eval_qwen2vl_unified_v4_real_doors.sh

#SBATCH --job-name=qwen2vl-uv4-rd
#SBATCH --partition=dgx
#SBATCH --gres=gpu:1
#SBATCH --mem=32G
#SBATCH --cpus-per-task=4
#SBATCH --time=2:00:00
#SBATCH --output=/home/ad.msoe.edu/celliot/Digital-twin-v2/isaac-sim-msoe/logs/eval_qwen2vl_unified_v4_real_doors_%j.out
#SBATCH --error=/home/ad.msoe.edu/celliot/Digital-twin-v2/isaac-sim-msoe/logs/eval_qwen2vl_unified_v4_real_doors_%j.out

set -euo pipefail

SCRIPT_DIR="/home/ad.msoe.edu/celliot/Digital-twin-v2/isaac-sim-msoe"
cd "$SCRIPT_DIR"

mkdir -p "$SCRIPT_DIR/logs"

PYTHON="/home/ad.msoe.edu/celliot/.conda/envs/vlm/bin/python"
JSONL="data/val_real_doors.jsonl"
ADAPTER="data/qwen2vl_unified_lora_v4"
OUT_FT="data/real_doors_eval_qwen2vl_2b_unified_v4_finetuned.json"
OUT_ZS="data/real_doors_eval_qwen2vl_2b_unified_v4_zeroshot.json"

UNIFIED_PROMPT="What is shown in this image? Options: door open, door closed, glass cracked, glass intact."

echo "=== Qwen2-VL-2B unified V4 — real_doors (400-frame) eval ==="
echo "Node:     $(hostname)"
echo "Started:  $(date)"
echo "GPU:      $(nvidia-smi --query-gpu=name --format=csv,noheader 2>/dev/null || echo 'unknown')"
echo "Python:   $($PYTHON --version)"
echo "Adapter:  $ADAPTER"
echo "JSONL:    $JSONL  ($(wc -l < $JSONL) frames)"
echo "Prompt:   $UNIFIED_PROMPT"
echo ""

if [ ! -f "$JSONL" ]; then
    echo "ERROR: $JSONL missing"; exit 1
fi

# ---- [1/2] fine-tuned eval -----------------------------------------------
echo "--- [1/2] Fine-tuned Qwen2-VL adapter (unified V4) ---"
$PYTHON simulations/robot/eval_vlm_baseline.py \
    --model       Qwen/Qwen2-VL-2B-Instruct \
    --model_class qwen2_vl \
    --val_jsonl   "$JSONL" \
    --adapter     "$ADAPTER" \
    --prompt      "$UNIFIED_PROMPT" \
    --output      "$OUT_FT"

echo ""
echo "--- [2/2] Zero-shot Qwen2-VL (no adapter) ---"
$PYTHON simulations/robot/eval_vlm_baseline.py \
    --model       Qwen/Qwen2-VL-2B-Instruct \
    --model_class qwen2_vl \
    --val_jsonl   "$JSONL" \
    --prompt      "$UNIFIED_PROMPT" \
    --output      "$OUT_ZS"

echo ""
echo "Finished: $(date)"
echo "  Fine-tuned : $OUT_FT"
echo "  Zero-shot  : $OUT_ZS"
