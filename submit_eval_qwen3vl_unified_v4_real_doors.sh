#!/bin/bash
# submit_eval_qwen3vl_unified_v4_real_doors.sh — Unified V4 adapter on real doors.
#
# Runs the qwen3vl_unified_lora_v4 adapter (4-class: door open/closed +
# glass cracked/intact) AND the off-the-shelf Qwen3-VL-2B baseline on
# data/val_real_doors.jsonl — the 400-frame public real-door eval set
# (200 Closed, 200 Open+Semi-merged).
#
# Both runs use the unified 4-class training prompt so the adapter sees its
# native instruction format. Eval parser substring-matches "OPEN"/"CLOSED"
# which catches both [DOOR_OPEN]/[DOOR_CLOSED] (fine-tuned) and free-form
# (baseline) responses.
#
# Outputs:
#   data/real_doors_eval_qwen3vl_2b_unified_v4_finetuned.json
#   data/real_doors_eval_qwen3vl_2b_unified_v4_zeroshot.json
#
# Note: prior eval (job 244368) ran the door-only qwen3vl_door_lora_v4_curated
# adapter on this same JSONL. Don't overwrite those results — separate filenames.
#
# Usage:
#   sbatch submit_eval_qwen3vl_unified_v4_real_doors.sh

#SBATCH --job-name=qwen3vl-uv4-rd
#SBATCH --partition=dgx
#SBATCH --gres=gpu:1
#SBATCH --mem=32G
#SBATCH --cpus-per-task=4
#SBATCH --time=2:00:00
#SBATCH --output=/home/ad.msoe.edu/celliot/Digital-twin-v2/isaac-sim-msoe/logs/eval_unified_v4_real_doors_%j.out
#SBATCH --error=/home/ad.msoe.edu/celliot/Digital-twin-v2/isaac-sim-msoe/logs/eval_unified_v4_real_doors_%j.out

set -euo pipefail

SCRIPT_DIR="/home/ad.msoe.edu/celliot/Digital-twin-v2/isaac-sim-msoe"
cd "$SCRIPT_DIR"

mkdir -p "$SCRIPT_DIR/logs"

PYTHON="/home/ad.msoe.edu/celliot/.conda/envs/vlm/bin/python"
JSONL="data/val_real_doors.jsonl"
ADAPTER="data/qwen3vl_unified_lora_v4"
OUT_FT="data/real_doors_eval_qwen3vl_2b_unified_v4_finetuned.json"
OUT_ZS="data/real_doors_eval_qwen3vl_2b_unified_v4_zeroshot.json"

UNIFIED_PROMPT="What is shown in this image? Options: door open, door closed, glass cracked, glass intact."

echo "=== Qwen3-VL-2B unified V4 — real_doors (400-frame) eval ==="
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
