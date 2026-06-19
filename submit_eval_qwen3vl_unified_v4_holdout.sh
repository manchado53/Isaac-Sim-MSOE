#!/bin/bash
# submit_eval_qwen3vl_unified_v4_holdout.sh — Qwen3-VL unified V4 on holdout.
#
# Runs the qwen3vl_unified_lora_v4 adapter AND the off-the-shelf Qwen3-VL-2B
# baseline on the curated-V4 held-out simulation doors (Doors_13182085,
# Doors_1915613 — 7,776 frames, balanced CLOSED/OPEN, never seen in training).
#
# Both runs use the unified 4-class training prompt so the adapter sees its
# native instruction format. Eval parser substring-matches "OPEN"/"CLOSED" so
# both [DOOR_OPEN]/[DOOR_CLOSED] (fine-tuned) and free-form (baseline)
# responses parse correctly.
#
# Outputs:
#   data/holdout_eval_qwen3vl_2b_unified_v4_finetuned.json
#   data/holdout_eval_qwen3vl_2b_unified_v4_zeroshot.json
#
# Usage:
#   sbatch submit_eval_qwen3vl_unified_v4_holdout.sh

#SBATCH --job-name=qwen3vl-uv4-hold
#SBATCH --partition=dgx
#SBATCH --gres=gpu:1
#SBATCH --mem=32G
#SBATCH --cpus-per-task=4
#SBATCH --time=6:00:00
#SBATCH --output=/home/ad.msoe.edu/celliot/Digital-twin-v2/isaac-sim-msoe/logs/eval_qwen3vl_unified_v4_holdout_%j.out
#SBATCH --error=/home/ad.msoe.edu/celliot/Digital-twin-v2/isaac-sim-msoe/logs/eval_qwen3vl_unified_v4_holdout_%j.out

set -euo pipefail

SCRIPT_DIR="/home/ad.msoe.edu/celliot/Digital-twin-v2/isaac-sim-msoe"
cd "$SCRIPT_DIR"

mkdir -p "$SCRIPT_DIR/logs"

PYTHON="/home/ad.msoe.edu/celliot/.conda/envs/vlm/bin/python"
JSONL="data/holdout_v4.jsonl"
ADAPTER="data/qwen3vl_unified_lora_v4"
OUT_FT="data/holdout_eval_qwen3vl_2b_unified_v4_finetuned.json"
OUT_ZS="data/holdout_eval_qwen3vl_2b_unified_v4_zeroshot.json"

UNIFIED_PROMPT="What is shown in this image? Options: door open, door closed, glass cracked, glass intact."

echo "=== Qwen3-VL-2B unified V4 — holdout (7776-frame) eval ==="
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
