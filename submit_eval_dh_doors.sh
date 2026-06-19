#!/bin/bash
# submit_eval_dh_doors.sh — Sim-to-real evaluation on 54 Diercks Hall door photos.
#
# Runs two back-to-back evals on data/dh_doors_test.jsonl:
#   1. Qwen3-VL-2B baseline (zero-shot, off-the-shelf)
#   2. Qwen3-VL-2B + fine-tuned LoRA adapter (trained on curated sim data)
#
# Outputs:
#   data/dh_doors_baseline_qwen3vl.json
#   data/dh_doors_finetuned_qwen3vl.json

#SBATCH --job-name=dh-doors-eval
#SBATCH --partition=dgx
#SBATCH --gres=gpu:1
#SBATCH --mem=32G
#SBATCH --cpus-per-task=4
#SBATCH --time=4:00:00
#SBATCH --output=/home/ad.msoe.edu/celliot/Digital-twin-v2/isaac-sim-msoe/logs/dh_doors_eval_%j.out
#SBATCH --error=/home/ad.msoe.edu/celliot/Digital-twin-v2/isaac-sim-msoe/logs/dh_doors_eval_%j.out

set -euo pipefail

PROJECT="/home/ad.msoe.edu/celliot/Digital-twin-v2/isaac-sim-msoe"
PYTHON="/home/ad.msoe.edu/celliot/.conda/envs/vlm/bin/python"
cd "$PROJECT"

MODEL_ID="Qwen/Qwen3-VL-2B-Instruct"
MODEL_CLASS="qwen3_vl"
JSONL="data/dh_doors_test.jsonl"
ADAPTER="data/qwen3vl_door_lora_v4_curated"

echo "=== Diercks Hall Real-World Door Eval ==="
echo "Node:     $(hostname)"
echo "Started:  $(date)"
echo "GPU:      $(nvidia-smi --query-gpu=name --format=csv,noheader 2>/dev/null || echo 'unknown')"
echo "Model:    $MODEL_ID  (class=$MODEL_CLASS)"
echo "Dataset:  $JSONL  ($(wc -l < $JSONL) frames)"
echo ""

# Pre-flight checks
if [ ! -f "$JSONL" ]; then
    echo "ERROR: $JSONL not found."
    exit 1
fi
if [ ! -d "$ADAPTER" ]; then
    echo "ERROR: adapter dir $ADAPTER not found."
    exit 1
fi

# ------------------------------------------------------------------
# Run 1: zero-shot baseline
# ------------------------------------------------------------------
echo "--- [1/2] Baseline (zero-shot, no adapter) ---"
$PYTHON simulations/robot/eval_vlm_baseline.py \
    --model        "$MODEL_ID" \
    --model_class  "$MODEL_CLASS" \
    --val_jsonl    "$JSONL" \
    --output       "data/dh_doors_baseline_qwen3vl.json"
echo ""

# ------------------------------------------------------------------
# Run 2: fine-tuned (sim-to-real transfer)
# ------------------------------------------------------------------
echo "--- [2/2] Fine-tuned (adapter=$ADAPTER) ---"
$PYTHON simulations/robot/eval_vlm_baseline.py \
    --model        "$MODEL_ID" \
    --model_class  "$MODEL_CLASS" \
    --val_jsonl    "$JSONL" \
    --adapter      "$ADAPTER" \
    --output       "data/dh_doors_finetuned_qwen3vl.json"

echo ""
echo "Finished: $(date)"
echo "Results:"
echo "  Baseline   : data/dh_doors_baseline_qwen3vl.json"
echo "  Fine-tuned : data/dh_doors_finetuned_qwen3vl.json"
