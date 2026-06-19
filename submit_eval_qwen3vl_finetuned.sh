#!/bin/bash
# submit_eval_qwen3vl_finetuned.sh — Evaluate the Qwen3-VL-2B fine-tune on the
# bright-only val split (3,694 frames) for a direct comparison against the
# zero-shot Qwen3-VL-2B baseline (75.9% on the same set).
#
# Usage:
#   sbatch submit_eval_qwen3vl_finetuned.sh
#
# Output:
#   data/finetuned_eval_qwen3vl_2b_bright.json
#   logs/eval_qwen3vl_finetuned_<JOBID>.out

#SBATCH --job-name=qwen3vl-ft-eval
#SBATCH --partition=dgx
#SBATCH --gres=gpu:1
#SBATCH --mem=32G
#SBATCH --cpus-per-task=4
#SBATCH --time=4:00:00
#SBATCH --output=/home/ad.msoe.edu/celliot/Digital-twin-v2/isaac-sim-msoe/logs/eval_qwen3vl_finetuned_%j.out
#SBATCH --error=/home/ad.msoe.edu/celliot/Digital-twin-v2/isaac-sim-msoe/logs/eval_qwen3vl_finetuned_%j.out

set -euo pipefail

SCRIPT_DIR="/home/ad.msoe.edu/celliot/Digital-twin-v2/isaac-sim-msoe"
cd "$SCRIPT_DIR"

mkdir -p "$SCRIPT_DIR/logs"

PYTHON="/home/ad.msoe.edu/celliot/.conda/envs/vlm/bin/python"
VAL_JSONL="data/val_bright_only.jsonl"
ADAPTER="data/qwen3vl_door_lora_v4_curated"
OUT_PATH="data/finetuned_eval_qwen3vl_2b_bright.json"

echo "=== Qwen3-VL-2B Fine-Tuned Evaluation (bright-only) ==="
echo "Node:     $(hostname)"
echo "Started:  $(date)"
echo "GPU:      $(nvidia-smi --query-gpu=name --format=csv,noheader 2>/dev/null || echo 'unknown')"
echo "Python:   $($PYTHON --version)"
echo "Val:      $VAL_JSONL  ($(wc -l < $VAL_JSONL) frames)"
echo "Adapter:  $ADAPTER"
echo "Output:   $OUT_PATH"
echo ""

if [ ! -f "$VAL_JSONL" ]; then
    echo "ERROR: $VAL_JSONL not found."; exit 1
fi
if [ ! -d "$ADAPTER" ]; then
    echo "ERROR: $ADAPTER not found."; exit 1
fi

$PYTHON simulations/robot/eval_vlm_baseline.py \
    --model       Qwen/Qwen3-VL-2B-Instruct \
    --model_class qwen3_vl \
    --val_jsonl   "$VAL_JSONL" \
    --adapter     "$ADAPTER" \
    --output      "$OUT_PATH"
EVAL_EXIT=$?

echo ""
echo "Eval exit code: $EVAL_EXIT"
echo "Finished: $(date)"
exit $EVAL_EXIT
