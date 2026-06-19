#!/bin/bash
# submit_eval_qwen3vl_holdout.sh — Held-out-door generalization eval for the
# Qwen3-VL-2B fine-tune.
#
# Step 1: extract holdout NPZ shards (Doors_13182085, Doors_1915613) into
#         data/finetune_images_v4_holdout/ + data/holdout_v4.jsonl
# Step 2: evaluate the LoRA + projector adapter on every holdout frame.
#
# These two doors are NOT in the training set — this is the true generalization
# test. CLAUDE.md notes the prior Qwen2-VL fine-tune scored 0% on a held-out
# door (Doors_13185454), so this is the most important number to look at.
#
# Usage:
#   sbatch submit_eval_qwen3vl_holdout.sh
#
# Output:
#   data/holdout_v4.jsonl                          (all holdout frames)
#   data/holdout_eval_qwen3vl_2b_finetuned.json    (per-sample eval results)
#   logs/eval_qwen3vl_holdout_<JOBID>.out

#SBATCH --job-name=qwen3vl-holdout
#SBATCH --partition=dgx
#SBATCH --gres=gpu:1
#SBATCH --mem=48G
#SBATCH --cpus-per-task=4
#SBATCH --time=8:00:00
#SBATCH --output=/home/ad.msoe.edu/celliot/Digital-twin-v2/isaac-sim-msoe/logs/eval_qwen3vl_holdout_%j.out
#SBATCH --error=/home/ad.msoe.edu/celliot/Digital-twin-v2/isaac-sim-msoe/logs/eval_qwen3vl_holdout_%j.out

set -euo pipefail

SCRIPT_DIR="/home/ad.msoe.edu/celliot/Digital-twin-v2/isaac-sim-msoe"
cd "$SCRIPT_DIR"

mkdir -p "$SCRIPT_DIR/logs"

PYTHON="/home/ad.msoe.edu/celliot/.conda/envs/vlm/bin/python"
ADAPTER="data/qwen3vl_door_lora_v4_curated"
JSONL="data/holdout_v4.jsonl"
OUT_PATH="data/holdout_eval_qwen3vl_2b_finetuned.json"

echo "=== Qwen3-VL-2B Fine-Tuned — HELD-OUT DOORS ==="
echo "Node:     $(hostname)"
echo "Started:  $(date)"
echo "GPU:      $(nvidia-smi --query-gpu=name --format=csv,noheader 2>/dev/null || echo 'unknown')"
echo "Python:   $($PYTHON --version)"
echo "Adapter:  $ADAPTER"
echo ""

# ---- Step 1: extract holdout NPZ → JPEG + JSONL --------------------------
echo "--- Extracting holdout shards ---"
$PYTHON simulations/robot/prepare_holdout_data.py
echo ""

if [ ! -f "$JSONL" ]; then
    echo "ERROR: extraction did not produce $JSONL"; exit 1
fi
echo "Holdout set: $JSONL  ($(wc -l < $JSONL) frames)"
echo ""

# ---- Step 2: evaluate adapter on every holdout frame ---------------------
echo "--- Evaluating adapter on holdout doors ---"
$PYTHON simulations/robot/eval_vlm_baseline.py \
    --model       Qwen/Qwen3-VL-2B-Instruct \
    --model_class qwen3_vl \
    --val_jsonl   "$JSONL" \
    --adapter     "$ADAPTER" \
    --output      "$OUT_PATH"
EVAL_EXIT=$?

echo ""
echo "Eval exit code: $EVAL_EXIT"
echo "Finished: $(date)"
exit $EVAL_EXIT
