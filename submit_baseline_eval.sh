#!/bin/bash
# submit_baseline_eval.sh — SLURM job for Qwen2-VL evaluation on the val split.
#
# Evaluates against data/val.jsonl (15% stratified val split from the 8
# production doors). Uses conda 'vlm' env to avoid GLIBC mismatch from
# isaacsim/python.sh on dgx nodes.
#
# Usage:
#   sbatch submit_baseline_eval.sh               # baseline (no adapter)
#   sbatch submit_baseline_eval.sh --adapter data/qwen2vl_door_lora_v2  # fine-tuned
#
# One-time conda env setup (run on a GPU node):
#   conda create -n vlm python=3.10 -y && conda activate vlm
#   pip install torch torchvision --index-url https://download.pytorch.org/whl/cu118
#   pip install transformers>=4.45 peft>=0.12 pillow numpy scikit-learn accelerate

#SBATCH --job-name=qwen2vl-eval
#SBATCH --partition=dgx
#SBATCH --gres=gpu:1
#SBATCH --mem=32G
#SBATCH --cpus-per-task=4
#SBATCH --time=4:00:00
#SBATCH --output=/home/ad.msoe.edu/celliot/Digital-twin-v2/isaac-sim-msoe/logs/baseline_eval_%j.out
#SBATCH --error=/home/ad.msoe.edu/celliot/Digital-twin-v2/isaac-sim-msoe/logs/baseline_eval_%j.out

set -euo pipefail

SCRIPT_DIR="/home/ad.msoe.edu/celliot/Digital-twin-v2/isaac-sim-msoe"
cd "$SCRIPT_DIR"

echo "=== Qwen2-VL Baseline Evaluation ==="
echo "Node:      $(hostname)"
echo "Started:   $(date)"
echo "GPU:       $(nvidia-smi --query-gpu=name --format=csv,noheader 2>/dev/null || echo 'unknown')"
echo ""

# Check val split exists
TEST_JSONL="data/val.jsonl"
if [ ! -f "$TEST_JSONL" ]; then
    echo "ERROR: $TEST_JSONL not found."
    echo "  Run: sbatch submit_prepare_data.sh"
    exit 1
fi
echo "Eval set: $TEST_JSONL  ($(wc -l < "$TEST_JSONL") frames)"
echo ""

PYTHON="/home/ad.msoe.edu/celliot/.conda/envs/vlm/bin/python"
if [ ! -f "$PYTHON" ]; then
    echo "ERROR: vlm conda env not found at $PYTHON"
    exit 1
fi
echo "Python: $($PYTHON --version)"

echo "--- Evaluation ---"
$PYTHON simulations/robot/eval_baseline.py \
    --test_jsonl "$TEST_JSONL" \
    "$@"
EVAL_EXIT=$?

echo ""
echo "Eval exit code: $EVAL_EXIT"
echo "Finished: $(date)"

exit $EVAL_EXIT
