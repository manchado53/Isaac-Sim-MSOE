#!/bin/bash
# submit_baseline_eval.sh — SLURM job for Qwen2-VL baseline evaluation (no Isaac Sim).
#
# Evaluates the BASE Qwen2-VL-2B-Instruct model against door_labels_static.npz.
# Uses conda 'vlm' env to avoid GLIBC mismatch from isaacsim/python.sh on dgx nodes.
#
# Usage:
#   sbatch scripts/slurm/submit_baseline_eval.sh               # baseline (no adapter)
#   sbatch scripts/slurm/submit_baseline_eval.sh --adapter data/qwen2vl_door_lora  # fine-tuned
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

# Check dataset exists
NPZ="data/door_labels_static_multi_camera.npz"
if [ ! -f "$NPZ" ]; then
    echo "ERROR: $NPZ not found."
    echo "  Run: sbatch scripts/slurm/submit_isaac_job.sh simulations/robot/door_static_collection.py"
    exit 1
fi
echo "Dataset: $NPZ  ($(du -sh "$NPZ" | cut -f1))"
echo ""

PYTHON="/home/ad.msoe.edu/celliot/.conda/envs/vlm/bin/python"
if [ ! -f "$PYTHON" ]; then
    echo "ERROR: vlm conda env not found at $PYTHON"
    exit 1
fi
echo "Python: $($PYTHON --version)"

echo "--- Evaluation ---"
$PYTHON simulations/robot/eval_baseline.py \
    --npz "$NPZ" \
    --n_samples 1800 \
    "$@"
EVAL_EXIT=$?

echo ""
echo "Eval exit code: $EVAL_EXIT"
echo "Finished: $(date)"

exit $EVAL_EXIT
