#!/bin/bash
#SBATCH --job-name=qwen3vl-infer
#SBATCH --partition=dgx
#SBATCH --gres=gpu:1
#SBATCH --mem=32G
#SBATCH --cpus-per-task=4
#SBATCH --time=0:30:00
#SBATCH --output=/home/ad.msoe.edu/celliot/Digital-twin-v2/isaac-sim-msoe/logs/infer_single_%j.out
#SBATCH --error=/home/ad.msoe.edu/celliot/Digital-twin-v2/isaac-sim-msoe/logs/infer_single_%j.out

set -euo pipefail

SCRIPT_DIR="/home/ad.msoe.edu/celliot/Digital-twin-v2/isaac-sim-msoe"
cd "$SCRIPT_DIR"

echo "=== Qwen3-VL single-image inference ==="
echo "Node:    $(hostname)"
echo "Started: $(date)"
echo "GPU:     $(nvidia-smi --query-gpu=name --format=csv,noheader 2>/dev/null || echo 'unknown')"
echo ""

PYTHON="/home/ad.msoe.edu/celliot/.conda/envs/vlm/bin/python"
if [ ! -f "$PYTHON" ]; then
    echo "ERROR: vlm conda env not found at $PYTHON"
    exit 1
fi

$PYTHON simulations/robot/infer_single_image.py \
    --image       hole_in_wall.png \
    --adapter     data/qwen3vl_unified_lora_v4 \
    --model       Qwen/Qwen3-VL-2B-Instruct \
    --model-class qwen3_vl \
    --prompt-mode unified

echo ""
echo "Finished: $(date)"
