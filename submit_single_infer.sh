#!/bin/bash
# One-shot single-image inference with the fine-tuned adapter.
#   sbatch submit_single_infer.sh [extra args passed to infer_single_image.py]

#SBATCH --job-name=qwen2vl-oneshot
#SBATCH --partition=dgx
#SBATCH --gres=gpu:1
#SBATCH --mem=16G
#SBATCH --cpus-per-task=2
#SBATCH --time=0:20:00
#SBATCH --output=/home/ad.msoe.edu/celliot/Digital-twin-v2/isaac-sim-msoe/logs/oneshot_infer_%j.out
#SBATCH --error=/home/ad.msoe.edu/celliot/Digital-twin-v2/isaac-sim-msoe/logs/oneshot_infer_%j.out

set -euo pipefail

cd /home/ad.msoe.edu/celliot/Digital-twin-v2/isaac-sim-msoe

PYTHON="/home/ad.msoe.edu/celliot/.conda/envs/vlm/bin/python"
echo "Started: $(date)"
echo "Node:    $(hostname)"
echo "GPU:     $(nvidia-smi --query-gpu=name --format=csv,noheader 2>/dev/null || echo 'unknown')"
echo ""

$PYTHON simulations/robot/infer_single_image.py \
    --image   data/no_window_no_door_image/Gemini_Generated_Image_quxknpquxknpquxk.png \
    --adapter data/qwen2vl_door_lora_dgx \
    "$@"

echo ""
echo "Finished: $(date)"
