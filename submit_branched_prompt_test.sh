#!/bin/bash
# Test the conditional/branched prompt on 1 no-door image + 1 CLOSED door + 1 OPEN door.
#   sbatch submit_branched_prompt_test.sh

#SBATCH --job-name=qwen2vl-branched
#SBATCH --partition=dgx
#SBATCH --gres=gpu:1
#SBATCH --mem=16G
#SBATCH --cpus-per-task=2
#SBATCH --time=0:20:00
#SBATCH --output=/home/ad.msoe.edu/celliot/Digital-twin-v2/isaac-sim-msoe/logs/branched_prompt_%j.out
#SBATCH --error=/home/ad.msoe.edu/celliot/Digital-twin-v2/isaac-sim-msoe/logs/branched_prompt_%j.out

set -euo pipefail
cd /home/ad.msoe.edu/celliot/Digital-twin-v2/isaac-sim-msoe

PYTHON="/home/ad.msoe.edu/celliot/.conda/envs/vlm/bin/python"
echo "Started: $(date)"
echo "Node:    $(hostname)"
echo "GPU:     $(nvidia-smi --query-gpu=name --format=csv,noheader 2>/dev/null || echo 'unknown')"
echo ""

$PYTHON simulations/robot/infer_single_image.py \
    --adapter data/qwen2vl_door_lora_dgx \
    --prompt-mode branched \
    --image data/no_window_no_door_image/Gemini_Generated_Image_quxknpquxknpquxk.png \
    --image data/val_images_labeled_bright_only/CLOSED_Doors_8286057_00001.jpg \
    --image data/val_images_labeled_bright_only/OPEN_Doors_8286057_00097.jpg

echo ""
echo "Finished: $(date)"
