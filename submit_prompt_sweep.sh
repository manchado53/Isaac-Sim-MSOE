#!/bin/bash
# Prompt-engineering sweep: test 4 branched-prompt variants on 1 no-door +
# 3 OPEN + 3 CLOSED images to find a prompt that keeps classification accuracy
# while triggering the no-door description branch.

#SBATCH --job-name=qwen2vl-sweep
#SBATCH --partition=dgx
#SBATCH --gres=gpu:1
#SBATCH --mem=16G
#SBATCH --cpus-per-task=2
#SBATCH --time=0:30:00
#SBATCH --output=/home/ad.msoe.edu/celliot/Digital-twin-v2/isaac-sim-msoe/logs/prompt_sweep_%j.out
#SBATCH --error=/home/ad.msoe.edu/celliot/Digital-twin-v2/isaac-sim-msoe/logs/prompt_sweep_%j.out

set -euo pipefail
cd /home/ad.msoe.edu/celliot/Digital-twin-v2/isaac-sim-msoe

PYTHON="/home/ad.msoe.edu/celliot/.conda/envs/vlm/bin/python"
echo "Started: $(date)"
echo "Node:    $(hostname)"
echo ""

VAL=data/val_images_labeled_bright_only

$PYTHON simulations/robot/infer_single_image.py \
    --adapter data/qwen2vl_door_lora_dgx \
    --prompt-mode sweep \
    --image data/no_window_no_door_image/Gemini_Generated_Image_quxknpquxknpquxk.png \
    --image $VAL/OPEN_Doors_8286057_00097.jpg \
    --image $VAL/OPEN_Doors_8286057_00103.jpg \
    --image $VAL/OPEN_Doors_10649156_00130.jpg \
    --image $VAL/CLOSED_Doors_8286057_00001.jpg \
    --image $VAL/CLOSED_Doors_8286057_00038.jpg \
    --image $VAL/CLOSED_Doors_10649156_00000.jpg

echo ""
echo "Finished: $(date)"
