#!/bin/bash
# submit_sam2_propose.sh — SAM2 automatic-mask proposals for DH glass photos.
#
# Runs segment_dh_panels.py in --propose mode on a GPU. Writes top-K candidate
# masks per image into data/dh_cracked_eval/proposals/ as NPZ + preview JPG.
# The interactive --review pass happens separately on a login/desktop node.
#
# Usage:
#   sbatch submit_sam2_propose.sh

#SBATCH --job-name=sam2-propose
#SBATCH --partition=teaching
#SBATCH --gres=gpu:1
#SBATCH --mem=24G
#SBATCH --cpus-per-task=4
#SBATCH --time=1:00:00
#SBATCH --output=/home/ad.msoe.edu/celliot/Digital-twin-v2/isaac-sim-msoe/logs/sam2_propose_%j.out
#SBATCH --error=/home/ad.msoe.edu/celliot/Digital-twin-v2/isaac-sim-msoe/logs/sam2_propose_%j.out

set -euo pipefail

SCRIPT_DIR="/home/ad.msoe.edu/celliot/Digital-twin-v2/isaac-sim-msoe"
cd "$SCRIPT_DIR"

echo "=== SAM2 propose (DH glass) ==="
echo "Node:    $(hostname)"
echo "Started: $(date)"
echo "GPU:     $(nvidia-smi --query-gpu=name --format=csv,noheader 2>/dev/null || echo 'unknown')"

PYTHON="/home/ad.msoe.edu/celliot/.conda/envs/sam2_env/bin/python"
if [ ! -f "$PYTHON" ]; then
    echo "ERROR: sam2_env python not found at $PYTHON"
    exit 1
fi
echo "Python:  $($PYTHON --version)"

CHECKPOINT="data/sam2_checkpoints/sam2.1_hiera_large.pt"
if [ ! -f "$CHECKPOINT" ]; then
    echo "ERROR: SAM2 checkpoint not found at $CHECKPOINT"
    exit 1
fi

$PYTHON simulations/robot/segment_dh_panels.py propose \
    --src dh_glass_panels \
    --out data/dh_cracked_eval/proposals \
    --checkpoint "$CHECKPOINT" \
    --cfg configs/sam2.1/sam2.1_hiera_l.yaml \
    --top-k 20 \
    --overwrite \
    "$@"

EXIT=$?
echo ""
echo "Exit: $EXIT"
echo "Finished: $(date)"
exit $EXIT
