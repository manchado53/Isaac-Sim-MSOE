#!/bin/bash
# submit_real_glass_eval.sh — SLURM job for Qwen2-VL eval on DH cracked-glass set.
#
# Runs eval_real_glass.py against data/dh_cracked_eval/ (30 intact + 90
# cracked DH photos from Stages 1-3 of the cracked-glass pipeline). Uses conda
# 'vlm' env to avoid the GLIBC mismatch that blocks isaacsim/python.sh on dgx.
#
# Usage:
#   sbatch submit_real_glass_eval.sh                                    # baseline
#   sbatch submit_real_glass_eval.sh --adapter data/qwen2vl_door_lora   # fine-tuned

#SBATCH --job-name=qwen2vl-glass-eval
#SBATCH --partition=dgx
#SBATCH --gres=gpu:1
#SBATCH --mem=32G
#SBATCH --cpus-per-task=4
#SBATCH --time=2:00:00
#SBATCH --output=/home/ad.msoe.edu/celliot/Digital-twin-v2/isaac-sim-msoe/logs/real_glass_eval_%j.out
#SBATCH --error=/home/ad.msoe.edu/celliot/Digital-twin-v2/isaac-sim-msoe/logs/real_glass_eval_%j.out

set -euo pipefail

SCRIPT_DIR="/home/ad.msoe.edu/celliot/Digital-twin-v2/isaac-sim-msoe"
cd "$SCRIPT_DIR"

echo "=== Qwen2-VL DH Cracked-Glass Eval ==="
echo "Node:    $(hostname)"
echo "Started: $(date)"
echo "GPU:     $(nvidia-smi --query-gpu=name --format=csv,noheader 2>/dev/null || echo 'unknown')"

PYTHON="/home/ad.msoe.edu/celliot/.conda/envs/vlm/bin/python"
if [ ! -f "$PYTHON" ]; then
    echo "ERROR: vlm conda env not found at $PYTHON"
    exit 1
fi
echo "Python:  $($PYTHON --version)"

DATA_ROOT="data/dh_cracked_eval"
if [ ! -d "$DATA_ROOT/cracked" ] || [ ! -d "$DATA_ROOT/intact" ]; then
    echo "ERROR: $DATA_ROOT is incomplete (missing intact/ or cracked/)."
    echo "  Run Stages 1-3 first:"
    echo "    python simulations/robot/extract_crack_textures.py  ..."
    echo "    sbatch submit_sam2_propose.sh"
    echo "    python simulations/robot/segment_dh_panels.py review  ..."
    echo "    python simulations/robot/composite_cracked_glass.py  ..."
    exit 1
fi
N_INTACT=$(ls "$DATA_ROOT/intact" 2>/dev/null | wc -l)
N_CRACK=$(ls "$DATA_ROOT/cracked" 2>/dev/null | wc -l)
echo "Dataset: $N_INTACT intact + $N_CRACK cracked"

$PYTHON simulations/robot/eval_real_glass.py \
    --data-root "$DATA_ROOT" \
    "$@"
EXIT=$?

echo ""
echo "Exit: $EXIT"
echo "Finished: $(date)"
exit $EXIT
