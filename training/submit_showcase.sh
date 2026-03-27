#!/bin/bash
#SBATCH --job-name=isaac-showcase
#SBATCH --partition=teaching
#SBATCH --gres=gpu:t4:1
#SBATCH --time=1:00:00
#SBATCH --mem=64G
#SBATCH --output=%x_%j.out
#SBATCH --error=%x_%j.err
#SBATCH --account=students

# Parallel environments showcase video
# Submit with: cd ~/isaac-sim-msoe && sbatch training/submit_showcase.sh

set -e

PROJECT_DIR="$HOME/isaac-sim-msoe"
ISAAC_LAB_DIR="$HOME/IsaacLab"

export TMPDIR="/tmp/$USER"
mkdir -p "$TMPDIR"

export OMNI_KIT_ACCEPT_EULA=YES
export PRIVACY_CONSENT=Y
export PYTHONUNBUFFERED=1

if command -v conda &> /dev/null; then
    eval "$(conda shell.bash hook)"
    conda deactivate 2>/dev/null || true
fi

mkdir -p "$PROJECT_DIR/videos"

echo "========================================="
echo "Parallel Environments Showcase"
echo "Job ID:  $SLURM_JOB_ID"
echo "Node:    $SLURMD_NODENAME"
echo "========================================="
nvidia-smi
echo ""

CMD="$ISAAC_LAB_DIR/isaaclab.sh -p $PROJECT_DIR/training/showcase_parallel_envs.py \
    --num_envs 16 --duration 25 --fps 30 --headless --enable_cameras"

echo "Running: $CMD"
echo ""
eval $CMD

echo ""
echo "========================================="
echo "Done — video saved to videos/parallel_showcase.mp4"
echo "========================================="
