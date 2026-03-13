#!/bin/bash
#SBATCH --job-name=eval-ridgeback
#SBATCH --partition=teaching
#SBATCH --gres=gpu:1
#SBATCH --time=01:00:00
#SBATCH --mem=32G
#SBATCH --output=/home/ad.msoe.edu/manchadoa/isaac-sim-msoe/training/logs/eval_%j.out
#SBATCH --error=/home/ad.msoe.edu/manchadoa/isaac-sim-msoe/training/logs/eval_%j.err

# Evaluate trained Ridgeback model and record video
# Usage: sbatch training/submit_eval.sh [path/to/model.zip]
#
# If no checkpoint given, uses the latest training run's final model.

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

mkdir -p "$PROJECT_DIR/training/logs"

# Find checkpoint: use argument or latest run
CHECKPOINT="${1:-}"
if [ -z "$CHECKPOINT" ]; then
    LATEST_RUN=$(ls -td "$PROJECT_DIR/training/logs/sb3/Ridgeback-Patrol"/*/ 2>/dev/null | head -1)
    if [ -z "$LATEST_RUN" ]; then
        echo "ERROR: No training runs found and no checkpoint specified."
        exit 1
    fi
    CHECKPOINT="${LATEST_RUN}model.zip"
fi

if [ ! -f "$CHECKPOINT" ]; then
    echo "ERROR: Checkpoint not found: $CHECKPOINT"
    exit 1
fi

echo "========================================="
echo "Ridgeback Patrol Evaluation"
echo "Job ID:       $SLURM_JOB_ID"
echo "Node:         $SLURMD_NODENAME"
echo "Checkpoint:   $CHECKPOINT"
echo "========================================="
echo ""

nvidia-smi
echo ""

CMD="$ISAAC_LAB_DIR/isaaclab.sh -p $PROJECT_DIR/training/eval_ridgeback.py --checkpoint $CHECKPOINT --headless --num_envs 16 --video_length 500"

echo "Running: $CMD"
echo ""

eval $CMD

echo ""
echo "========================================="
echo "Evaluation Complete (exit code: $?)"
echo "========================================="
