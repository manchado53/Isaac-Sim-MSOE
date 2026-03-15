#!/bin/bash
#SBATCH --job-name=eval-scenarios
#SBATCH --partition=teaching
#SBATCH --gres=gpu:t4:1
#SBATCH --time=02:00:00
#SBATCH --mem=64G
#SBATCH --output=/home/ad.msoe.edu/manchadoa/isaac-sim-msoe/training/logs/eval_scenarios_%j.out
#SBATCH --error=/home/ad.msoe.edu/manchadoa/isaac-sim-msoe/training/logs/eval_scenarios_%j.err

# Scenario-based evaluation for Nova Carter Patrol
# Usage: sbatch training/eval/submit_eval_scenarios.sh [path/to/model.zip]
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
    LATEST_RUN=$(ls -td "$PROJECT_DIR/training/logs/sb3/NovaCarter-Patrol"/*/ 2>/dev/null | head -1)
    if [ -z "$LATEST_RUN" ]; then
        echo "ERROR: No training runs found and no checkpoint specified."
        exit 1
    fi
    # Prefer best_model if it exists, otherwise use final model
    if [ -f "${LATEST_RUN}best_model.zip" ]; then
        CHECKPOINT="${LATEST_RUN}best_model"
    else
        CHECKPOINT="${LATEST_RUN}model"
    fi
fi

# Strip .zip extension if present (SB3 adds it)
CHECKPOINT="${CHECKPOINT%.zip}"

if [ ! -f "${CHECKPOINT}.zip" ]; then
    echo "ERROR: Checkpoint not found: ${CHECKPOINT}.zip"
    exit 1
fi

echo "========================================="
echo "Nova Carter Patrol — Scenario Evaluation"
echo "Job ID:       $SLURM_JOB_ID"
echo "Node:         $SLURMD_NODENAME"
echo "Checkpoint:   ${CHECKPOINT}.zip"
echo "========================================="
echo ""

nvidia-smi
echo ""

CMD="$ISAAC_LAB_DIR/isaaclab.sh -p $PROJECT_DIR/training/eval/eval_scenarios.py --checkpoint $CHECKPOINT --headless"

echo "Running: $CMD"
echo ""

eval $CMD

echo ""
echo "========================================="
echo "Scenario Evaluation Complete (exit code: $?)"
echo "========================================="
