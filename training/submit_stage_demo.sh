#!/bin/bash
#SBATCH --job-name=curriculum-demo
#SBATCH --partition=teaching
#SBATCH --gres=gpu:t4:1
#SBATCH --mem=32G
#SBATCH --output=/home/ad.msoe.edu/manchadoa/isaac-sim-msoe/training/logs/demo_%j.out
#SBATCH --error=/home/ad.msoe.edu/manchadoa/isaac-sim-msoe/training/logs/demo_%j.err
#SBATCH --time=01:00:00
#SBATCH --account=undergrad_research

# Curriculum Stage Demo — records all 4 stages in one short MP4
# Submit with: cd ~/isaac-sim-msoe && sbatch training/submit_stage_demo.sh

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

echo "========================================="
echo "Curriculum Stage Demo"
echo "Job ID:  $SLURM_JOB_ID"
echo "Node:    $SLURMD_NODENAME"
echo "Output:  $PROJECT_DIR/training/logs/curriculum_demo/"
echo "========================================="
echo ""

nvidia-smi
echo ""

CMD="$ISAAC_LAB_DIR/isaaclab.sh -p $PROJECT_DIR/training/demo_curriculum_stages.py --headless"
echo "Running: $CMD"
echo ""

eval $CMD
EXIT_CODE=$?

echo ""
echo "========================================="
echo "Demo Complete (exit code: $EXIT_CODE)"
echo "Video: $PROJECT_DIR/training/logs/curriculum_demo/curriculum_demo-episode-0.mp4"
echo "========================================="
