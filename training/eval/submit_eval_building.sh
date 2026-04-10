#!/bin/bash
#SBATCH --job-name=eval-building
#SBATCH --partition=teaching
#SBATCH --gres=gpu:t4:1
#SBATCH --time=03:00:00
#SBATCH --mem=64G
#SBATCH --output=/home/ad.msoe.edu/manchadoa/isaac-sim-msoe/training/logs/eval_building_%j.out
#SBATCH --error=/home/ad.msoe.edu/manchadoa/isaac-sim-msoe/training/logs/eval_building_%j.err
#SBATCH --account=undergrad_research

# Building evaluation for fine-tuned Nova Carter policy.
# Runs all 12 building scenarios (5 trials each) and saves results JSON.
#
# Usage:
#   sbatch training/eval/submit_eval_building.sh \
#       training/logs/sb3/NovaCarter-Building/stageC_<timestamp>/best_model_building_stageC.zip
#
# Results written to: training/logs/eval/building_eval_results.json
# Log this run in training/EXPERIMENTS.md after the job completes.

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

mkdir -p "$PROJECT_DIR/training/logs/eval"

CHECKPOINT="${1:-}"
if [ -z "$CHECKPOINT" ]; then
    # Default: latest building run
    LATEST_RUN=$(ls -td "$PROJECT_DIR/training/logs/sb3/NovaCarter-Building"/*/ 2>/dev/null | head -1)
    if [ -z "$LATEST_RUN" ]; then
        echo "ERROR: No building training runs found. Pass checkpoint path as argument."
        exit 1
    fi
    CHECKPOINT="${LATEST_RUN}best_model_building_stageC.zip"
fi

CHECKPOINT="${CHECKPOINT%.zip}"
if [ ! -f "${CHECKPOINT}.zip" ]; then
    echo "ERROR: Checkpoint not found: ${CHECKPOINT}.zip"
    exit 1
fi

echo "========================================="
echo "Nova Carter — Building Scenario Evaluation"
echo "Job ID:       $SLURM_JOB_ID"
echo "Node:         $SLURMD_NODENAME"
echo "Checkpoint:   ${CHECKPOINT}.zip"
echo "========================================="
echo ""
nvidia-smi
echo ""

cd "$ISAAC_LAB_DIR"
./isaaclab.sh -p "$PROJECT_DIR/training/eval/eval_scenarios.py" \
    --checkpoint "$CHECKPOINT" \
    --task "Isaac-NovaCarter-Building-v0" \
    --headless

echo ""
echo "Eval complete. Check training/logs/eval/building_eval_results.json"
echo "Log results in training/EXPERIMENTS.md."
