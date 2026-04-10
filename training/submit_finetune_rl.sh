#!/bin/bash
#SBATCH --job-name=finetune-building
#SBATCH --partition=teaching
#SBATCH --gres=gpu:t4:1
#SBATCH --mem=64G
#SBATCH --output=/home/ad.msoe.edu/manchadoa/isaac-sim-msoe/training/logs/finetune_%j.out
#SBATCH --error=/home/ad.msoe.edu/manchadoa/isaac-sim-msoe/training/logs/finetune_%j.err
#SBATCH --time=08:00:00
#SBATCH --account=undergrad_research

# Fine-tune Nova Carter policy in the Innovation Center building.
#
# Usage:
#   # Stage A (Floor G, load stage4 base):
#   sbatch training/submit_finetune_rl.sh \
#       --checkpoint training/logs/sb3/NovaCarter-Patrol/2026-04-08_04-35-47/best_model_stage4.zip \
#       --stage A
#
#   # Stage B (Floor 1, load Stage A best):
#   sbatch training/submit_finetune_rl.sh \
#       --checkpoint training/logs/sb3/NovaCarter-Building/stageA_<timestamp>/best_model_building_stageA.zip \
#       --stage B
#
#   # Stage C (all floors, load Stage B best):
#   sbatch training/submit_finetune_rl.sh \
#       --checkpoint training/logs/sb3/NovaCarter-Building/stageB_<timestamp>/best_model_building_stageB.zip \
#       --stage C
#
# NOTE: num_envs=1 — T4 (16GB) cannot hold 77k-mesh BVH duplicated per env.
# 8hr time limit is generous for 500k steps at single-env throughput.

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

echo "[INFO] Starting building fine-tuning job $SLURM_JOB_ID"
echo "[INFO] Args: $@"
nvidia-smi

cd "$ISAAC_LAB_DIR"
./isaaclab.sh -p "$PROJECT_DIR/training/train_nova_carter_building.py" \
    --num_envs 1 \
    --total_timesteps 500000 \
    "$@"

echo "[INFO] Job $SLURM_JOB_ID finished."
