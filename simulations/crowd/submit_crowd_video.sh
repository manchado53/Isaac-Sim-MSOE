#!/bin/bash
#SBATCH --job-name=crowd-video
#SBATCH --partition=teaching
#SBATCH --gres=gpu:t4:1
#SBATCH --time=1:00:00
#SBATCH --mem=32G
#SBATCH --output=%x_%j.out
#SBATCH --error=%x_%j.err
#SBATCH --account=students
#SBATCH --mail-user=somed@msoe.edu
#SBATCH --mail-type=END,FAIL

# Records a 20-second hallway-camera crowd video
# Submit with: cd ~/isaac-sim-msoe && sbatch simulations/crowd/submit_crowd_video.sh

set -e

PROJECT_DIR="$HOME/isaac-sim-msoe"
ISAACSIM_DIR="$HOME/isaacsim"

export OMNI_KIT_ACCEPT_EULA=YES
export PRIVACY_CONSENT=Y

echo "[INFO] Node: $(hostname)"
echo "[INFO] GPU:  $(nvidia-smi --query-gpu=name --format=csv,noheader | head -1)"
echo "[INFO] Start: $(date)"

cd "$PROJECT_DIR"

"$ISAACSIM_DIR/python.sh" simulations/crowd/crowd_simulation_video.py

echo "[INFO] Done at $(date)"
echo "[INFO] Video in: $PROJECT_DIR/videos/"
