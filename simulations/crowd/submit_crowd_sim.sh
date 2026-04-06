#!/bin/bash
#SBATCH --job-name=crowd-sim
#SBATCH --partition=teaching
#SBATCH --gres=gpu:t4:1
#SBATCH --time=1:00:00
#SBATCH --mem=32G
#SBATCH --output=%x_%j.out
#SBATCH --error=%x_%j.err
#SBATCH --account=students

# Kern Center crowd (student flow) simulation
# Submit with: cd ~/isaac-sim-msoe && sbatch simulations/crowd/submit_crowd_sim.sh

set -e

PROJECT_DIR="$HOME/isaac-sim-msoe"
ISAACSIM_DIR="$HOME/isaacsim"

# ── Environment ──
export OMNI_KIT_ACCEPT_EULA=YES
export PRIVACY_CONSENT=Y

echo "[INFO] Node:   $(hostname)"
echo "[INFO] GPU:    $(nvidia-smi --query-gpu=name --format=csv,noheader | head -1)"
echo "[INFO] Start:  $(date)"

cd "$PROJECT_DIR"

# ── Run crowd simulation ──
"$ISAACSIM_DIR/python.sh" simulations/crowd/crowd_simulation.py

echo "[INFO] Simulation done at $(date)"

# ── Run analysis on the latest log (no GPU needed, but convenient to chain) ──
"$ISAACSIM_DIR/python.sh" simulations/crowd/analyze_crowd.py

echo "[INFO] Analysis done at $(date)"
echo "[INFO] Figures in: $PROJECT_DIR/simulations/crowd/figures/"
