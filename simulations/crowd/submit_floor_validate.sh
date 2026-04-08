#!/bin/bash
#SBATCH --job-name=floorval
#SBATCH --partition=teaching
#SBATCH --gres=gpu:t4:1
#SBATCH --time=0:15:00
#SBATCH --mem=32G
#SBATCH --output=%x_%j.out
#SBATCH --error=%x_%j.err
#SBATCH --account=students

set -e
cd "$HOME/isaac-sim-msoe"
echo "[INFO] Node: $(hostname)"
echo "[INFO] Start: $(date)"
"$HOME/isaacsim/python.sh" simulations/crowd/validate_floor_z.py
echo "[INFO] Done: $(date)"
