#!/bin/bash
#SBATCH --job-name=inspect-hierarchy
#SBATCH --partition=teaching
#SBATCH --gres=gpu:t4:1
#SBATCH --time=0:15:00
#SBATCH --mem=16G
#SBATCH --output=%x_%j.out
#SBATCH --error=%x_%j.err
#SBATCH --account=students
#SBATCH --mail-user=somed@ad.msoe.edu
#SBATCH --mail-type=END,FAIL

cd "$HOME/isaac-sim-msoe"
"$HOME/isaacsim/python.sh" simulations/crowd/inspect_building_hierarchy.py
echo "[INFO] Results: videos/building_hierarchy.txt"
