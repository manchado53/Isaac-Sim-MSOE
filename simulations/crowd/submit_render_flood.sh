#!/bin/bash
#SBATCH --job-name=render-flood
#SBATCH --partition=teaching
#SBATCH --gres=gpu:t4:1
#SBATCH --time=0:40:00
#SBATCH --mem=32G
#SBATCH --output=%x_%j.out
#SBATCH --error=%x_%j.err
#SBATCH --account=students
#SBATCH --mail-user=somed@ad.msoe.edu
#SBATCH --mail-type=END,FAIL

cd "$HOME/isaac-sim-msoe"
"$HOME/isaacsim/python.sh" simulations/crowd/render_flood.py
echo "[INFO] Output: videos/flood_render.mp4"
