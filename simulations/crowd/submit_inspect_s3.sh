#!/bin/bash
#SBATCH --job-name=inspect-s3
#SBATCH --partition=teaching
#SBATCH --gres=gpu:t4:1
#SBATCH --time=0:10:00
#SBATCH --mem=16G
#SBATCH --output=%x_%j.out
#SBATCH --error=%x_%j.err
#SBATCH --account=students
#SBATCH --mail-user=somed@ad.msoe.edu
#SBATCH --mail-type=END,FAIL

cd "$HOME/isaac-sim-msoe"
"$HOME/isaacsim/python.sh" simulations/crowd/inspect_s3_character.py
echo "[INFO] Results: videos/s3_character_inspect.txt + videos/s3_character_test.png"
