#!/bin/bash
#SBATCH --job-name=edge-val
#SBATCH --partition=teaching
#SBATCH --gres=gpu:t4:1
#SBATCH --time=0:10:00
#SBATCH --mem=32G
#SBATCH --output=%x_%j.out
#SBATCH --error=%x_%j.err
#SBATCH --account=students
#SBATCH --mail-user=somed@ad.msoe.edu
#SBATCH --mail-type=END,FAIL

cd "$HOME/isaac-sim-msoe"
rm -f videos/edge_validation.txt
"$HOME/isaacsim/python.sh" simulations/crowd/validate_edges.py
