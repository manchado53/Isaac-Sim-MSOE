#!/bin/bash
# submit_finetune_job.sh — SLURM job for Qwen2-VL LoRA fine-tuning (no Isaac Sim).
#
# Targets dgxh100 partition (H100) for Flash Attention 2 support.
# Falls back to SDPA automatically on T4 (teaching partition) if you change --partition.
#
# Usage:
#   sbatch submit_finetune_job.sh
#   sbatch submit_finetune_job.sh --epochs 5 --lr 5e-6   # pass args to finetune_qwen2vl.py
#
# Prerequisites:
#   1. Run nova_carter_door_collection.py to produce data/door_labels.npz
#   2. Run prepare_finetune_data.py to produce data/train.jsonl + data/val.jsonl
#   3. Ensure peft, scikit-learn are installed in the Python env

#SBATCH --job-name=qwen2vl-finetune
#SBATCH --partition=dgx
#SBATCH --gres=gpu:1
#SBATCH --mem=32G
#SBATCH --cpus-per-task=4
#SBATCH --time=4:00:00
#SBATCH --output=/home/ad.msoe.edu/celliot/Digital-twin-v2/isaac-sim-msoe/logs/finetune_%j.out
#SBATCH --error=/home/ad.msoe.edu/celliot/Digital-twin-v2/isaac-sim-msoe/logs/finetune_%j.out

set -euo pipefail

SCRIPT_DIR="/home/ad.msoe.edu/celliot/Digital-twin-v2/isaac-sim-msoe"
cd "$SCRIPT_DIR"

mkdir -p "$SCRIPT_DIR/logs"

echo "=== Qwen2-VL Fine-Tuning Job ==="
echo "Node:      $(hostname)"
echo "Started:   $(date)"
echo "GPU:       $(nvidia-smi --query-gpu=name --format=csv,noheader 2>/dev/null || echo 'unknown')"
echo ""

# Check prerequisites
if [ ! -f "data/train.jsonl" ] || [ ! -f "data/val.jsonl" ]; then
    echo "ERROR: data/train.jsonl or data/val.jsonl not found."
    echo "  Run: python3 simulations/robot/prepare_finetune_data.py"
    exit 1
fi

TRAIN_N=$(wc -l < data/train.jsonl)
VAL_N=$(wc -l < data/val.jsonl)
echo "Dataset:  train=$TRAIN_N  val=$VAL_N"
echo ""

# Use conda 'vlm' env — does NOT use isaacsim/python.sh because fine-tuning
# does not need Isaac Sim and libcarb.so requires GLIBC 2.32+ not available
# on dgx/dgxh100 nodes. Set up once with:
#   conda create -n vlm python=3.10 -y
#   conda activate vlm
#   pip install torch torchvision --index-url https://download.pytorch.org/whl/cu118
#   pip install transformers>=4.45 peft>=0.12 pillow numpy scikit-learn accelerate
PYTHON="/home/ad.msoe.edu/celliot/.conda/envs/vlm/bin/python"
if [ ! -f "$PYTHON" ]; then
    echo "ERROR: vlm conda env not found at $PYTHON"
    exit 1
fi
echo "Python: $($PYTHON --version)"

echo "--- Fine-tuning ---"
$PYTHON simulations/robot/finetune_qwen2vl.py "$@"
FT_EXIT=$?

echo ""
echo "Fine-tune exit code: $FT_EXIT"
echo "Finished: $(date)"

if [ $FT_EXIT -eq 0 ]; then
    echo "Adapter saved to: data/qwen2vl_door_lora/"
    echo ""
    echo "Next: run inference with the adapter:"
    echo "  sbatch submit_isaac_job.sh simulations/robot/nova_carter_vlm_phase23.py --adapter data/qwen2vl_door_lora/"
fi

exit $FT_EXIT
