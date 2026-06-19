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
#   1. Run submit_production_collection.sh to produce
#      data/door_production_Doors_*.npz (one per training door)
#   2. This script runs prepare_finetune_data.py on the compute node to
#      produce data/train.jsonl + data/val.jsonl (85/15 split)
#   3. Ensure peft, scikit-learn are installed in the Python env

#SBATCH --job-name=qwen2vl-finetune
#SBATCH --partition=dgxh100
#SBATCH --gres=gpu:1
#SBATCH --mem=32G
#SBATCH --cpus-per-task=4
#SBATCH --time=12:00:00
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
echo ""

# Check production shards exist (v4 = 5-arc grid + manual occlusion review)
SHARD_COUNT=$(ls data/door_production_v4_train_Doors_*.npz 2>/dev/null | wc -l)
if [ "$SHARD_COUNT" -eq 0 ]; then
    echo "ERROR: no training shards found (data/door_production_v4_train_Doors_*.npz)."
    echo "  Run submit_production_collection.sh first."
    exit 1
fi
echo "Found $SHARD_COUNT training production shards (holdout shards are excluded)"

# Regenerate train/val splits on this compute node to avoid NFS cache issues.
# Running prepare_finetune_data.py here (rather than relying on pre-generated files
# from the login node) guarantees the JSONL files are fully flushed before training.
echo "--- Generating data splits ---"
$PYTHON simulations/robot/prepare_finetune_data.py
echo ""

# Training on the DINOv2-diversity-curated subset (10k frames, 50/50 balanced).
# Val stays uncurated so it remains a clean independent eval.
if [ ! -f data/train_v4_curated.jsonl ]; then
    echo "ERROR: data/train_v4_curated.jsonl not found."
    echo "  Run submit_curate_diversity.sh first to produce the curated train split."
    exit 1
fi

TRAIN_N=$(wc -l < data/train_v4_curated.jsonl)
VAL_N=$(wc -l < data/val_v4.jsonl)
echo "Dataset:  train=$TRAIN_N (curated)  val=$VAL_N"
echo ""

echo "--- Fine-tuning ---"
$PYTHON simulations/robot/finetune_qwen2vl.py \
    --train_jsonl train_v4_curated.jsonl \
    --val_jsonl   val_v4.jsonl \
    --output_dir  data/qwen2vl_door_lora_v4_curated \
    "$@"
FT_EXIT=$?

echo ""
echo "Fine-tune exit code: $FT_EXIT"
echo "Finished: $(date)"

if [ $FT_EXIT -eq 0 ]; then
    echo "Adapter saved to: data/qwen2vl_door_lora_v4_curated/"
    echo ""
    echo "Next: run inference with the adapter:"
    echo "  sbatch submit_isaac_job.sh simulations/robot/nova_carter_vlm_phase23.py --adapter data/qwen2vl_door_lora_v4_curated/"
fi

exit $FT_EXIT
