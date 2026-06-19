#!/bin/bash
# submit_finetune_qwen3vl.sh — SLURM job for Qwen3-VL-2B LoRA + projector fine-tune.
#
# Mirrors submit_finetune_job.sh (Qwen2-VL) but targets the Qwen3-VL pipeline.
# Qwen3-VL was the top zero-shot performer in the bright-only comparison
# (75.9% overall vs 64.6% for Qwen2-VL-2B), so we're rebasing the fine-tune on it.
#
# Usage:
#   sbatch submit_finetune_qwen3vl.sh
#   sbatch submit_finetune_qwen3vl.sh --epochs 5 --lr 5e-6
#
# Output:
#   data/qwen3vl_door_lora_v4/   (adapter — load with PeftModel.from_pretrained)

#SBATCH --job-name=qwen3vl-finetune
#SBATCH --partition=dgx
#SBATCH --gres=gpu:1
#SBATCH --mem=32G
#SBATCH --cpus-per-task=4
#SBATCH --time=12:00:00
#SBATCH --output=/home/ad.msoe.edu/celliot/Digital-twin-v2/isaac-sim-msoe/logs/finetune_qwen3vl_%j.out
#SBATCH --error=/home/ad.msoe.edu/celliot/Digital-twin-v2/isaac-sim-msoe/logs/finetune_qwen3vl_%j.out

set -euo pipefail

SCRIPT_DIR="/home/ad.msoe.edu/celliot/Digital-twin-v2/isaac-sim-msoe"
cd "$SCRIPT_DIR"

mkdir -p "$SCRIPT_DIR/logs"

echo "=== Qwen3-VL-2B Fine-Tuning Job ==="
echo "Node:      $(hostname)"
echo "Started:   $(date)"
echo "GPU:       $(nvidia-smi --query-gpu=name --format=csv,noheader 2>/dev/null || echo 'unknown')"
echo ""

PYTHON="/home/ad.msoe.edu/celliot/.conda/envs/vlm/bin/python"
if [ ! -f "$PYTHON" ]; then
    echo "ERROR: vlm conda env not found at $PYTHON"
    exit 1
fi
echo "Python: $($PYTHON --version)"
echo ""

# Verify production data shards exist (same v4 set used by Qwen2-VL fine-tune)
SHARD_COUNT=$(ls data/door_production_v4_train_Doors_*.npz 2>/dev/null | wc -l)
if [ "$SHARD_COUNT" -eq 0 ]; then
    echo "ERROR: no training shards found (data/door_production_v4_train_Doors_*.npz)."
    echo "  Run submit_production_collection.sh first."
    exit 1
fi
echo "Found $SHARD_COUNT training production shards"

# Regenerate splits on the compute node (NFS write-through guarantee)
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
VAL_N=$(wc -l   < data/val_v4.jsonl)
echo "Dataset:  train=$TRAIN_N (curated)  val=$VAL_N"
echo ""

echo "--- Fine-tuning ---"
$PYTHON simulations/robot/finetune_qwen3vl.py \
    --train_jsonl train_v4_curated.jsonl \
    --val_jsonl   val_v4.jsonl \
    --output_dir  data/qwen3vl_door_lora_v4_curated \
    "$@"
FT_EXIT=$?

echo ""
echo "Fine-tune exit code: $FT_EXIT"
echo "Finished: $(date)"

if [ $FT_EXIT -eq 0 ]; then
    echo "Adapter saved to: data/qwen3vl_door_lora_v4_curated/"
    echo ""
    echo "Next: evaluate the adapter on the same val split:"
    echo "  sbatch submit_baseline_eval.sh --adapter data/qwen3vl_door_lora_v4_curated"
fi

exit $FT_EXIT
