#!/bin/bash
# Fine-tune Qwen2-VL-2B on the unified 4-class door+glass dataset.
# Targets dgx (V100 32 GB). finetune_qwen2vl.py auto-selects SDPA because
# V100 is compute capability 7.0 (<8.0, the FA2 threshold).
#
# Parallel to submit_finetune_unified.sh (which trains Qwen3-VL). Both
# consume the same data/{train,val}_unified.jsonl so results are directly
# comparable. Uses finetune_qwen2vl.py (single visual.merger projector;
# no deepstack mergers).
#
# Inputs  : data/train_unified.jsonl + data/val_unified.jsonl
# Output  : data/qwen2vl_unified_lora_v4/
#
# Produced by: sbatch submit_prepare_unified.sh
#
# Usage:
#   sbatch submit_finetune_unified_qwen2vl.sh
#   sbatch submit_finetune_unified_qwen2vl.sh --epochs 2 --lr 5e-6

#SBATCH --job-name=qwen2vl-unified
#SBATCH --partition=dgx
#SBATCH --gres=gpu:1
#SBATCH --mem=32G
#SBATCH --cpus-per-task=4
#SBATCH --time=12:00:00
#SBATCH --output=/home/ad.msoe.edu/celliot/Digital-twin-v2/isaac-sim-msoe/logs/finetune_unified_qwen2vl_%j.out
#SBATCH --error=/home/ad.msoe.edu/celliot/Digital-twin-v2/isaac-sim-msoe/logs/finetune_unified_qwen2vl_%j.out

set -euo pipefail
SCRIPT_DIR="/home/ad.msoe.edu/celliot/Digital-twin-v2/isaac-sim-msoe"
cd "$SCRIPT_DIR"
mkdir -p "$SCRIPT_DIR/logs"

echo "=== Qwen2-VL-2B 4-Class Unified Fine-Tune ==="
echo "Node:      $(hostname)"
echo "Started:   $(date)"
echo "GPU:       $(nvidia-smi --query-gpu=name --format=csv,noheader 2>/dev/null || echo 'unknown')"
echo ""

PYTHON="/home/ad.msoe.edu/celliot/.conda/envs/vlm/bin/python"
if [ ! -f "$PYTHON" ]; then
    echo "ERROR: vlm conda env not found at $PYTHON"; exit 1
fi
echo "Python: $($PYTHON --version)"
echo ""

# Sanity check inputs
for f in data/train_unified.jsonl data/val_unified.jsonl; do
    if [ ! -f "$f" ]; then
        echo "ERROR: $f missing. Run: sbatch submit_prepare_unified.sh"
        exit 1
    fi
done

TRAIN_N=$(wc -l < data/train_unified.jsonl)
VAL_N=$(wc -l < data/val_unified.jsonl)
echo "Dataset:  train=$TRAIN_N  val=$VAL_N"
echo ""

echo "--- Fine-tuning ---"
$PYTHON simulations/robot/finetune_qwen2vl.py \
    --train_jsonl train_unified.jsonl \
    --val_jsonl   val_unified.jsonl \
    --output_dir  data/qwen2vl_unified_lora_v4 \
    "$@"
FT_EXIT=$?

echo ""
echo "Fine-tune exit code: $FT_EXIT"
echo "Finished: $(date)"

if [ $FT_EXIT -eq 0 ]; then
    echo "Adapter saved → data/qwen2vl_unified_lora_v4/"
    echo "Next: adapt submit_eval_unified.sh to point --adapter at qwen2vl_unified_lora_v4/"
fi

exit $FT_EXIT
