#!/bin/bash
# Fine-tune Qwen3-VL-2B on the unified 4-class door+glass dataset.
# Targets dgxh100 (H100 + FA2). Falls back to dgx by changing --partition.
#
# Qwen3-VL was the top zero-shot performer in the bright-only comparison
# (75.9% overall vs 64.6% for Qwen2-VL-2B), so the unified adapter is
# based on Qwen3-VL. Uses finetune_qwen3vl.py which handles the extra
# projector groups (visual.merger + visual.deepstack_merger_list).
#
# Inputs  : data/train_unified.jsonl + data/val_unified.jsonl
# Output  : data/qwen3vl_unified_lora_v4/
#
# Produced by: sbatch submit_prepare_unified.sh
#
# Usage:
#   sbatch submit_finetune_unified.sh
#   sbatch submit_finetune_unified.sh --epochs 2 --lr 5e-6

#SBATCH --job-name=qwen3vl-unified
#SBATCH --partition=dgxh100
#SBATCH --gres=gpu:1
#SBATCH --mem=32G
#SBATCH --cpus-per-task=4
#SBATCH --time=12:00:00
#SBATCH --output=/home/ad.msoe.edu/celliot/Digital-twin-v2/isaac-sim-msoe/logs/finetune_unified_%j.out
#SBATCH --error=/home/ad.msoe.edu/celliot/Digital-twin-v2/isaac-sim-msoe/logs/finetune_unified_%j.out

set -euo pipefail
SCRIPT_DIR="/home/ad.msoe.edu/celliot/Digital-twin-v2/isaac-sim-msoe"
cd "$SCRIPT_DIR"
mkdir -p "$SCRIPT_DIR/logs"

echo "=== Qwen3-VL-2B 4-Class Unified Fine-Tune ==="
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
$PYTHON simulations/robot/finetune_qwen3vl.py \
    --train_jsonl train_unified.jsonl \
    --val_jsonl   val_unified.jsonl \
    --output_dir  data/qwen3vl_unified_lora_v4 \
    "$@"
FT_EXIT=$?

echo ""
echo "Fine-tune exit code: $FT_EXIT"
echo "Finished: $(date)"

if [ $FT_EXIT -eq 0 ]; then
    echo "Adapter saved → data/qwen3vl_unified_lora_v4/"
    echo "Next: sbatch submit_eval_unified.sh"
fi

exit $FT_EXIT
