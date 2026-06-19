#!/bin/bash
# submit_eval_qwen3vl_zeroshot_holdout.sh — Zero-shot Qwen3-VL-2B baseline on
# the held-out-door eval set, for direct comparison against the fine-tuned run.
#
# Re-uses data/holdout_v4.jsonl produced by submit_eval_qwen3vl_holdout.sh.
# Submit with --dependency=afterok:<extraction_job_id> if extraction is still
# in flight, e.g.:
#   sbatch --dependency=afterok:244246 submit_eval_qwen3vl_zeroshot_holdout.sh
#
# Output:
#   data/holdout_eval_qwen3vl_2b_zeroshot.json
#   logs/eval_qwen3vl_zeroshot_holdout_<JOBID>.out

#SBATCH --job-name=qwen3vl-zs-hold
#SBATCH --partition=dgx
#SBATCH --gres=gpu:1
#SBATCH --mem=32G
#SBATCH --cpus-per-task=4
#SBATCH --time=8:00:00
#SBATCH --output=/home/ad.msoe.edu/celliot/Digital-twin-v2/isaac-sim-msoe/logs/eval_qwen3vl_zeroshot_holdout_%j.out
#SBATCH --error=/home/ad.msoe.edu/celliot/Digital-twin-v2/isaac-sim-msoe/logs/eval_qwen3vl_zeroshot_holdout_%j.out

set -euo pipefail

SCRIPT_DIR="/home/ad.msoe.edu/celliot/Digital-twin-v2/isaac-sim-msoe"
cd "$SCRIPT_DIR"

mkdir -p "$SCRIPT_DIR/logs"

PYTHON="/home/ad.msoe.edu/celliot/.conda/envs/vlm/bin/python"
JSONL="data/holdout_v4.jsonl"
OUT_PATH="data/holdout_eval_qwen3vl_2b_zeroshot.json"

echo "=== Qwen3-VL-2B ZERO-SHOT — HELD-OUT DOORS ==="
echo "Node:     $(hostname)"
echo "Started:  $(date)"
echo "GPU:      $(nvidia-smi --query-gpu=name --format=csv,noheader 2>/dev/null || echo 'unknown')"
echo "Python:   $($PYTHON --version)"
echo ""

if [ ! -f "$JSONL" ]; then
    echo "ERROR: $JSONL not found — run submit_eval_qwen3vl_holdout.sh first"
    exit 1
fi
echo "Holdout set: $JSONL  ($(wc -l < $JSONL) frames)"
echo ""

$PYTHON simulations/robot/eval_vlm_baseline.py \
    --model       Qwen/Qwen3-VL-2B-Instruct \
    --model_class qwen3_vl \
    --val_jsonl   "$JSONL" \
    --output      "$OUT_PATH"
EVAL_EXIT=$?

echo ""
echo "Eval exit code: $EVAL_EXIT"
echo "Finished: $(date)"
exit $EVAL_EXIT
