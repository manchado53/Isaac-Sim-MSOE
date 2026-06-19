#!/bin/bash
# submit_vlm_comparison.sh — zero-shot eval array for the VLM comparison.
#
# Runs candidate base models against a validation JSONL (default: data/val.jsonl)
# using the same prompt / parser / confidence-extraction logic as the existing
# Qwen2-VL-2B reference.
#
# Array index → model:
#   0  Qwen/Qwen3-VL-2B-Instruct
#   1  HuggingFaceTB/SmolVLM2-2.2B-Instruct
#   2  OpenGVLab/InternVL3-2B-hf   (HF-native port; original repo needs custom .chat() API)
#   3  Qwen/Qwen2.5-VL-3B-Instruct
#   4  Qwen/Qwen2.5-VL-7B-Instruct     (tight on V100 32GB — may need dgxh100)
#   5  Qwen/Qwen2-VL-2B-Instruct       (matches the existing reference run)
#
# Submit (original val.jsonl, models 0-4):
#   sbatch submit_vlm_comparison.sh
#
# Submit a single model only:
#   sbatch --array=4 submit_vlm_comparison.sh
#
# Smoke test (eval only first 20 samples):
#   sbatch --array=2 --export=ALL,LIMIT=20 submit_vlm_comparison.sh
#
# Rerun on a different val split (e.g. the brightness-filtered subset), with
# a suffix appended to output filenames so nothing is overwritten:
#   sbatch --array=0-5 \
#          --export=ALL,VAL_JSONL=data/val_bright_only.jsonl,OUT_SUFFIX=_bright \
#          submit_vlm_comparison.sh
#
# Results land at data/baseline_eval_<slug>${OUT_SUFFIX}.json, one per model.

#SBATCH --job-name=vlm-compare
#SBATCH --partition=dgx
#SBATCH --gres=gpu:1
#SBATCH --mem=48G
#SBATCH --cpus-per-task=4
#SBATCH --time=4:00:00
#SBATCH --array=0-4
#SBATCH --output=/home/ad.msoe.edu/celliot/Digital-twin-v2/isaac-sim-msoe/logs/vlm_compare_%A_%a.out
#SBATCH --error=/home/ad.msoe.edu/celliot/Digital-twin-v2/isaac-sim-msoe/logs/vlm_compare_%A_%a.out

set -euo pipefail

SCRIPT_DIR="/home/ad.msoe.edu/celliot/Digital-twin-v2/isaac-sim-msoe"
cd "$SCRIPT_DIR"

# Array index → (model_id, model_class, output_slug)
MODELS=(
    "Qwen/Qwen3-VL-2B-Instruct|qwen3_vl|qwen3_vl_2b"
    "HuggingFaceTB/SmolVLM2-2.2B-Instruct|smolvlm2|smolvlm2_2_2b"
    "OpenGVLab/InternVL3-2B-hf|internvl3|internvl3_2b"
    "Qwen/Qwen2.5-VL-3B-Instruct|qwen2_5_vl|qwen2_5_vl_3b"
    "Qwen/Qwen2.5-VL-7B-Instruct|qwen2_5_vl|qwen2_5_vl_7b"
    "Qwen/Qwen2-VL-2B-Instruct|qwen2_vl|qwen2_vl_2b"
)

IFS='|' read -r MODEL_ID MODEL_CLASS OUT_SLUG <<< "${MODELS[$SLURM_ARRAY_TASK_ID]}"
OUT_PATH="data/baseline_eval_${OUT_SLUG}${OUT_SUFFIX:-}.json"

echo "=== VLM Comparison — task ${SLURM_ARRAY_TASK_ID} ==="
echo "Node:       $(hostname)"
echo "Started:    $(date)"
echo "GPU:        $(nvidia-smi --query-gpu=name,memory.total --format=csv,noheader 2>/dev/null || echo 'unknown')"
echo "Model:      $MODEL_ID"
echo "Class:      $MODEL_CLASS"
echo "Output:     $OUT_PATH"
echo ""

TEST_JSONL="${VAL_JSONL:-data/val.jsonl}"
if [ ! -f "$TEST_JSONL" ]; then
    echo "ERROR: $TEST_JSONL not found."
    exit 1
fi
echo "Val set: $TEST_JSONL  ($(wc -l < "$TEST_JSONL") frames)"
echo "Suffix:  ${OUT_SUFFIX:-<none>}"
echo ""

PYTHON="/home/ad.msoe.edu/celliot/.conda/envs/vlm/bin/python"
if [ ! -f "$PYTHON" ]; then
    echo "ERROR: vlm conda env not found at $PYTHON"
    exit 1
fi
echo "Python: $($PYTHON --version)"
echo ""

echo "--- Evaluation ---"
# Optional smoke-test: sbatch --export=ALL,LIMIT=20 ...  → eval only first 20 samples
LIMIT_ARG=()
if [ -n "${LIMIT:-}" ]; then
    LIMIT_ARG=(--limit "$LIMIT")
    OUT_PATH="${OUT_PATH%.json}_limit${LIMIT}.json"
    echo "LIMIT mode: ${LIMIT} samples → $OUT_PATH"
fi
$PYTHON simulations/robot/eval_vlm_baseline.py \
    --model "$MODEL_ID" \
    --model_class "$MODEL_CLASS" \
    --val_jsonl "$TEST_JSONL" \
    --output "$OUT_PATH" \
    "${LIMIT_ARG[@]}"
EVAL_EXIT=$?

echo ""
echo "Eval exit code: $EVAL_EXIT"
echo "Finished: $(date)"

exit $EVAL_EXIT
