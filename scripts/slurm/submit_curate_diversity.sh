#!/bin/bash
# submit_curate_diversity.sh -- DINOv2 embedding + diversity-based curation
# for the cracked-glass train split.
#
# Two stages, single SLURM job:
#   1. embed_train_frames.py -- ~5-20 min on V100/T4 (~20k frames @ DINOv2-base)
#   2. curate_diversity.py   -- ~2-5 min, stratified k-means -> 10k frames
#
# Input:   <dataset>/labels_train.jsonl (produced by cracked_glass_collection.py)
# Output (alongside the JSONL):
#   train_embeddings.npy           (cached -- re-runnable for different k)
#   train_embedding_meta.json
#   train_curated.jsonl            (drop-in replacement for labels_train.jsonl)
#   train_curated_summary.json
#
# Usage:
#   sbatch submit_curate_diversity.sh --dataset datasets/cracked_glass/20261204_1500
#   sbatch submit_curate_diversity.sh --dataset <path> --k 5000
#   sbatch submit_curate_diversity.sh --dataset <path> --skip_embed

#SBATCH --job-name=cracked-glass-curate
#SBATCH --partition=dgx
#SBATCH --gres=gpu:1
#SBATCH --mem=48G
#SBATCH --cpus-per-task=4
#SBATCH --time=2:00:00
#SBATCH --output=logs/cracked_glass_curate_%j.out
#SBATCH --error=logs/cracked_glass_curate_%j.out

set -euo pipefail

PROJECT="${PROJECT:-$HOME/isaac-sim-msoe}"
PYTHON="${PYTHON:-python3}"
cd "$PROJECT"

echo "=== Cracked-Glass Diversity Curation ==="
echo "Node:     $(hostname)"
echo "Started:  $(date)"
echo "GPU:      $(nvidia-smi --query-gpu=name --format=csv,noheader 2>/dev/null || echo 'unknown')"
echo "Python:   $($PYTHON --version)"
echo ""

# Parse args -- separate our flags from pass-through for curate_diversity.py.
DATASET=""
K=10000
SKIP_EMBED=0
PASSTHRU=()
while [[ $# -gt 0 ]]; do
    case "$1" in
        --dataset)    DATASET="$2"; shift 2 ;;
        --k)          K="$2"; shift 2 ;;
        --skip_embed) SKIP_EMBED=1; shift ;;
        *)            PASSTHRU+=("$1"); shift ;;
    esac
done

if [ -z "$DATASET" ]; then
    echo "ERROR: --dataset <path> required (e.g. datasets/cracked_glass/20261204_1500)"
    exit 1
fi

DATASET_ABS="$(cd "$DATASET" && pwd)"
INPUT_JSONL="$DATASET_ABS/labels_train.jsonl"
EMB_PATH="$DATASET_ABS/train_embeddings.npy"

if [ ! -f "$INPUT_JSONL" ]; then
    echo "ERROR: $INPUT_JSONL not found."
    echo "  Run cracked_glass_collection.py first, then split labels.jsonl by split field."
    exit 1
fi
echo "Dataset:  $DATASET_ABS"
echo "Input:    $INPUT_JSONL  ($(wc -l < "$INPUT_JSONL") frames)"
echo ""

# Ensure faiss is available -- install on first run.
if ! $PYTHON -c "import faiss" 2>/dev/null; then
    echo "--- Installing faiss-cpu (one-time) ---"
    $PYTHON -m pip install --quiet faiss-cpu
fi

# Stage 1: embed (skip if cached and --skip_embed).
if [ "$SKIP_EMBED" -eq 1 ] && [ -f "$EMB_PATH" ]; then
    echo "--- Skipping embedding (using cached $EMB_PATH) ---"
else
    if [ -f "$EMB_PATH" ]; then
        echo "--- Embeddings already exist at $EMB_PATH"
        echo "    Use --skip_embed to reuse, or delete to regenerate. Regenerating ---"
    fi
    echo "--- Stage 1: DINOv2 embeddings ---"
    $PYTHON SyntheticDataGen/cracked_glass/embed_train_frames.py \
        --jsonl "$INPUT_JSONL"
    echo ""
fi

# Stage 2: curate.
echo "--- Stage 2: Stratified k-means curation (k=$K) ---"
$PYTHON SyntheticDataGen/cracked_glass/curate_diversity.py \
    --jsonl "$INPUT_JSONL" \
    --k "$K" \
    "${PASSTHRU[@]}"

echo ""
echo "Finished: $(date)"
