#!/bin/bash
# submit_curate_diversity.sh — DINOv2 embedding + diversity-based curation.
#
# Two stages, single SLURM job:
#   1. embed_train_frames.py — ~15-20 min on V100 (~25k frames @ DINOv2-base)
#   2. curate_diversity.py   — ~2-5 min, stratified k-means → 10k frames
#
# Input:  data/train_v4.jsonl  (produced by submit_prepare_data.sh on v4 NPZs)
# Output:
#   data/train_v4_embeddings.npy       (cached — re-runnable for different k)
#   data/train_v4_embedding_meta.json
#   data/train_v4_curated.jsonl        (drop-in replacement for train_v4.jsonl)
#   data/train_v4_curated_summary.json
#
# Usage:
#   sbatch submit_curate_diversity.sh
#   sbatch submit_curate_diversity.sh --k 5000          # different target size
#   sbatch submit_curate_diversity.sh --skip_embed      # only re-cluster

#SBATCH --job-name=curate-diversity
#SBATCH --partition=dgx
#SBATCH --gres=gpu:1
#SBATCH --mem=48G
#SBATCH --cpus-per-task=4
#SBATCH --time=2:00:00
#SBATCH --output=/home/ad.msoe.edu/celliot/Digital-twin-v2/isaac-sim-msoe/logs/curate_diversity_%j.out
#SBATCH --error=/home/ad.msoe.edu/celliot/Digital-twin-v2/isaac-sim-msoe/logs/curate_diversity_%j.out

set -euo pipefail

PROJECT="/home/ad.msoe.edu/celliot/Digital-twin-v2/isaac-sim-msoe"
PYTHON="/home/ad.msoe.edu/celliot/.conda/envs/vlm/bin/python"
cd "$PROJECT"

echo "=== Diversity Curation ==="
echo "Node:     $(hostname)"
echo "Started:  $(date)"
echo "GPU:      $(nvidia-smi --query-gpu=name --format=csv,noheader 2>/dev/null || echo 'unknown')"
echo "Python:   $($PYTHON --version)"
echo ""

# Parse args
K=10000
SKIP_EMBED=0
PASSTHRU=()
while [[ $# -gt 0 ]]; do
    case "$1" in
        --k)          K="$2"; shift 2 ;;
        --skip_embed) SKIP_EMBED=1; shift ;;
        *)            PASSTHRU+=("$1"); shift ;;
    esac
done

EMB_PATH="$PROJECT/data/train_v4_embeddings.npy"
INPUT_JSONL="$PROJECT/data/train_v4.jsonl"

if [ ! -f "$INPUT_JSONL" ]; then
    echo "ERROR: $INPUT_JSONL not found."
    echo "  Wait for submit_prepare_data.sh to finish before running curation."
    exit 1
fi
echo "Input: $INPUT_JSONL  ($(wc -l < "$INPUT_JSONL") frames)"
echo ""

# ------------------------------------------------------------------
# Ensure faiss is available — install on first run
# ------------------------------------------------------------------
if ! $PYTHON -c "import faiss" 2>/dev/null; then
    echo "--- Installing faiss-cpu (one-time) ---"
    $PYTHON -m pip install --quiet faiss-cpu
fi

# ------------------------------------------------------------------
# Stage 1: embed (skip if cached and --skip_embed)
# ------------------------------------------------------------------
if [ "$SKIP_EMBED" -eq 1 ] && [ -f "$EMB_PATH" ]; then
    echo "--- Skipping embedding (using cached $EMB_PATH) ---"
else
    if [ -f "$EMB_PATH" ]; then
        echo "--- Embeddings already exist at $EMB_PATH"
        echo "    Use --skip_embed to reuse, or delete to regenerate. Regenerating ---"
    fi
    echo "--- Stage 1: DINOv2 embeddings ---"
    $PYTHON simulations/robot/embed_train_frames.py
    echo ""
fi

# ------------------------------------------------------------------
# Stage 2: curate
# ------------------------------------------------------------------
echo "--- Stage 2: Stratified k-means curation (k=$K) ---"
$PYTHON simulations/robot/curate_diversity.py --k "$K" "${PASSTHRU[@]}"

echo ""
echo "Finished: $(date)"
