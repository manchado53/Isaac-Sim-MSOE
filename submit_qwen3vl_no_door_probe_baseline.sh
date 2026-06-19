#!/bin/bash
# submit_qwen3vl_no_door_probe_baseline.sh — Same probe, baseline model.
#
# Mirror of submit_qwen3vl_no_door_probe.sh but runs the off-the-shelf
# Qwen3-VL-2B-Instruct with NO LoRA adapter. Lets us compare the fine-tuned
# adapter's behavior against the base model on the same 3 doorless/glassless
# scene images and the same prompts (training, branched, open-ended).
#
# Usage:
#   sbatch submit_qwen3vl_no_door_probe_baseline.sh

#SBATCH --job-name=qwen3vl-probe-base
#SBATCH --partition=dgx
#SBATCH --gres=gpu:1
#SBATCH --mem=32G
#SBATCH --cpus-per-task=4
#SBATCH --time=0:30:00
#SBATCH --output=/home/ad.msoe.edu/celliot/Digital-twin-v2/isaac-sim-msoe/logs/no_door_probe_baseline_%j.out
#SBATCH --error=/home/ad.msoe.edu/celliot/Digital-twin-v2/isaac-sim-msoe/logs/no_door_probe_baseline_%j.out

set -euo pipefail

SCRIPT_DIR="/home/ad.msoe.edu/celliot/Digital-twin-v2/isaac-sim-msoe"
cd "$SCRIPT_DIR"

mkdir -p "$SCRIPT_DIR/logs"

PYTHON="/home/ad.msoe.edu/celliot/.conda/envs/vlm/bin/python"
IMG_DIR="data/no_window_no_door_image"

echo "=== Qwen3-VL-2B BASELINE (no adapter) — No-Door Probe ==="
echo "Node:     $(hostname)"
echo "Started:  $(date)"
echo "GPU:      $(nvidia-smi --query-gpu=name --format=csv,noheader 2>/dev/null || echo 'unknown')"
echo "Adapter:  none (off-the-shelf Qwen3-VL-2B-Instruct)"
echo "Images:   $IMG_DIR"
echo ""

IMAGES=()
for f in "$IMG_DIR"/*.png "$IMG_DIR"/*.jpg "$IMG_DIR"/*.jpeg; do
    [ -f "$f" ] && IMAGES+=("--image" "$f")
done
if [ ${#IMAGES[@]} -eq 0 ]; then
    echo "ERROR: no images in $IMG_DIR"; exit 1
fi
echo "Found $(( ${#IMAGES[@]} / 2 )) image(s)"
echo ""

echo "================================================================"
echo "  PROMPT MODE: unified — same prompts as fine-tuned probe"
echo "  Prompts:"
echo "    1. unified-classification — training prompt verbatim"
echo "       'What is shown? Options: door open, door closed,"
echo "        glass cracked, glass intact.'"
echo "    2. unified-branched — same + 'if none, describe in at most 20 words'"
echo "    3. open-ended-describe — 'Describe this image in 1-2 sentences.'"
echo "       (control: pure world-knowledge probe, no door/glass framing)"
echo "================================================================"
$PYTHON simulations/robot/infer_single_image.py \
    --model       Qwen/Qwen3-VL-2B-Instruct \
    --model-class qwen3_vl \
    --prompt-mode unified \
    "${IMAGES[@]}"

echo ""
echo "Finished: $(date)"
