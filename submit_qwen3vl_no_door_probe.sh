#!/bin/bash
# submit_qwen3vl_no_door_probe.sh — Catastrophic-forgetting probe.
#
# Runs the unified Qwen3-VL fine-tune on 3 doorless / windowless scene images
# with a branched prompt: keep the original training question and append a
# fallback that asks for a 1-2 sentence description if no door is visible.
# A model that retained general world knowledge will switch to descriptive
# mode; a model that's been over-specialized will collapse to "Door Status: ..."
#
# Also runs --prompt-mode all, which adds two pure description prompts as a
# control — disentangles "instruction following" from "general capability".
#
# Usage:
#   sbatch submit_qwen3vl_no_door_probe.sh

#SBATCH --job-name=qwen3vl-probe
#SBATCH --partition=dgx
#SBATCH --gres=gpu:1
#SBATCH --mem=32G
#SBATCH --cpus-per-task=4
#SBATCH --time=0:30:00
#SBATCH --output=/home/ad.msoe.edu/celliot/Digital-twin-v2/isaac-sim-msoe/logs/no_door_probe_%j.out
#SBATCH --error=/home/ad.msoe.edu/celliot/Digital-twin-v2/isaac-sim-msoe/logs/no_door_probe_%j.out

set -euo pipefail

SCRIPT_DIR="/home/ad.msoe.edu/celliot/Digital-twin-v2/isaac-sim-msoe"
cd "$SCRIPT_DIR"

mkdir -p "$SCRIPT_DIR/logs"

PYTHON="/home/ad.msoe.edu/celliot/.conda/envs/vlm/bin/python"
ADAPTER="data/qwen3vl_unified_lora_v4"
IMG_DIR="data/no_window_no_door_image"

echo "=== Qwen3-VL-2B No-Door / Catastrophic-Forgetting Probe ==="
echo "Node:     $(hostname)"
echo "Started:  $(date)"
echo "GPU:      $(nvidia-smi --query-gpu=name --format=csv,noheader 2>/dev/null || echo 'unknown')"
echo "Adapter:  $ADAPTER"
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

# 'all' mode runs three prompts per image:
#   1. door-classification (training prompt verbatim)
#   2. open-ended-describe ("Describe this image in detail.")
#   3. open-ended-objects  ("What objects do you see in this image?")
# 'branched' mode runs the single conditional prompt the user requested.
echo "================================================================"
echo "  PROMPT MODE: unified — matches qwen3vl_unified_lora_v4 training"
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
    --adapter     "$ADAPTER" \
    --prompt-mode unified \
    "${IMAGES[@]}"

echo ""
echo "Finished: $(date)"
