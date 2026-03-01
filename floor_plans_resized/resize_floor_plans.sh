#!/bin/bash
# Resize floor plan images to smaller versions for easier viewing
# Uses ImageMagick's convert command

INPUT_DIR="/home/ad.msoe.edu/loducaj/isaac-sim-msoe/floor_plans"
OUTPUT_DIR="/home/ad.msoe.edu/loducaj/isaac-sim-msoe/floor_plans_resized"

for f in "$INPUT_DIR"/*.jpg; do
    filename=$(basename "$f")
    echo "Resizing $filename..."
    convert "$f" -resize 1200x1200 -quality 70 "$OUTPUT_DIR/$filename"
done

echo "Done. Resized images saved to $OUTPUT_DIR"
