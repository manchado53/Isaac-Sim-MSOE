#!/usr/bin/env python3
"""
Highlight wall boundaries on a floor plan image.

Detects dark pixels (walls) in the floor plan and recolors them red,
making it easier to see where walls are vs open space.

Usage:
    ~/isaacsim/python.sh floor_plans_resized/draw_borders.py floor_plans_resized/floor1/floor1.jpg

    # Process all images in a folder:
    ~/isaacsim/python.sh floor_plans_resized/draw_borders.py floor_plans_resized/floor1/

Output:
    <filename>_borders.jpg for each input image
"""

import os
import sys

import matplotlib
matplotlib.use("Agg")
import numpy as np
from PIL import Image


def highlight_walls(image_path, output_path=None, threshold=80):
    """
    Replace dark pixels (walls) with red, keep everything else.

    Args:
        image_path: Path to the floor plan image.
        output_path: Where to save. Defaults to <name>_borders.jpg.
        threshold: Pixels darker than this (0-255) are considered walls.
    """
    img = Image.open(image_path).convert("RGB")
    pixels = np.array(img)

    # A pixel is a "wall" if all three channels (R, G, B) are below the threshold
    # This catches black and dark gray lines
    is_wall = np.all(pixels < threshold, axis=2)

    # Copy the image and recolor walls to red
    result = pixels.copy()
    result[is_wall] = [255, 0, 0]  # Red

    if output_path is None:
        base, ext = os.path.splitext(image_path)
        output_path = base + "_borders.jpg"

    out_img = Image.fromarray(result)
    out_img.save(output_path, quality=95)
    print(f"Saved: {output_path}")


def main():
    if len(sys.argv) < 2:
        print("Usage: ~/isaacsim/python.sh draw_borders.py <image_or_folder>")
        sys.exit(1)

    target = sys.argv[1]

    if os.path.isdir(target):
        # Process all jpg files in the folder
        for filename in sorted(os.listdir(target)):
            if filename.lower().endswith(".jpg") and "_borders" not in filename and "_mapped" not in filename and "_grid" not in filename:
                filepath = os.path.join(target, filename)
                highlight_walls(filepath)
    elif os.path.isfile(target):
        highlight_walls(target)
    else:
        print(f"Not found: {target}")
        sys.exit(1)


if __name__ == "__main__":
    main()
