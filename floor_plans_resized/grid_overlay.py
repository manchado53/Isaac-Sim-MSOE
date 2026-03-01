#!/usr/bin/env python3
"""
Generate a floor plan image with a coordinate grid overlay.

This creates a version of the floor plan with gridlines and coordinate
labels so you can read off (x, y) pixel positions for node placement.

Usage:
    ~/isaacsim/python.sh floor_plans_resized/grid_overlay.py floor_plans_resized/floor1.jpg

Output:
    floor1_grid.jpg — the floor plan with a coordinate grid on top
"""

import os
import sys

import matplotlib
matplotlib.use("Agg")  # No display needed
import matplotlib.pyplot as plt
import matplotlib.image as mpimg


def main():
    if len(sys.argv) < 2:
        print("Usage: ~/isaacsim/python.sh grid_overlay.py <floor_plan_image>")
        sys.exit(1)

    image_path = sys.argv[1]
    if not os.path.exists(image_path):
        print(f"Error: Image not found: {image_path}")
        sys.exit(1)

    img = mpimg.imread(image_path)
    h, w = img.shape[:2]

    base = os.path.splitext(image_path)[0]
    output_path = base + "_grid.jpg"

    fig, ax = plt.subplots(1, 1, figsize=(16, 12))
    ax.imshow(img)

    # Draw grid every 50 pixels
    grid_spacing = 50
    for x in range(0, w, grid_spacing):
        ax.axvline(x=x, color="cyan", alpha=0.3, linewidth=0.5)
        if x % 100 == 0:
            ax.text(x, 10, str(x), fontsize=5, color="cyan", ha="center", va="top")
    for y in range(0, h, grid_spacing):
        ax.axhline(y=y, color="cyan", alpha=0.3, linewidth=0.5)
        if y % 100 == 0:
            ax.text(5, y, str(y), fontsize=5, color="cyan", ha="left", va="center")

    ax.set_title(f"Grid Overlay — {w}x{h} pixels (grid every {grid_spacing}px, labels every 100px)")
    ax.set_xlim(0, w)
    ax.set_ylim(h, 0)

    fig.savefig(output_path, dpi=200, bbox_inches="tight")
    plt.close(fig)
    print(f"Saved grid overlay to: {output_path}")
    print(f"Image dimensions: {w}w x {h}h pixels")
    print()
    print("Use the grid to read off (x, y) coordinates for nodes.")
    print("Then add them to floor1_nodes.json or run map_nodes.py.")


if __name__ == "__main__":
    main()
