#!/usr/bin/env python3
"""
Grid Overlay Tool

Draws a coordinate grid on the occupancy map with labeled world coordinates (cm).
Use this to visually identify coordinates for manual node placement.

    python3 test_building/test_nodes/grid_overlay.py
"""

import os
import json
import numpy as np
from PIL import Image, ImageDraw, ImageFont

# ---------------------------------------------------------------------------
# CONFIG
# ---------------------------------------------------------------------------
FLOOR = "floor_1"
GRID_SPACING_CM = 1000  # grid line every 1000 cm (10 meters)
GRID_COLOR = (0, 120, 255, 80)  # semi-transparent blue
LABEL_COLOR = (0, 120, 255)
MAJOR_GRID_SPACING_CM = 5000  # thicker line every 5000 cm (50 meters)
MAJOR_GRID_COLOR = (0, 80, 200, 140)

# ---------------------------------------------------------------------------
# Paths
# ---------------------------------------------------------------------------
TEST_NODES_DIR = os.path.dirname(os.path.abspath(__file__))


def world_to_pixel(wx, wy, xy_min, cell_size):
    col = (wx - xy_min[0]) / cell_size
    row = (wy - xy_min[1]) / cell_size
    return int(round(col)), int(round(row))


def main():
    meta_path = os.path.join(TEST_NODES_DIR, f"{FLOOR}_meta.json")
    with open(meta_path) as f:
        meta = json.load(f)

    xy_min = meta["xy_min"]
    xy_max = meta["xy_max"]
    cell_size = meta["cell_size_cm"]
    grid_w = meta["grid_width"]
    grid_h = meta["grid_height"]

    # Load base image
    occ_img_path = os.path.join(TEST_NODES_DIR, f"{FLOOR}_occupancy.png")
    base = Image.open(occ_img_path).convert("RGBA")

    # Create transparent overlay for grid lines
    overlay = Image.new("RGBA", base.size, (0, 0, 0, 0))
    draw = ImageDraw.Draw(overlay)

    # Draw vertical lines (constant X)
    x_start = int(np.ceil(xy_min[0] / GRID_SPACING_CM) * GRID_SPACING_CM)
    x_end = int(np.floor(xy_max[0] / GRID_SPACING_CM) * GRID_SPACING_CM)

    for wx in range(x_start, x_end + 1, GRID_SPACING_CM):
        col, _ = world_to_pixel(wx, 0, xy_min, cell_size)
        if col < 0 or col >= grid_w:
            continue
        is_major = (wx % MAJOR_GRID_SPACING_CM == 0)
        color = MAJOR_GRID_COLOR if is_major else GRID_COLOR
        width = 3 if is_major else 1
        draw.line([(col, 0), (col, grid_h - 1)], fill=color, width=width)
        # Label at top
        label = f"X={wx}"
        draw.text((col + 4, 10), label, fill=LABEL_COLOR)

    # Draw horizontal lines (constant Y)
    y_start = int(np.ceil(xy_min[1] / GRID_SPACING_CM) * GRID_SPACING_CM)
    y_end = int(np.floor(xy_max[1] / GRID_SPACING_CM) * GRID_SPACING_CM)

    for wy in range(y_start, y_end + 1, GRID_SPACING_CM):
        _, row = world_to_pixel(0, wy, xy_min, cell_size)
        if row < 0 or row >= grid_h:
            continue
        is_major = (wy % MAJOR_GRID_SPACING_CM == 0)
        color = MAJOR_GRID_COLOR if is_major else GRID_COLOR
        width = 3 if is_major else 1
        draw.line([(0, row), (grid_w - 1, row)], fill=color, width=width)
        # Label at left
        label = f"Y={wy}"
        draw.text((10, row + 4), label, fill=LABEL_COLOR)

    # Composite overlay onto base
    result = Image.alpha_composite(base, overlay).convert("RGB")

    out_path = os.path.join(TEST_NODES_DIR, f"{FLOOR}_grid_overlay.png")
    result.save(out_path)
    print(f"Saved: {out_path}")
    print(f"Grid spacing: {GRID_SPACING_CM} cm ({GRID_SPACING_CM/100:.0f} m)")
    print(f"World X range: {xy_min[0]:.0f} to {xy_max[0]:.0f} cm")
    print(f"World Y range: {xy_min[1]:.0f} to {xy_max[1]:.0f} cm")
    print(f"\nUse the grid lines to read off (X, Y) coordinates,")
    print(f"then put them in place_nodes.py MANUAL_NODES list.")


if __name__ == "__main__":
    main()
