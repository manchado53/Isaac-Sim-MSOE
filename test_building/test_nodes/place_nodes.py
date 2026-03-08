#!/usr/bin/env python3
"""
Manual Node Placement Tool

Place nodes on the floor occupancy map by specifying world coordinates (in cm).
Edit the MANUAL_NODES list below with your (x, y) coordinates, then run:

    python3 test_building/test_nodes/place_nodes.py

Coordinates use the same world coordinate system as the USD file.
The occupancy map origin is at xy_min from the metadata.
"""

import os
import sys
import json
import numpy as np
from PIL import Image, ImageDraw

# ---------------------------------------------------------------------------
# CONFIG — Edit these
# ---------------------------------------------------------------------------
FLOOR = "floor_1"  # which floor to place nodes on

# Manual node coordinates in WORLD space (cm) — (x, y)
# These are the same coordinates as in the USD / Isaac Sim scene.
# Example: the building X range is roughly -8540 to 4200, Y range -950 to 13097
MANUAL_NODES = [
    # (x_cm, y_cm, label)
    # Example nodes — replace with your own:
    (-2000.0, 5000.0, "node_1"),
    (-1000.0, 6000.0, "node_2"),
    (0.0,     7000.0, "node_3"),
]

# Node appearance
NODE_RADIUS = 12  # pixels
NODE_COLOR = (255, 0, 0)  # red
LABEL_COLOR = (255, 255, 0)  # yellow

# ---------------------------------------------------------------------------
# Paths
# ---------------------------------------------------------------------------
TEST_NODES_DIR = os.path.dirname(os.path.abspath(__file__))


def world_to_grid(wx, wy, xy_min, cell_size):
    """Convert world coordinates (cm) to grid pixel coordinates."""
    col = (wx - xy_min[0]) / cell_size
    row = (wy - xy_min[1]) / cell_size
    return int(round(col)), int(round(row))


def main():
    # Load metadata
    meta_path = os.path.join(TEST_NODES_DIR, f"{FLOOR}_meta.json")
    if not os.path.exists(meta_path):
        print(f"ERROR: {meta_path} not found. Run occupancy map generation first.")
        sys.exit(1)

    with open(meta_path) as f:
        meta = json.load(f)

    xy_min = meta["xy_min"]
    xy_max = meta["xy_max"]
    cell_size = meta["cell_size_cm"]
    grid_w = meta["grid_width"]
    grid_h = meta["grid_height"]

    print(f"Floor: {FLOOR}")
    print(f"Grid: {grid_w} x {grid_h}, cell_size={cell_size} cm")
    print(f"World X: [{xy_min[0]:.1f}, {xy_max[0]:.1f}] cm")
    print(f"World Y: [{xy_min[1]:.1f}, {xy_max[1]:.1f}] cm")
    print()

    # Load the occupancy map image as base
    occ_img_path = os.path.join(TEST_NODES_DIR, f"{FLOOR}_occupancy.png")
    if not os.path.exists(occ_img_path):
        print(f"ERROR: {occ_img_path} not found.")
        sys.exit(1)

    img = Image.open(occ_img_path).convert("RGB")
    draw = ImageDraw.Draw(img)

    print(f"Placing {len(MANUAL_NODES)} nodes:")
    placed = []
    for node in MANUAL_NODES:
        wx, wy = node[0], node[1]
        label = node[2] if len(node) > 2 else ""

        col, row = world_to_grid(wx, wy, xy_min, cell_size)

        # Check bounds
        if col < 0 or col >= grid_w or row < 0 or row >= grid_h:
            print(f"  SKIP {label}: ({wx}, {wy}) -> pixel ({col}, {row}) — out of bounds")
            continue

        print(f"  {label}: world ({wx:.1f}, {wy:.1f}) -> pixel ({col}, {row})")

        # Draw node
        r = NODE_RADIUS
        draw.ellipse([col - r, row - r, col + r, row + r], fill=NODE_COLOR)

        # Draw label
        if label:
            draw.text((col + r + 3, row - r), label, fill=LABEL_COLOR)

        placed.append({"label": label, "world_x": wx, "world_y": wy,
                        "grid_col": col, "grid_row": row})

    # Save output
    out_img = os.path.join(TEST_NODES_DIR, f"{FLOOR}_manual_nodes.png")
    img.save(out_img)
    print(f"\nSaved: {out_img}")

    out_json = os.path.join(TEST_NODES_DIR, f"{FLOOR}_manual_nodes.json")
    with open(out_json, "w") as f:
        json.dump({"floor": FLOOR, "cell_size_cm": cell_size,
                    "xy_min": xy_min, "xy_max": xy_max,
                    "nodes": placed}, f, indent=2)
    print(f"Saved: {out_json}")


if __name__ == "__main__":
    main()
