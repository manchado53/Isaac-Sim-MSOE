#!/usr/bin/env python3
"""
Crop V4_floor_plans occupancy maps to the building footprint.

Finds the bounding box of OCCUPIED (wall) cells, adds a small
padding border, then crops the .npy grid, regenerates the .png,
and updates grid_width / grid_height / xy_min / xy_max in the
corresponding _meta.json.

No Isaac Sim needed — pure numpy + PIL.
Run with:
  sbatch submit_isaac_job.sh test_building/crop_floor_plans.py
"""

import os
import json
import numpy as np

FREE     = 0
OCCUPIED = 1
UNKNOWN  = 2
PADDING  = 15   # cells of border to keep around the outermost walls

PROJECT_DIR = os.path.dirname(os.path.abspath(__file__))
OUTPUT_DIR  = os.path.join(PROJECT_DIR, "V4_floor_plans")

FLOORS = ["floor_1", "floor_2", "floor_3", "floor_4"]

try:
    from PIL import Image
    def save_png(img_array, path):
        Image.fromarray(img_array).save(path)
except ImportError:
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    def save_png(img_array, path):
        plt.imsave(path, img_array)


def crop_floor(floor_name):
    npy_path  = os.path.join(OUTPUT_DIR, f"{floor_name}_occupancy.npy")
    png_path  = os.path.join(OUTPUT_DIR, f"{floor_name}_occupancy.png")
    meta_path = os.path.join(OUTPUT_DIR, f"{floor_name}_meta.json")

    if not os.path.exists(npy_path):
        print(f"  SKIP {floor_name}: {npy_path} not found")
        return

    grid = np.load(npy_path)
    h, w = grid.shape
    print(f"\n{floor_name}:  original grid {w}×{h}")

    # Find bounding box of occupied (wall) cells
    occ_rows, occ_cols = np.where(grid == OCCUPIED)
    if len(occ_rows) == 0:
        print(f"  SKIP: no occupied cells found")
        return

    row_min = max(0,  occ_rows.min() - PADDING)
    row_max = min(h,  occ_rows.max() + PADDING + 1)
    col_min = max(0,  occ_cols.min() - PADDING)
    col_max = min(w,  occ_cols.max() + PADDING + 1)

    cropped = grid[row_min:row_max, col_min:col_max]
    new_h, new_w = cropped.shape
    print(f"  Crop rows [{row_min}:{row_max}], cols [{col_min}:{col_max}]")
    print(f"  Cropped grid {new_w}×{new_h}  "
          f"(removed {w - new_w} cols, {h - new_h} rows)")

    # Save cropped npy
    np.save(npy_path, cropped)

    # Regenerate PNG
    img = np.full((new_h, new_w, 3), 128, dtype=np.uint8)   # gray = unknown
    img[cropped == FREE]     = [255, 255, 255]
    img[cropped == OCCUPIED] = [0,   0,   0  ]
    save_png(img, png_path)
    print(f"  Saved: {png_path}")

    # Update meta.json
    with open(meta_path) as f:
        meta = json.load(f)

    cell_size = meta["cell_size_cm"]   # metres (field name is legacy)
    old_xy_min = meta["xy_min"]        # [x, y] world coords of grid origin

    new_xy_min_x = old_xy_min[0] + col_min * cell_size
    new_xy_min_y = old_xy_min[1] + row_min * cell_size
    new_xy_max_x = new_xy_min_x + new_w * cell_size
    new_xy_max_y = new_xy_min_y + new_h * cell_size

    meta["grid_width"]  = new_w
    meta["grid_height"] = new_h
    meta["xy_min"]      = [new_xy_min_x, new_xy_min_y]
    meta["xy_max"]      = [new_xy_max_x, new_xy_max_y]
    meta["crop_row_offset"] = int(row_min)
    meta["crop_col_offset"] = int(col_min)

    with open(meta_path, "w") as f:
        json.dump(meta, f, indent=2)
    print(f"  Saved: {meta_path}")
    print(f"  xy_min: {old_xy_min}  →  [{new_xy_min_x:.4f}, {new_xy_min_y:.4f}]")


if __name__ == "__main__":
    print("=" * 50)
    print("Cropping V4 floor plan occupancy maps")
    print("=" * 50)
    for floor in FLOORS:
        crop_floor(floor)
    print("\nDone.")
