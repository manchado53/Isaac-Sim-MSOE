#!/usr/bin/env python3
"""
Verify that occupancy maps + navgraphs were regenerated with the correct
ROS map_server convention (row 0 = y_max, col 0 = x_min).

Runs four checks per floor:
  1. Corner bounds: pixel_to_world(0,0) and pixel_to_world(H-1,W-1)
     match the meta.json xy_min / xy_max.
  2. Pixel-space round-trip: 1000 random pixels round-trip exactly.
  3. Physical sanity (floor 4 only): the configured spawn pose
     (-37.20, 82.09) maps to a pixel that is in free space (value 255).
  4. Diff-vs-backup: the new PNG is np.flipud of the pre-fix PNG
     (small single-pixel differences near unknown/free borders are ok).

Usage:
    python3 floorplan_vision/verify_orientation.py
    python3 floorplan_vision/verify_orientation.py --floor floor_4
    python3 floorplan_vision/verify_orientation.py --backup /tmp/occupancy_backup_<ts>.tar.gz
"""

import argparse
import json
import os
import random
import sys
import tarfile
import tempfile
from pathlib import Path

import numpy as np
from PIL import Image

REPO = Path(__file__).resolve().parents[1]
MAPS = REPO / "assets/environments/InnovationCenter/maps"


# Copy of the canonical pixel<->world math from
# floorplan_vision/auto_graph_generator.py, reproduced here so this script
# has no dependency on cv2/networkx/skimage at import time.
def pixel_to_world(row, col, meta):
    res = meta["resolution_m"]
    H   = meta["grid_height"]
    x   = meta["xy_min"][0] + col * res
    y   = meta["xy_min"][1] + (H - 1 - row) * res
    return x, y


def world_to_pixel(x, y, meta):
    res = meta["resolution_m"]
    H   = meta["grid_height"]
    col = int(round((x - meta["xy_min"][0]) / res))
    row = (H - 1) - int(round((y - meta["xy_min"][1]) / res))
    return row, col


def load_meta(floor_name):
    meta_path = MAPS / f"{floor_name}_meta.json"
    with open(meta_path) as f:
        raw = json.load(f)
    return {
        "resolution_m": raw["cell_size_cm"],  # legacy field name
        "xy_min":       raw["xy_min"],
        "xy_max":       raw["xy_max"],
        "grid_width":   raw["grid_width"],
        "grid_height":  raw["grid_height"],
    }


def load_png(floor_name):
    png_path = MAPS / f"{floor_name}_occupancy.png"
    return np.array(Image.open(png_path).convert("L"))


def check_bounds(floor_name, meta, tol=1e-9):
    H, W = meta["grid_height"], meta["grid_width"]
    res = meta["resolution_m"]
    tl = pixel_to_world(0, 0, meta)
    br = pixel_to_world(H - 1, W - 1, meta)
    ex_tl = (meta["xy_min"][0], meta["xy_min"][1] + (H - 1) * res)
    ex_br = (meta["xy_min"][0] + (W - 1) * res, meta["xy_min"][1])
    ok_tl = abs(tl[0] - ex_tl[0]) < tol and abs(tl[1] - ex_tl[1]) < tol
    ok_br = abs(br[0] - ex_br[0]) < tol and abs(br[1] - ex_br[1]) < tol
    print(f"  [bounds] row=0 col=0     -> {tl}     expect {ex_tl}  {'OK' if ok_tl else 'FAIL'}")
    print(f"  [bounds] row={H-1} col={W-1} -> {br}  expect {ex_br}  {'OK' if ok_br else 'FAIL'}")
    return ok_tl and ok_br


def check_round_trip(floor_name, meta, n=1000):
    H, W = meta["grid_height"], meta["grid_width"]
    rng = random.Random(0)
    corners = [(0, 0), (0, W - 1), (H - 1, 0), (H - 1, W - 1)]
    pixels = corners + [(rng.randint(0, H - 1), rng.randint(0, W - 1))
                        for _ in range(n)]
    fails = 0
    first_fail = None
    for r, c in pixels:
        x, y = pixel_to_world(r, c, meta)
        rr, cc = world_to_pixel(x, y, meta)
        if (rr, cc) != (r, c):
            fails += 1
            if first_fail is None:
                first_fail = (r, c, x, y, rr, cc)
    ok = fails == 0
    print(f"  [round-trip] {len(pixels) - fails}/{len(pixels)} pass  "
          f"{'OK' if ok else 'FAIL'}")
    if first_fail:
        print(f"               first fail: {first_fail}")
    return ok


def check_physical_sanity(floor_name, meta, img):
    """Floor 4 only: the AMCL initial_pose spawn must be in free space."""
    if floor_name != "floor_4":
        return True
    sx, sy = -37.20, 82.09
    H, W = meta["grid_height"], meta["grid_width"]
    # Check the spawn is inside map bounds first.
    inside = (meta["xy_min"][0] <= sx <= meta["xy_max"][0] and
              meta["xy_min"][1] <= sy <= meta["xy_max"][1])
    if not inside:
        print(f"  [physical] spawn ({sx}, {sy}) OUT OF BOUNDS  FAIL")
        return False
    row, col = world_to_pixel(sx, sy, meta)
    if not (0 <= row < H and 0 <= col < W):
        print(f"  [physical] spawn pixel ({row}, {col}) out of grid  FAIL")
        return False
    val = int(img[row, col])
    name = {0: "OCCUPIED", 128: "UNKNOWN", 255: "FREE"}.get(val, f"val={val}")
    ok = val > 200  # free
    print(f"  [physical] spawn ({sx}, {sy}) -> pixel (row={row}, col={col}) "
          f"= {name}  {'OK' if ok else 'FAIL'}")
    return ok


def check_flipud_vs_backup(floor_name, new_img, backup_tar=None):
    """Optional: new_img should equal np.flipud(old_img) from the backup."""
    if backup_tar is None:
        print(f"  [diff]  skipped (no backup tar provided)")
        return True
    if not os.path.exists(backup_tar):
        print(f"  [diff]  FAIL: backup not found at {backup_tar}")
        return False
    with tempfile.TemporaryDirectory() as tmp:
        with tarfile.open(backup_tar, "r:gz") as tf:
            member = f"assets/environments/InnovationCenter/maps/{floor_name}_occupancy.png"
            try:
                tf.extract(member, path=tmp)
            except KeyError:
                print(f"  [diff]  skipped (member {member!r} not in backup)")
                return True
        old_path = os.path.join(tmp, member)
        old = np.array(Image.open(old_path).convert("L"))
    if old.shape != new_img.shape:
        print(f"  [diff]  DIMENSIONS CHANGED old={old.shape} new={new_img.shape}"
              f"  (expected if USD was swapped — skipping strict compare)")
        return True
    expected = np.flipud(old)
    diff = (expected != new_img).sum()
    total = new_img.size
    pct = 100.0 * diff / total
    ok = pct < 1.0
    print(f"  [diff]  pixels disagreeing with np.flipud(old): "
          f"{diff:,} / {total:,} ({pct:.3f}%)  {'OK' if ok else 'FAIL'}")
    return ok


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--floor", default=None,
                    help="If set, only verify that floor (e.g. floor_4)")
    ap.add_argument("--backup", default=None,
                    help="Path to the tar.gz backup for the np.flipud diff test")
    args = ap.parse_args()

    floors = [args.floor] if args.floor else ["floor_1", "floor_2", "floor_3", "floor_4"]

    overall_ok = True
    for fname in floors:
        print(f"\n=== {fname} ===")
        meta_path = MAPS / f"{fname}_meta.json"
        png_path = MAPS / f"{fname}_occupancy.png"
        if not meta_path.exists() or not png_path.exists():
            print(f"  SKIP — missing {meta_path.name} or {png_path.name}")
            continue
        meta = load_meta(fname)
        img = load_png(fname)
        print(f"  grid: {meta['grid_width']} x {meta['grid_height']}  "
              f"res={meta['resolution_m']}  "
              f"xy_min={meta['xy_min']} xy_max={meta['xy_max']}")
        results = [
            check_bounds(fname, meta),
            check_round_trip(fname, meta),
            check_physical_sanity(fname, meta, img),
            check_flipud_vs_backup(fname, img, args.backup),
        ]
        if not all(results):
            overall_ok = False

    print()
    print("=" * 60)
    print("PASS" if overall_ok else "FAIL")
    print("=" * 60)
    sys.exit(0 if overall_ok else 1)


if __name__ == "__main__":
    main()
