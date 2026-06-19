#!/usr/bin/env python3
"""
Glass-variation review grid — frame picker for total_glass_testV*.mp4.

Analogous to visual_review_grid.py, but no Isaac Sim rendering: the glass
dataset (data_glass/tanner_vlm_dataset/) is already rendered, so we just
symlink one representative frame per (panel, lighting, state) cell into a
staging dir, then stitch with frames_to_video.py.

Grid per panel
--------------
* 5 lightings × 4 states = 20 cells
    lightings: bright_fluorescent, daylight_flood, warm_evening,
               overcast_flat, high_contrast
    states:    clear (intact) + edge_v1, v4, tempered_v5 (cracked)

Fixed axes
----------
    arc=+00, d=020, uv=1, h=+00  (neutral viewpoint, matches the door grid's
    single head-on pose). Falls back to the nearest available combination
    if the neutral one is missing.

Panels
------
Ordered train (10) → val (1) → test (2) = 13 panels, 260 frames total.

Output
------
    data/glass_review_grid/
        {panel_ord:02d}_L{0-4}_M{0-3}_{original_name}   (symlinks)

Stitch video
------------
    python simulations/robot/frames_to_video.py \
        data/glass_review_grid --fps 4 -o total_glass_testV1.mp4
"""

import os
import shutil
from pathlib import Path

_PROJECT_ROOT = Path(__file__).resolve().parents[2]
DATA_ROOT  = _PROJECT_ROOT / "data_glass" / "tanner_vlm_dataset" / "frames"
OUT_DIR    = _PROJECT_ROOT / "data" / "glass_review_grid"

LIGHTINGS = [
    "bright_fluorescent",
    "daylight_flood",
    "warm_evening",
    "overcast_flat",
    "high_contrast",
]

# (label_prefix, material_axis) — label matches the filename prefix (cracked_/intact_)
STATES = [
    ("intact",  "clear"),
    ("cracked", "edge_v1"),
    ("cracked", "v4"),
    ("cracked", "tempered_v5"),
]

# (panel_number, split_dir). Ordering chosen so the video progresses
# train → val → test (matching the door video's fixed registry order).
PANELS = [
    (2,  "train"), (3,  "train"), (4,  "train"), (6,  "train"),
    (13, "train"), (15, "train"), (31, "train"), (36, "train"),
    (40, "train"), (42, "train"),
    (22, "val"),
    (18, "test"), (28, "test"),
]

# Neutral axis values (match the most common combination in the sweep).
ARC_PREF = "+00"
D_PREF   = "020"
UV_PREF  = "1"
H_PREF   = "+00"


def _find_frame(label: str, pan: int, split: str,
                lighting: str, material: str) -> Path | None:
    """Strict match on neutral axes, else progressively relax."""
    split_dir = DATA_ROOT / split
    head = f"{label}_pan{pan:02d}_L-{lighting}_M-{material}"

    candidates = [
        f"{head}_arc{ARC_PREF}_d{D_PREF}_uv{UV_PREF}_h{H_PREF}_f*.jpg",
        f"{head}_arc{ARC_PREF}_d{D_PREF}_uv{UV_PREF}_h*_f*.jpg",
        f"{head}_arc{ARC_PREF}_d{D_PREF}_uv*_h*_f*.jpg",
        f"{head}_arc{ARC_PREF}_d*_uv*_h*_f*.jpg",
        f"{head}_arc*_d*_uv*_h*_f*.jpg",
    ]
    for pat in candidates:
        matches = sorted(split_dir.glob(pat))
        if matches:
            return matches[0]
    return None


def main():
    if OUT_DIR.exists():
        shutil.rmtree(OUT_DIR)
    OUT_DIR.mkdir(parents=True)

    picked = 0
    missed = []
    for pi, (pan, split) in enumerate(PANELS):
        for li, light in enumerate(LIGHTINGS):
            for mi, (label, mat) in enumerate(STATES):
                src = _find_frame(label, pan, split, light, mat)
                if src is None:
                    missed.append(f"pan{pan:02d}({split}) L={light} M={mat}")
                    continue
                dst = OUT_DIR / f"{pi:02d}_L{li}_M{mi}_{src.name}"
                dst.symlink_to(os.path.relpath(src.resolve(), dst.parent))
                picked += 1

    print(f"Picked {picked} frames, {len(missed)} missing")
    for m in missed:
        print(f"  [MISS] {m}")
    print(f"\nStaging dir: {OUT_DIR}")
    print("\nStitch video:")
    print(f"  python simulations/robot/frames_to_video.py "
          f"{OUT_DIR.relative_to(_PROJECT_ROOT)} --fps 4 "
          f"-o total_glass_testV1.mp4")


if __name__ == "__main__":
    main()
