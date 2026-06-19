#!/usr/bin/env python3
"""
plot_real_doors_sample_grid.py — 4x4 sample grid from the 400-frame
real-door eval set (val_real_doors/{Closed,Open}/).

Picks 8 closed + 8 open images, evenly spaced through each sorted folder
(deterministic — no random seed needed). Top 2 rows = CLOSED, bottom 2 rows
= OPEN. Each cell carries a small color-coded label badge in the corner so
the audience can read the ground truth at a glance.

Output:
  data/plot_real_doors_sample_grid.png
"""

import os

import matplotlib.pyplot as plt
import numpy as np
from PIL import Image

PROJECT_DIR = os.path.abspath(
    os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", ".."))
SRC_DIR  = os.path.join(PROJECT_DIR, "val_real_doors")
OUT_PNG  = os.path.join(PROJECT_DIR, "data", "plot_real_doors_sample_grid.png")

N_PER_CLASS = 8
N_COLS      = 4
N_ROWS      = 4   # 2 rows closed + 2 rows open

CLOSED_COLOR = "#b91c1c"  # red — closed
OPEN_COLOR   = "#15803d"  # green — open


def evenly_spaced(items, n):
    if len(items) <= n:
        return items
    idx = np.linspace(0, len(items) - 1, n).round().astype(int)
    return [items[i] for i in idx]


def load_class(folder):
    d = os.path.join(SRC_DIR, folder)
    files = sorted(f for f in os.listdir(d)
                   if f.lower().endswith((".png", ".jpg", ".jpeg")))
    picks = evenly_spaced(files, N_PER_CLASS)
    return [(os.path.join(d, fn), fn) for fn in picks]


def main():
    closed = load_class("Closed")
    opened = load_class("Open")

    # Layout: rows 0-1 = closed (8 images), rows 2-3 = open (8 images)
    cells = [(p, fn, "CLOSED", CLOSED_COLOR) for p, fn in closed] + \
            [(p, fn, "OPEN",   OPEN_COLOR)   for p, fn in opened]

    fig, axes = plt.subplots(N_ROWS, N_COLS, figsize=(13, 10),
                             facecolor="#f4f4f5")
    fig.suptitle("Real-door eval samples — 16 of 400 frames",
                 fontsize=15, fontweight="bold", y=0.995)

    for ax, (path, fn, label, color) in zip(axes.flat, cells):
        img = Image.open(path).convert("RGB")
        ax.imshow(np.asarray(img))
        ax.set_xticks([]); ax.set_yticks([])
        for spine in ax.spines.values():
            spine.set_edgecolor(color)
            spine.set_linewidth(2.4)

        ax.text(0.03, 0.96, label, transform=ax.transAxes,
                ha="left", va="top",
                fontsize=10, fontweight="bold", color="white",
                bbox=dict(facecolor=color, edgecolor="none",
                          boxstyle="round,pad=0.25", alpha=0.92))

    fig.tight_layout(rect=[0, 0, 1, 0.97])
    fig.savefig(OUT_PNG, dpi=160, bbox_inches="tight",
                facecolor=fig.get_facecolor())
    print(f"Wrote {OUT_PNG}")
    print(f"  Closed picks: {[fn for _, fn in closed]}")
    print(f"  Open picks:   {[fn for _, fn in opened]}")


if __name__ == "__main__":
    main()
