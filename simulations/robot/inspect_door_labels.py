#!/usr/bin/env python3
"""
inspect_door_labels.py — Visualize and summarize door_labels.npz.

Reads data/door_labels.npz (produced by nova_carter_door_collection.py) and writes:
  data/sample_frames.png     — 2x4 grid of evenly-spaced frames with labels
  data/class_samples.png     — 4 CLOSED + 4 OPEN example frames side by side

Also prints dataset statistics to stdout.

Usage:
  python3 simulations/robot/inspect_door_labels.py
  python3 simulations/robot/inspect_door_labels.py --npz data/door_labels.npz --out data/
"""

import argparse
import os
import sys

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

CLOSED_THRESHOLD_DEG = 5.0   # matches nova_carter_door_collection.py / prepare_finetune_data.py


def label_to_class(label: float) -> str:
    return "CLOSED" if label * 90.0 <= CLOSED_THRESHOLD_DEG else "OPEN"


def make_grid(frames, titles, ncols=4, figsize_per=(3.2, 2.6)):
    nrows = (len(frames) + ncols - 1) // ncols
    fig, axes = plt.subplots(nrows, ncols,
                             figsize=(figsize_per[0] * ncols, figsize_per[1] * nrows))
    axes = np.array(axes).flatten()
    for ax, frame, title in zip(axes, frames, titles):
        ax.imshow(frame)
        ax.set_title(title, fontsize=8)
        ax.axis("off")
    for ax in axes[len(frames):]:
        ax.axis("off")
    fig.tight_layout(pad=0.5)
    return fig


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--npz", default="data/door_labels.npz")
    parser.add_argument("--out", default="data/")
    args = parser.parse_args()

    script_dir  = os.path.dirname(os.path.abspath(__file__))
    project_dir = os.path.abspath(os.path.join(script_dir, "..", ".."))
    npz_path    = os.path.join(project_dir, args.npz)
    out_dir     = os.path.join(project_dir, args.out)

    if not os.path.isfile(npz_path):
        print(f"ERROR: {npz_path} not found. Run nova_carter_door_collection.py first.")
        sys.exit(1)

    print(f"Loading {npz_path} ...")
    data      = np.load(npz_path, allow_pickle=False)
    frames    = data["frames"]     # (N, H, W, 3) uint8
    labels    = data["labels"]     # (N,) float32 [0, 1]
    door_ids  = data["door_ids"]   # (N,) int32
    frame_ids = data["frame_ids"]  # (N,) int32
    N = len(frames)

    # ------------------------------------------------------------------
    # Stats
    # ------------------------------------------------------------------
    angles = labels * 90.0
    closed_mask = angles <= CLOSED_THRESHOLD_DEG
    open_mask   = ~closed_mask
    n_closed = int(closed_mask.sum())
    n_open   = int(open_mask.sum())

    print(f"\n=== door_labels.npz summary ===")
    print(f"  Total frames  : {N}")
    print(f"  Image shape   : {frames.shape[1]}x{frames.shape[2]} px")
    print(f"  Label range   : {labels.min():.4f} – {labels.max():.4f}  "
          f"({angles.min():.1f}° – {angles.max():.1f}°)")
    print(f"  Label mean/std: {labels.mean():.4f} ± {labels.std():.4f}")
    print(f"\n  Class distribution (threshold {CLOSED_THRESHOLD_DEG}°):")
    print(f"    CLOSED : {n_closed:5d}  ({100*n_closed/N:.1f}%)")
    print(f"    OPEN   : {n_open:5d}  ({100*n_open/N:.1f}%)")
    print(f"\n  Unique doors captured : {len(np.unique(door_ids))}")
    for did in np.unique(door_ids):
        mask = door_ids == did
        print(f"    Door {did}: {mask.sum()} frames")

    os.makedirs(out_dir, exist_ok=True)

    # ------------------------------------------------------------------
    # sample_frames.png — 8 evenly-spaced frames across the full dataset
    # ------------------------------------------------------------------
    n_sample = min(8, N)
    sample_idx = np.linspace(0, N - 1, n_sample, dtype=int)
    sample_frames = [frames[i] for i in sample_idx]
    sample_titles = [f"frame {frame_ids[i]}  label={labels[i]:.2f}" for i in sample_idx]

    fig = make_grid(sample_frames, sample_titles, ncols=4)
    out_sample = os.path.join(out_dir, "sample_frames.png")
    fig.savefig(out_sample, dpi=120, bbox_inches="tight")
    plt.close(fig)
    print(f"\nSaved: {out_sample}")

    # ------------------------------------------------------------------
    # class_samples.png — 4 CLOSED + 4 OPEN examples
    # ------------------------------------------------------------------
    closed_idx = np.where(closed_mask)[0]
    open_idx   = np.where(open_mask)[0]

    n_each = 4
    def pick(idx_arr):
        if len(idx_arr) == 0:
            return []
        chosen = np.linspace(0, len(idx_arr) - 1, min(n_each, len(idx_arr)), dtype=int)
        return [idx_arr[c] for c in chosen]

    closed_picks = pick(closed_idx)
    open_picks   = pick(open_idx)
    all_picks    = closed_picks + open_picks

    if all_picks:
        cls_frames = [frames[i] for i in all_picks]
        cls_titles = [
            f"CLOSED  {labels[i]*90:.1f}°" if i in closed_picks
            else f"OPEN  {labels[i]*90:.1f}°"
            for i in all_picks
        ]
        fig2 = make_grid(cls_frames, cls_titles, ncols=4)
        out_class = os.path.join(out_dir, "class_samples.png")
        fig2.savefig(out_class, dpi=120, bbox_inches="tight")
        plt.close(fig2)
        print(f"Saved: {out_class}")

    # ------------------------------------------------------------------
    # label_histogram.png
    # ------------------------------------------------------------------
    fig3, ax = plt.subplots(figsize=(7, 3.5))
    ax.hist(angles, bins=36, range=(0, 90), color="steelblue", edgecolor="white", linewidth=0.4)
    ax.axvline(CLOSED_THRESHOLD_DEG, color="red", linestyle="--", linewidth=1.2,
               label=f"CLOSED threshold ({CLOSED_THRESHOLD_DEG}°)")
    ax.set_xlabel("Door angle (degrees)")
    ax.set_ylabel("Frame count")
    ax.set_title(f"Label distribution  (N={N})")
    ax.legend(fontsize=9)
    fig3.tight_layout()
    out_hist = os.path.join(out_dir, "label_histogram.png")
    fig3.savefig(out_hist, dpi=120, bbox_inches="tight")
    plt.close(fig3)
    print(f"Saved: {out_hist}")

    print("\nDone.")


if __name__ == "__main__":
    main()
