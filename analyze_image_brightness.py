#!/usr/bin/env python3
"""Compute mean RGB for each image in val_images_labeled and cluster by brightness."""
import os
import sys
from pathlib import Path
import numpy as np
from PIL import Image
from concurrent.futures import ProcessPoolExecutor
from sklearn.cluster import KMeans

IMG_DIR = Path("/home/ad.msoe.edu/celliot/Digital-twin-v2/isaac-sim-msoe/data/val_images_labeled")
OUT_CSV = Path("/home/ad.msoe.edu/celliot/Digital-twin-v2/isaac-sim-msoe/data/val_image_rgb_stats.csv")
OUT_PLOT = Path("/home/ad.msoe.edu/celliot/Digital-twin-v2/isaac-sim-msoe/data/val_image_brightness_hist.png")


def mean_rgb(path: Path):
    try:
        with Image.open(path) as im:
            im = im.convert("RGB")
            arr = np.asarray(im, dtype=np.float32)
            r, g, b = arr[..., 0].mean(), arr[..., 1].mean(), arr[..., 2].mean()
            return path.name, float(r), float(g), float(b)
    except Exception as e:
        return path.name, -1.0, -1.0, -1.0


def main():
    files = sorted(IMG_DIR.glob("*.jpg"))
    print(f"Found {len(files)} images")

    with ProcessPoolExecutor(max_workers=8) as ex:
        results = list(ex.map(mean_rgb, files, chunksize=64))

    names = [r[0] for r in results]
    rgb = np.array([[r[1], r[2], r[3]] for r in results], dtype=np.float32)
    bright = rgb.mean(axis=1)

    # Save CSV
    with open(OUT_CSV, "w") as f:
        f.write("filename,mean_r,mean_g,mean_b,mean_brightness\n")
        for name, row, b in zip(names, rgb, bright):
            f.write(f"{name},{row[0]:.3f},{row[1]:.3f},{row[2]:.3f},{b:.3f}\n")
    print(f"Wrote {OUT_CSV}")

    # Simple threshold counts
    print("\n=== Brightness threshold summary (mean of RGB, 0-255) ===")
    for thr in [5, 10, 20, 30, 50]:
        n = int((bright < thr).sum())
        print(f"  brightness < {thr:3d}: {n:5d} ({100*n/len(bright):.2f}%)")

    # KMeans cluster on brightness (1D)
    km = KMeans(n_clusters=3, random_state=0, n_init=10).fit(bright.reshape(-1, 1))
    labels = km.labels_
    centers = km.cluster_centers_.flatten()
    order = np.argsort(centers)  # dark -> light
    print("\n=== KMeans (k=3) on brightness ===")
    for rank, ci in enumerate(order):
        mask = labels == ci
        n = int(mask.sum())
        bmin = bright[mask].min() if n else 0
        bmax = bright[mask].max() if n else 0
        print(f"  cluster {rank} (center={centers[ci]:6.2f}): n={n:5d}  range=[{bmin:.2f}, {bmax:.2f}]")

    # Plot histogram
    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
        fig, ax = plt.subplots(figsize=(10, 5))
        ax.hist(bright, bins=80, color="steelblue", edgecolor="black")
        for c in centers:
            ax.axvline(c, color="red", linestyle="--", alpha=0.6)
        ax.set_xlabel("Mean brightness (avg of R,G,B), 0-255")
        ax.set_ylabel("Number of images")
        ax.set_title(f"val_images_labeled brightness distribution (n={len(bright)})")
        fig.tight_layout()
        fig.savefig(OUT_PLOT, dpi=120)
        print(f"Wrote {OUT_PLOT}")
    except Exception as e:
        print(f"Plot skipped: {e}")

    # Show 10 darkest images
    idx = np.argsort(bright)[:15]
    print("\n=== 15 darkest images ===")
    for i in idx:
        print(f"  brightness={bright[i]:6.2f}  rgb=({rgb[i,0]:5.1f},{rgb[i,1]:5.1f},{rgb[i,2]:5.1f})  {names[i]}")


if __name__ == "__main__":
    main()
