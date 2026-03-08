#!/usr/bin/env python3
"""
Node Detection Tool

Compares a base occupancy image with an annotated version where nodes
have been manually drawn. Detects the drawn marks, extracts their
pixel positions, converts to world coordinates, and outputs proper nodes.

Usage:
    1. Copy floor_1_occupancy.png and draw dots/marks on it
    2. Save the annotated version as floor_1_annotated.png in this folder
    3. Run: python3 test_building/test_nodes/detect_nodes.py

The script diffs the two images to find where you drew, clusters the
changed pixels into individual nodes, and outputs coordinates.
"""

import os
import sys
import json
import numpy as np
from PIL import Image, ImageDraw

# ---------------------------------------------------------------------------
# CONFIG
# ---------------------------------------------------------------------------
FLOOR = "floor_1"
# Minimum pixel difference to count as "drawn on" (0-255 scale)
DIFF_THRESHOLD = 30
# Minimum cluster size (pixels) to count as a node (filters noise)
MIN_CLUSTER_SIZE = 5
# How close pixels need to be to belong to the same cluster
CLUSTER_DISTANCE = 20

# Output node appearance
NODE_RADIUS = 15
NODE_COLOR = (255, 0, 0)  # red
LABEL_COLOR = (255, 255, 0)  # yellow

# ---------------------------------------------------------------------------
# Paths
# ---------------------------------------------------------------------------
TEST_NODES_DIR = os.path.dirname(os.path.abspath(__file__))


def pixel_to_world(col, row, xy_min, cell_size):
    """Convert grid pixel coordinates to world coordinates (cm)."""
    wx = xy_min[0] + col * cell_size
    wy = xy_min[1] + row * cell_size
    return wx, wy


def cluster_points(points, max_dist):
    """Simple greedy clustering of 2D points."""
    if len(points) == 0:
        return []

    points = np.array(points)
    used = np.zeros(len(points), dtype=bool)
    clusters = []

    for i in range(len(points)):
        if used[i]:
            continue
        # Start new cluster
        cluster = [points[i]]
        used[i] = True
        # Find all nearby points
        queue = [points[i]]
        while queue:
            current = queue.pop(0)
            dists = np.sqrt(np.sum((points - current) ** 2, axis=1))
            neighbors = np.where((dists < max_dist) & (~used))[0]
            for n in neighbors:
                used[n] = True
                cluster.append(points[n])
                queue.append(points[n])
        clusters.append(np.array(cluster))

    return clusters


def main():
    # Load metadata
    meta_path = os.path.join(TEST_NODES_DIR, f"{FLOOR}_meta.json")
    with open(meta_path) as f:
        meta = json.load(f)

    xy_min = meta["xy_min"]
    cell_size = meta["cell_size_cm"]

    # Load both images from the images subfolder
    images_dir = os.path.join(TEST_NODES_DIR, "images")
    base_path = os.path.join(images_dir, f"{FLOOR}_occupancy.png")
    annotated_path = os.path.join(images_dir, f"{FLOOR}_occupancy_annoated.png")

    if not os.path.exists(base_path):
        print(f"ERROR: Base image not found: {base_path}")
        sys.exit(1)
    if not os.path.exists(annotated_path):
        print(f"ERROR: Annotated image not found: {annotated_path}")
        print(f"Draw nodes on a copy of {FLOOR}_occupancy.png and save as {FLOOR}_annotated.png")
        sys.exit(1)

    base = np.array(Image.open(base_path).convert("RGB"), dtype=np.float32)
    annotated = np.array(Image.open(annotated_path).convert("RGB"), dtype=np.float32)

    if base.shape != annotated.shape:
        print(f"ERROR: Image sizes don't match!")
        print(f"  Base: {base.shape}")
        print(f"  Annotated: {annotated.shape}")
        sys.exit(1)

    print(f"Image size: {base.shape[1]} x {base.shape[0]}")

    # Compute per-pixel difference
    diff = np.sqrt(np.sum((base - annotated) ** 2, axis=2))
    changed_mask = diff > DIFF_THRESHOLD

    n_changed = np.sum(changed_mask)
    print(f"Changed pixels: {n_changed}")

    if n_changed == 0:
        print("No differences found between images. Did you draw on the annotated image?")
        sys.exit(0)

    # Get coordinates of changed pixels
    changed_coords = np.argwhere(changed_mask)  # (row, col)
    changed_points = changed_coords[:, ::-1]  # convert to (col, row)

    # Cluster into individual nodes
    clusters = cluster_points(changed_points, CLUSTER_DISTANCE)
    # Filter out noise
    clusters = [c for c in clusters if len(c) >= MIN_CLUSTER_SIZE]

    print(f"Detected {len(clusters)} node(s):")

    # Find centroid of each cluster
    nodes = []
    for i, cluster in enumerate(clusters):
        centroid_col = int(np.mean(cluster[:, 0]))
        centroid_row = int(np.mean(cluster[:, 1]))
        wx, wy = pixel_to_world(centroid_col, centroid_row, xy_min, cell_size)
        label = f"node_{i+1}"
        print(f"  {label}: pixel ({centroid_col}, {centroid_row}) -> "
              f"world ({wx:.1f}, {wy:.1f}) cm  [{len(cluster)} pixels]")
        nodes.append({
            "label": label,
            "grid_col": centroid_col,
            "grid_row": centroid_row,
            "world_x": wx,
            "world_y": wy,
            "cluster_size": len(cluster),
        })

    # Draw detected nodes on base image
    img = Image.open(base_path).convert("RGB")
    draw = ImageDraw.Draw(img)

    for node in nodes:
        col, row = node["grid_col"], node["grid_row"]
        r = NODE_RADIUS
        draw.ellipse([col - r, row - r, col + r, row + r], fill=NODE_COLOR)
        draw.text((col + r + 3, row - r), node["label"], fill=LABEL_COLOR)

    out_img = os.path.join(TEST_NODES_DIR, f"{FLOOR}_detected_nodes.png")
    img.save(out_img)
    print(f"\nSaved image: {out_img}")

    # Save node data
    out_json = os.path.join(TEST_NODES_DIR, f"{FLOOR}_detected_nodes.json")
    with open(out_json, "w") as f:
        json.dump({"floor": FLOOR, "cell_size_cm": cell_size,
                    "xy_min": xy_min, "nodes": nodes}, f, indent=2)
    print(f"Saved data: {out_json}")

    # Print in place_nodes.py format for easy copy-paste
    print(f"\n# Copy-paste into place_nodes.py MANUAL_NODES list:")
    print("MANUAL_NODES = [")
    for node in nodes:
        print(f'    ({node["world_x"]:.1f}, {node["world_y"]:.1f}, "{node["label"]}"),')
    print("]")


if __name__ == "__main__":
    main()
