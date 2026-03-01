#!/usr/bin/env python3
"""
Validate edges against wall boundaries.

Uses the _borders.jpg image (red = walls) to check if any edge
crosses through a wall. Reports bad edges and can auto-fix by
removing them from the JSON.

Usage:
    # Report bad edges:
    ~/isaacsim/python.sh floor_plans_resized/validate_edges.py floor_plans_resized/floor1/floor1.jpg

    # Auto-remove bad edges and save fixed JSON:
    ~/isaacsim/python.sh floor_plans_resized/validate_edges.py floor_plans_resized/floor1/floor1.jpg --fix
"""

import json
import os
import sys

import numpy as np
from PIL import Image


def load_wall_mask(borders_path, red_threshold=180):
    """
    Load the borders image and create a boolean wall mask.
    True = wall pixel, False = open space.
    """
    img = Image.open(borders_path).convert("RGB")
    pixels = np.array(img)

    # Red pixels: high R, low G, low B
    is_wall = (pixels[:, :, 0] > red_threshold) & \
              (pixels[:, :, 1] < 100) & \
              (pixels[:, :, 2] < 100)

    return is_wall


def sample_line(x0, y0, x1, y1, num_samples=50):
    """Sample points along a line between two coordinates."""
    ts = np.linspace(0, 1, num_samples)
    xs = x0 + (x1 - x0) * ts
    ys = y0 + (y1 - y0) * ts
    return xs, ys


def check_edge_crosses_wall(wall_mask, x0, y0, x1, y1, num_samples=60):
    """
    Check if a line between two points crosses any wall pixels.

    Returns (crosses_wall, wall_fraction, worst_point)
        crosses_wall: True if any sampled point hits a wall
        wall_fraction: fraction of samples that hit walls (0.0 to 1.0)
        worst_point: (x, y) of the first wall hit, or None
    """
    h, w = wall_mask.shape
    xs, ys = sample_line(x0, y0, x1, y1, num_samples)

    wall_hits = 0
    first_hit = None

    for x, y in zip(xs, ys):
        ix, iy = int(round(x)), int(round(y))
        # Bounds check
        if 0 <= ix < w and 0 <= iy < h:
            if wall_mask[iy, ix]:
                wall_hits += 1
                if first_hit is None:
                    first_hit = (ix, iy)

    fraction = wall_hits / num_samples
    return wall_hits > 0, fraction, first_hit


def main():
    if len(sys.argv) < 2:
        print("Usage: ~/isaacsim/python.sh validate_edges.py <floor_plan.jpg> [--fix]")
        sys.exit(1)

    image_path = sys.argv[1]
    do_fix = "--fix" in sys.argv

    base = os.path.splitext(image_path)[0]
    json_path = base + "_nodes.json"
    borders_path = base + "_borders.jpg"

    if not os.path.exists(borders_path):
        print(f"Borders image not found: {borders_path}")
        print("Run draw_borders.py first.")
        sys.exit(1)

    if not os.path.exists(json_path):
        print(f"Nodes JSON not found: {json_path}")
        sys.exit(1)

    # Load wall mask
    print(f"Loading wall mask from: {borders_path}")
    wall_mask = load_wall_mask(borders_path)
    print(f"Wall mask: {wall_mask.shape[1]}x{wall_mask.shape[0]}, "
          f"{np.sum(wall_mask)} wall pixels ({100*np.mean(wall_mask):.1f}%)")

    # Load nodes
    with open(json_path) as f:
        data = json.load(f)

    node_positions = {}
    for node in data.get("nodes", []):
        node_positions[node["id"]] = (node["x"], node["y"])

    # Check each edge
    bad_edges = []
    good_edges = []

    print(f"\nChecking {len(data.get('edges', []))} edges...")
    print("-" * 70)

    for edge in data.get("edges", []):
        start_id, end_id = edge[0], edge[1]
        if start_id not in node_positions or end_id not in node_positions:
            print(f"  SKIP: {start_id} -> {end_id} (node not found)")
            continue

        x0, y0 = node_positions[start_id]
        x1, y1 = node_positions[end_id]

        crosses, fraction, hit_point = check_edge_crosses_wall(
            wall_mask, x0, y0, x1, y1
        )

        if crosses:
            bad_edges.append((edge, fraction, hit_point))
            severity = "HEAVY" if fraction > 0.15 else "LIGHT"
            print(f"  BAD [{severity} {fraction:.0%}]: {start_id} -> {end_id}")
        else:
            good_edges.append(edge)

    print("-" * 70)
    print(f"\nResults: {len(good_edges)} good, {len(bad_edges)} bad edges")

    if bad_edges:
        print(f"\nBad edges ({len(bad_edges)}):")
        for edge, fraction, hit_point in sorted(bad_edges, key=lambda x: -x[1]):
            print(f"  {fraction:5.0%} wall: {edge[0]} -> {edge[1]}")

    # Also check nodes that are inside walls
    print(f"\nChecking {len(node_positions)} nodes for wall placement...")
    h, w = wall_mask.shape
    bad_nodes = []
    for node_id, (x, y) in node_positions.items():
        ix, iy = int(round(x)), int(round(y))
        if 0 <= ix < w and 0 <= iy < h:
            # Check a small area around the node (3x3)
            region = wall_mask[max(0,iy-1):iy+2, max(0,ix-1):ix+2]
            if np.mean(region) > 0.5:
                bad_nodes.append(node_id)
                print(f"  Node IN WALL: {node_id} at ({x}, {y})")

    if not bad_nodes:
        print("  All nodes are in open space.")

    if do_fix and bad_edges:
        # Remove bad edges from JSON
        fixed_edges = [e for e in data["edges"]
                       if not any(e[0] == be[0][0] and e[1] == be[0][1] for be in bad_edges)]
        data["edges"] = fixed_edges

        fixed_path = base + "_nodes.json"
        backup_path = base + "_nodes_backup.json"

        # Save backup
        with open(json_path) as f:
            backup_data = f.read()
        with open(backup_path, "w") as f:
            f.write(backup_data)
        print(f"\nBackup saved to: {backup_path}")

        # Save fixed JSON
        with open(fixed_path, "w") as f:
            json.dump(data, f, indent=2)
        print(f"Fixed JSON saved to: {fixed_path}")
        print(f"Removed {len(bad_edges)} bad edges, {len(fixed_edges)} edges remaining.")
    elif do_fix:
        print("\nNo bad edges to fix!")


if __name__ == "__main__":
    main()
