#!/usr/bin/env python3
"""
Fix nodes and edges in floor plan JSON.

1. Moves nodes that are inside walls to nearest open space.
2. Tests all edges against wall mask.
3. For short-distance edges that barely clip walls, adjusts node positions.
4. Reports results so we can manually fix remaining issues.

Usage:
    ~/isaacsim/python.sh floor_plans_resized/fix_nodes.py floor_plans_resized/floor1/floor1.jpg
"""

import json
import os
import sys

import numpy as np
from PIL import Image


def load_wall_mask(borders_path, red_threshold=180):
    img = Image.open(borders_path).convert("RGB")
    pixels = np.array(img)
    is_wall = (pixels[:, :, 0] > red_threshold) & \
              (pixels[:, :, 1] < 100) & \
              (pixels[:, :, 2] < 100)
    return is_wall


def is_in_wall(wall_mask, x, y, radius=2):
    """Check if position is in or very near a wall (check small area)."""
    h, w = wall_mask.shape
    ix, iy = int(round(x)), int(round(y))
    if not (0 <= ix < w and 0 <= iy < h):
        return True
    region = wall_mask[max(0, iy-radius):iy+radius+1, max(0, ix-radius):ix+radius+1]
    return np.mean(region) > 0.3


def find_nearest_open(wall_mask, x, y, max_search=30):
    """Find the nearest open space to (x, y)."""
    h, w = wall_mask.shape
    best_dist = float('inf')
    best_pos = (x, y)

    for dy in range(-max_search, max_search + 1):
        for dx in range(-max_search, max_search + 1):
            nx, ny = int(round(x + dx)), int(round(y + dy))
            if 0 <= nx < w and 0 <= ny < h:
                if not is_in_wall(wall_mask, nx, ny, radius=2):
                    dist = dx*dx + dy*dy
                    if dist < best_dist:
                        best_dist = dist
                        best_pos = (nx, ny)

    return best_pos


def check_edge(wall_mask, x0, y0, x1, y1, num_samples=60):
    """Check if edge crosses walls. Returns fraction of samples hitting walls."""
    h, w = wall_mask.shape
    ts = np.linspace(0, 1, num_samples)
    xs = x0 + (x1 - x0) * ts
    ys = y0 + (y1 - y0) * ts

    wall_hits = 0
    for x, y in zip(xs, ys):
        ix, iy = int(round(x)), int(round(y))
        if 0 <= ix < w and 0 <= iy < h:
            if wall_mask[iy, ix]:
                wall_hits += 1

    return wall_hits / num_samples


def main():
    if len(sys.argv) < 2:
        print("Usage: ~/isaacsim/python.sh fix_nodes.py <floor_plan.jpg>")
        sys.exit(1)

    image_path = sys.argv[1]
    base = os.path.splitext(image_path)[0]
    json_path = base + "_nodes.json"
    borders_path = base + "_borders.jpg"

    print(f"Loading wall mask from: {borders_path}")
    wall_mask = load_wall_mask(borders_path)
    h, w = wall_mask.shape
    print(f"Wall mask: {w}x{h}")

    with open(json_path) as f:
        data = json.load(f)

    # Step 1: Fix nodes in walls
    print("\n=== Fixing nodes in walls ===")
    node_positions = {}
    moved_count = 0
    for node in data["nodes"]:
        x, y = node["x"], node["y"]
        if is_in_wall(wall_mask, x, y):
            new_x, new_y = find_nearest_open(wall_mask, x, y)
            print(f"  MOVED: {node['id']} ({x},{y}) -> ({new_x},{new_y})")
            node["x"] = new_x
            node["y"] = new_y
            moved_count += 1
        node_positions[node["id"]] = (node["x"], node["y"])

    print(f"  Moved {moved_count} nodes out of walls")

    # Step 2: Check all edges
    print("\n=== Checking edges ===")
    good_edges = []
    bad_edges = []
    for edge in data["edges"]:
        s, e = edge[0], edge[1]
        if s not in node_positions or e not in node_positions:
            continue
        x0, y0 = node_positions[s]
        x1, y1 = node_positions[e]
        frac = check_edge(wall_mask, x0, y0, x1, y1)
        if frac > 0:
            bad_edges.append((edge, frac))
            severity = "HEAVY" if frac > 0.10 else "light"
            print(f"  {severity:5s} {frac:5.0%}: {s} -> {e}")
        else:
            good_edges.append(edge)

    print(f"\n  {len(good_edges)} good, {len(bad_edges)} bad edges")

    # Step 3: For nodes that are near walls, try nudging them 1-3px
    print("\n=== Trying small nudges for borderline nodes ===")
    nudge_helped = 0
    for node in data["nodes"]:
        nid = node["id"]
        x, y = node["x"], node["y"]
        # Try small nudges
        best_x, best_y = x, y
        best_score = 0  # count of edges that become valid
        for dx in range(-4, 5):
            for dy in range(-4, 5):
                nx, ny = x + dx, y + dy
                if is_in_wall(wall_mask, nx, ny, radius=1):
                    continue
                score = 0
                for edge, frac in bad_edges:
                    if edge[0] == nid or edge[1] == nid:
                        other = edge[1] if edge[0] == nid else edge[0]
                        ox, oy = node_positions[other]
                        new_frac = check_edge(wall_mask, nx, ny, ox, oy)
                        if new_frac == 0 and frac > 0:
                            score += 1
                if score > best_score:
                    best_score = score
                    best_x, best_y = nx, ny

        if best_score > 0 and (best_x != x or best_y != y):
            print(f"  NUDGE: {nid} ({x},{y}) -> ({best_x},{best_y}), fixes {best_score} edges")
            node["x"] = best_x
            node["y"] = best_y
            node_positions[nid] = (best_x, best_y)
            nudge_helped += 1

    print(f"  Nudged {nudge_helped} nodes")

    # Step 4: Re-check all edges after nudges
    print("\n=== Final edge check ===")
    final_good = []
    final_bad = []
    for edge in data["edges"]:
        s, e = edge[0], edge[1]
        if s not in node_positions or e not in node_positions:
            continue
        x0, y0 = node_positions[s]
        x1, y1 = node_positions[e]
        frac = check_edge(wall_mask, x0, y0, x1, y1)
        if frac > 0:
            final_bad.append((edge, frac))
        else:
            final_good.append(edge)

    print(f"  {len(final_good)} good, {len(final_bad)} still bad")
    if final_bad:
        print("\n  Still bad:")
        for edge, frac in sorted(final_bad, key=lambda x: -x[1]):
            s, e = edge[0], edge[1]
            print(f"    {frac:5.0%}: {s} -> {e}")

    # Save
    with open(json_path, "w") as f:
        json.dump(data, f, indent=2)
    print(f"\nSaved fixed JSON to: {json_path}")


if __name__ == "__main__":
    main()
