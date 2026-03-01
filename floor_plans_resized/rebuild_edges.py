#!/usr/bin/env python3
"""
Rebuild missing edges in floor1_nodes.json.

After fix_nodes.py removed bad edges, this script:
1. Reads the current (fixed) node positions
2. Adds back all the edges that were removed
3. For edges that cross walls, tries nudging BOTH endpoints slightly
4. For edges that still cross walls, adds intermediate waypoint nodes
5. Validates everything

Usage:
    ~/isaacsim/python.sh floor_plans_resized/rebuild_edges.py floor_plans_resized/floor1/floor1.jpg
"""

import json
import math
import os
import sys

import numpy as np
from PIL import Image


def load_wall_mask(borders_path):
    img = Image.open(borders_path).convert("RGB")
    pixels = np.array(img)
    is_wall = (pixels[:, :, 0] > 180) & \
              (pixels[:, :, 1] < 100) & \
              (pixels[:, :, 2] < 100)
    return is_wall


def is_in_wall(wall_mask, x, y, radius=2):
    h, w = wall_mask.shape
    ix, iy = int(round(x)), int(round(y))
    if not (0 <= ix < w and 0 <= iy < h):
        return True
    region = wall_mask[max(0, iy-radius):iy+radius+1, max(0, ix-radius):ix+radius+1]
    return np.mean(region) > 0.3


def check_edge(wall_mask, x0, y0, x1, y1, num_samples=80):
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


def find_nearest_open(wall_mask, x, y, max_search=25):
    h, w = wall_mask.shape
    best_dist = float('inf')
    best_pos = (x, y)
    for dy in range(-max_search, max_search + 1):
        for dx in range(-max_search, max_search + 1):
            nx, ny = int(round(x + dx)), int(round(y + dy))
            if 0 <= nx < w and 0 <= ny < h:
                if not is_in_wall(wall_mask, nx, ny, radius=1):
                    dist = dx*dx + dy*dy
                    if dist < best_dist:
                        best_dist = dist
                        best_pos = (nx, ny)
    return best_pos


def try_nudge_for_edge(wall_mask, node_positions, nid1, nid2, max_nudge=6):
    """Try nudging both endpoints slightly to make edge valid."""
    x0, y0 = node_positions[nid1]
    x1, y1 = node_positions[nid2]

    best_frac = check_edge(wall_mask, x0, y0, x1, y1)
    best_dx0, best_dy0 = 0, 0
    best_dx1, best_dy1 = 0, 0

    for dx0 in range(-max_nudge, max_nudge + 1, 2):
        for dy0 in range(-max_nudge, max_nudge + 1, 2):
            nx0, ny0 = x0 + dx0, y0 + dy0
            if is_in_wall(wall_mask, nx0, ny0, radius=1):
                continue
            for dx1 in range(-max_nudge, max_nudge + 1, 2):
                for dy1 in range(-max_nudge, max_nudge + 1, 2):
                    nx1, ny1 = x1 + dx1, y1 + dy1
                    if is_in_wall(wall_mask, nx1, ny1, radius=1):
                        continue
                    frac = check_edge(wall_mask, nx0, ny0, nx1, ny1)
                    if frac < best_frac:
                        best_frac = frac
                        best_dx0, best_dy0 = dx0, dy0
                        best_dx1, best_dy1 = dx1, dy1
                    if frac == 0:
                        return (best_dx0, best_dy0, best_dx1, best_dy1, 0)

    return (best_dx0, best_dy0, best_dx1, best_dy1, best_frac)


def find_waypoint(wall_mask, x0, y0, x1, y1):
    """Find an L-shaped waypoint that avoids walls."""
    # Try L-shape: go horizontal first, then vertical
    mid1x, mid1y = x1, y0
    f1a = check_edge(wall_mask, x0, y0, mid1x, mid1y)
    f1b = check_edge(wall_mask, mid1x, mid1y, x1, y1)

    # Try L-shape: go vertical first, then horizontal
    mid2x, mid2y = x0, y1
    f2a = check_edge(wall_mask, x0, y0, mid2x, mid2y)
    f2b = check_edge(wall_mask, mid2x, mid2y, x1, y1)

    score1 = f1a + f1b
    score2 = f2a + f2b

    # Try intermediate waypoints along the corridor
    best_score = min(score1, score2)
    best_wp = None

    if score1 <= score2 and score1 == 0:
        if not is_in_wall(wall_mask, mid1x, mid1y, radius=1):
            return (mid1x, mid1y, 0)

    if score2 <= score1 and score2 == 0:
        if not is_in_wall(wall_mask, mid2x, mid2y, radius=1):
            return (mid2x, mid2y, 0)

    # Try a grid of intermediate points
    for t in [0.25, 0.5, 0.75]:
        mx = int(x0 + (x1 - x0) * t)
        my = int(y0 + (y1 - y0) * t)
        # Search around this midpoint for open space
        for dy in range(-20, 21, 4):
            for dx in range(-20, 21, 4):
                wx, wy = mx + dx, my + dy
                if is_in_wall(wall_mask, wx, wy, radius=1):
                    continue
                fa = check_edge(wall_mask, x0, y0, wx, wy)
                fb = check_edge(wall_mask, wx, wy, x1, y1)
                s = fa + fb
                if s < best_score:
                    best_score = s
                    best_wp = (wx, wy, s)

    if best_wp:
        return best_wp

    # Try L-shapes with nudged corners
    for dx in range(-15, 16, 3):
        for dy in range(-15, 16, 3):
            # L-shape 1 with nudge
            wx, wy = int(x1 + dx), int(y0 + dy)
            if not is_in_wall(wall_mask, wx, wy, radius=1):
                fa = check_edge(wall_mask, x0, y0, wx, wy)
                fb = check_edge(wall_mask, wx, wy, x1, y1)
                if fa + fb < best_score:
                    best_score = fa + fb
                    best_wp = (wx, wy, best_score)

            # L-shape 2 with nudge
            wx, wy = int(x0 + dx), int(y1 + dy)
            if not is_in_wall(wall_mask, wx, wy, radius=1):
                fa = check_edge(wall_mask, x0, y0, wx, wy)
                fb = check_edge(wall_mask, wx, wy, x1, y1)
                if fa + fb < best_score:
                    best_score = fa + fb
                    best_wp = (wx, wy, best_score)

    return best_wp if best_wp else None


def main():
    if len(sys.argv) < 2:
        print("Usage: ~/isaacsim/python.sh rebuild_edges.py <floor_plan.jpg>")
        sys.exit(1)

    image_path = sys.argv[1]
    base = os.path.splitext(image_path)[0]
    json_path = base + "_nodes.json"
    borders_path = base + "_borders.jpg"
    backup_path = base + "_nodes_backup.json"

    print(f"Loading wall mask from: {borders_path}")
    wall_mask = load_wall_mask(borders_path)

    # Load current (fixed) JSON
    with open(json_path) as f:
        data = json.load(f)

    # Load backup to get the full edge list
    with open(backup_path) as f:
        backup = json.load(f)

    # Build node position map from current fixed data
    node_positions = {}
    nodes_by_id = {}
    for node in data["nodes"]:
        node_positions[node["id"]] = (node["x"], node["y"])
        nodes_by_id[node["id"]] = node

    # Get current edges as set of tuples
    current_edges = set()
    for edge in data["edges"]:
        key = tuple(sorted([edge[0], edge[1]]))
        current_edges.add(key)

    # Find missing edges from backup
    missing_edges = []
    for edge in backup["edges"]:
        key = tuple(sorted([edge[0], edge[1]]))
        if key not in current_edges:
            missing_edges.append(edge)

    print(f"Current edges: {len(current_edges)}")
    print(f"Missing edges: {len(missing_edges)}")

    # Try to fix each missing edge
    added_edges = []
    waypoint_count = 0
    still_bad = []

    for edge in missing_edges:
        s, e = edge[0], edge[1]
        x0, y0 = node_positions[s]
        x1, y1 = node_positions[e]
        frac = check_edge(wall_mask, x0, y0, x1, y1)

        if frac == 0:
            # Edge is now valid with fixed positions
            added_edges.append(edge)
            print(f"  FIXED (nudge): {s} -> {e}")
            continue

        if frac <= 0.05:
            # Very minor wall clip - try nudging endpoints
            dx0, dy0, dx1, dy1, new_frac = try_nudge_for_edge(
                wall_mask, node_positions, s, e, max_nudge=5)
            if new_frac == 0:
                # Apply nudges
                new_x0, new_y0 = x0 + dx0, y0 + dy0
                new_x1, new_y1 = x1 + dx1, y1 + dy1
                if dx0 != 0 or dy0 != 0:
                    nodes_by_id[s]["x"] = new_x0
                    nodes_by_id[s]["y"] = new_y0
                    node_positions[s] = (new_x0, new_y0)
                    print(f"  NUDGED {s}: ({x0},{y0}) -> ({new_x0},{new_y0})")
                if dx1 != 0 or dy1 != 0:
                    nodes_by_id[e]["x"] = new_x1
                    nodes_by_id[e]["y"] = new_y1
                    node_positions[e] = (new_x1, new_y1)
                    print(f"  NUDGED {e}: ({x1},{y1}) -> ({new_x1},{new_y1})")
                added_edges.append(edge)
                print(f"  FIXED (nudge): {s} -> {e}")
                continue

        # Try waypoint
        wp = find_waypoint(wall_mask, x0, y0, x1, y1)
        if wp and wp[2] == 0:
            wx, wy = wp[0], wp[1]
            # Add waypoint node
            wp_id = f"f1_wp_{waypoint_count}"
            wp_name = f"Waypoint {s}-{e}"
            new_node = {"id": wp_id, "name": wp_name, "x": wx, "y": wy, "type": "hallway"}
            data["nodes"].append(new_node)
            node_positions[wp_id] = (wx, wy)
            nodes_by_id[wp_id] = new_node
            waypoint_count += 1

            # Add two edges through waypoint
            added_edges.append([s, wp_id])
            added_edges.append([wp_id, e])
            print(f"  WAYPOINT: {s} -> [{wp_id} at ({wx},{wy})] -> {e}")
        else:
            severity = f"{frac:.0%}"
            still_bad.append((edge, frac))
            print(f"  FAILED ({severity}): {s} -> {e}")

    # Add all fixed edges
    for edge in added_edges:
        data["edges"].append(edge)

    print(f"\n=== Summary ===")
    print(f"  Added {len(added_edges)} edges (including waypoint splits)")
    print(f"  Added {waypoint_count} waypoint nodes")
    print(f"  Still missing: {len(still_bad)} edges")

    if still_bad:
        print("\n  Still bad edges:")
        for edge, frac in sorted(still_bad, key=lambda x: -x[1]):
            print(f"    {frac:.0%}: {edge[0]} -> {edge[1]}")

    # Final validation
    print(f"\n=== Final Validation ===")
    total_good = 0
    total_bad = 0
    for edge in data["edges"]:
        s, e = edge[0], edge[1]
        if s not in node_positions or e not in node_positions:
            continue
        x0, y0 = node_positions[s]
        x1, y1 = node_positions[e]
        frac = check_edge(wall_mask, x0, y0, x1, y1)
        if frac > 0:
            total_bad += 1
            print(f"  BAD {frac:.0%}: {s} -> {e}")
        else:
            total_good += 1

    print(f"  {total_good} good, {total_bad} bad")

    # Save
    with open(json_path, "w") as f:
        json.dump(data, f, indent=2)
    print(f"\nSaved to: {json_path}")
    print(f"Total: {len(data['nodes'])} nodes, {len(data['edges'])} edges")


if __name__ == "__main__":
    main()
