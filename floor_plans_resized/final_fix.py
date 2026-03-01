#!/usr/bin/env python3
"""
Final cleanup of floor1_nodes.json.

1. Fixes nodes whose nudges broke other edges (finds globally optimal positions)
2. For remaining bad edges, uses multi-step waypoint routing
3. Uses a tolerance of 3% for very thin walls (doorways)

Usage:
    ~/isaacsim/python.sh floor_plans_resized/final_fix.py floor_plans_resized/floor1/floor1.jpg
"""

import json
import math
import os
import sys

import numpy as np
from PIL import Image


WALL_TOLERANCE = 0.05  # Allow up to 5% wall pixels (doorways/thin walls)


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


def edge_ok(wall_mask, x0, y0, x1, y1):
    return check_edge(wall_mask, x0, y0, x1, y1) <= WALL_TOLERANCE


def find_best_position(wall_mask, node_positions, node_id, edges, max_nudge=8):
    """Find best position for a node that satisfies most of its edges."""
    orig_x, orig_y = node_positions[node_id]

    # Get all connected nodes
    connected = []
    for edge in edges:
        if edge[0] == node_id:
            connected.append(edge[1])
        elif edge[1] == node_id:
            connected.append(edge[0])

    best_score = -1
    best_pos = (orig_x, orig_y)

    for dx in range(-max_nudge, max_nudge + 1):
        for dy in range(-max_nudge, max_nudge + 1):
            nx, ny = orig_x + dx, orig_y + dy
            if is_in_wall(wall_mask, nx, ny, radius=1):
                continue
            score = 0
            for other_id in connected:
                if other_id in node_positions:
                    ox, oy = node_positions[other_id]
                    if edge_ok(wall_mask, nx, ny, ox, oy):
                        score += 1
            if score > best_score or (score == best_score and abs(dx) + abs(dy) < abs(best_pos[0] - orig_x) + abs(best_pos[1] - orig_y)):
                best_score = score
                best_pos = (nx, ny)

    return best_pos, best_score, len(connected)


def find_2step_waypoint(wall_mask, x0, y0, x1, y1):
    """Find two waypoints for edges that need to go around a large obstacle."""
    # Try various L-shaped and Z-shaped paths
    best = None
    best_score = float('inf')

    # Generate candidate intermediate points
    dx = x1 - x0
    dy = y1 - y0

    candidates = []

    # L-shape options
    candidates.append([(x1, y0)])  # horizontal then vertical
    candidates.append([(x0, y1)])  # vertical then horizontal

    # Z-shape through middle
    mx, my = (x0 + x1) // 2, (y0 + y1) // 2
    for offset_x in range(-30, 31, 5):
        for offset_y in range(-30, 31, 5):
            candidates.append([(mx + offset_x, my + offset_y)])

    # 2-point paths (S-shape, U-shape)
    for step in range(-40, 41, 8):
        # Go wide around obstacle
        candidates.append([(x0 + step, y0), (x1 + step, y1)])
        candidates.append([(x0, y0 + step), (x1, y1 + step)])
        # L then straight
        candidates.append([(x0, y0 + step), (x1, y0 + step)])
        candidates.append([(x0 + step, y0), (x0 + step, y1)])

    for waypoints in candidates:
        # Filter out waypoints in walls
        valid = True
        for wx, wy in waypoints:
            if is_in_wall(wall_mask, wx, wy, radius=1):
                valid = False
                break
        if not valid:
            continue

        # Check all segments
        points = [(x0, y0)] + waypoints + [(x1, y1)]
        total_wall = 0
        for i in range(len(points) - 1):
            frac = check_edge(wall_mask, points[i][0], points[i][1],
                            points[i+1][0], points[i+1][1])
            total_wall += frac

        if total_wall < best_score:
            best_score = total_wall
            best = waypoints

    if best and best_score <= WALL_TOLERANCE * len(best):
        return best, best_score
    return None, float('inf')


def main():
    if len(sys.argv) < 2:
        print("Usage: ~/isaacsim/python.sh final_fix.py <floor_plan.jpg>")
        sys.exit(1)

    image_path = sys.argv[1]
    base = os.path.splitext(image_path)[0]
    json_path = base + "_nodes.json"
    borders_path = base + "_borders.jpg"
    backup_path = base + "_nodes_backup.json"

    print(f"Loading wall mask from: {borders_path}")
    wall_mask = load_wall_mask(borders_path)

    with open(json_path) as f:
        data = json.load(f)

    with open(backup_path) as f:
        backup = json.load(f)

    # Build node position map
    node_positions = {}
    nodes_by_id = {}
    for node in data["nodes"]:
        node_positions[node["id"]] = (node["x"], node["y"])
        nodes_by_id[node["id"]] = node

    # Get ALL desired edges (from backup)
    all_desired_edges = []
    for edge in backup["edges"]:
        all_desired_edges.append(edge)

    # Also include edges from current data (waypoint edges)
    current_edge_set = set()
    for edge in data["edges"]:
        key = tuple(sorted([edge[0], edge[1]]))
        current_edge_set.add(key)

    # Step 1: Optimize node positions globally
    print("\n=== Step 1: Optimizing node positions ===")
    # Build full desired edge list (backup edges + current waypoint edges)
    full_edges = list(backup["edges"]) + [e for e in data["edges"]
                                           if "wp" in e[0] or "wp" in e[1]]

    iterations = 3
    for iteration in range(iterations):
        moved = 0
        for node in data["nodes"]:
            nid = node["id"]
            if "wp" in nid:  # Skip waypoints for now
                continue
            old_pos = node_positions[nid]
            new_pos, score, total = find_best_position(
                wall_mask, node_positions, nid, full_edges, max_nudge=6)
            if new_pos != old_pos:
                node["x"], node["y"] = new_pos
                node_positions[nid] = new_pos
                moved += 1
        print(f"  Iteration {iteration+1}: moved {moved} nodes")
        if moved == 0:
            break

    # Step 2: Check which edges from backup are still bad
    print("\n=== Step 2: Checking backup edges ===")
    good_edges = []
    bad_edges = []

    for edge in backup["edges"]:
        s, e = edge[0], edge[1]
        x0, y0 = node_positions[s]
        x1, y1 = node_positions[e]
        frac = check_edge(wall_mask, x0, y0, x1, y1)
        if frac <= WALL_TOLERANCE:
            good_edges.append(edge)
        else:
            bad_edges.append((edge, frac))

    print(f"  {len(good_edges)} good, {len(bad_edges)} bad (with {WALL_TOLERANCE:.0%} tolerance)")

    # Step 3: For bad edges, try waypoints
    print("\n=== Step 3: Adding waypoints for bad edges ===")
    wp_count = len([n for n in data["nodes"] if "wp" in n["id"]])
    new_waypoint_edges = []

    for edge, frac in bad_edges:
        s, e = edge[0], edge[1]
        x0, y0 = node_positions[s]
        x1, y1 = node_positions[e]

        waypoints, score = find_2step_waypoint(wall_mask, x0, y0, x1, y1)
        if waypoints and score <= WALL_TOLERANCE * (len(waypoints) + 1):
            # Create waypoint nodes and edges
            prev_id = s
            for i, (wx, wy) in enumerate(waypoints):
                wp_id = f"f1_wp_{wp_count}"
                wp_count += 1
                new_node = {"id": wp_id, "name": f"WP {s}-{e}", "x": wx, "y": wy, "type": "hallway"}
                data["nodes"].append(new_node)
                node_positions[wp_id] = (wx, wy)
                new_waypoint_edges.append([prev_id, wp_id])
                prev_id = wp_id
            new_waypoint_edges.append([prev_id, e])
            print(f"  WAYPOINT: {s} -> ... -> {e} ({len(waypoints)} waypoints)")
        else:
            print(f"  FAILED ({frac:.0%}): {s} -> {e}")

    # Step 4: Build final edge list
    # Start with all good backup edges
    final_edges = list(good_edges)
    # Add waypoint edges from current data
    for edge in data["edges"]:
        if "wp" in edge[0] or "wp" in edge[1]:
            key = tuple(sorted([edge[0], edge[1]]))
            if key not in set(tuple(sorted(e)) for e in final_edges):
                final_edges.append(edge)
    # Add new waypoint edges
    for edge in new_waypoint_edges:
        final_edges.append(edge)

    # Deduplicate
    seen = set()
    deduped = []
    for edge in final_edges:
        key = tuple(sorted([edge[0], edge[1]]))
        if key not in seen:
            seen.add(key)
            deduped.append(edge)
    final_edges = deduped

    data["edges"] = final_edges

    # Remove unused waypoint nodes
    used_nodes = set()
    for edge in data["edges"]:
        used_nodes.add(edge[0])
        used_nodes.add(edge[1])
    data["nodes"] = [n for n in data["nodes"] if n["id"] in used_nodes or "wp" not in n["id"]]

    # Final validation
    print(f"\n=== Final Validation ===")
    total_good = 0
    total_bad = 0
    bad_list = []
    for edge in data["edges"]:
        s, e = edge[0], edge[1]
        if s not in node_positions or e not in node_positions:
            continue
        x0, y0 = node_positions[s]
        x1, y1 = node_positions[e]
        frac = check_edge(wall_mask, x0, y0, x1, y1)
        if frac > WALL_TOLERANCE:
            total_bad += 1
            bad_list.append((edge, frac))
        else:
            total_good += 1

    print(f"  {total_good} good, {total_bad} bad (tolerance {WALL_TOLERANCE:.0%})")
    if bad_list:
        for edge, frac in sorted(bad_list, key=lambda x: -x[1]):
            print(f"    {frac:.0%}: {edge[0]} -> {edge[1]}")

    # Save
    with open(json_path, "w") as f:
        json.dump(data, f, indent=2)
    print(f"\nSaved to: {json_path}")
    print(f"Total: {len(data['nodes'])} nodes, {len(data['edges'])} edges")


if __name__ == "__main__":
    main()
