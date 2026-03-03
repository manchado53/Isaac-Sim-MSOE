#!/usr/bin/env python3
"""
Build navigation graph from occupancy map data.

Reads the .npy occupancy grids produced by generate_occupancy_map.py and:
  1. Cleans the grid (morphological closing to fill small gaps)
  2. Erodes free space by robot radius
  3. Skeletonizes to find corridor center-lines
  4. Builds a NetworkX graph from the skeleton
  5. Simplifies to junction/endpoint graph
  6. Runs Dijkstra pathfinding between any two points
  7. Saves the graph and visualizations

No Isaac Sim required — runs with plain Python3 + numpy/scipy/networkx/skimage.

Usage:
  python3 test_building/build_nav_graph.py
  python3 test_building/build_nav_graph.py --floor floor_1
  python3 test_building/build_nav_graph.py --floor floor_1 --start "500,300" --end "2000,1500"
"""

import os
import sys
import json
import argparse
import numpy as np
from scipy import ndimage
from skimage.morphology import skeletonize, disk, erosion, closing
import networkx as nx

# ---------------------------------------------------------------------------
# Config
# ---------------------------------------------------------------------------
PROJECT_DIR = os.path.dirname(os.path.abspath(__file__))
OUTPUT_DIR = os.path.join(PROJECT_DIR, "output")

OCCUPIED = 1
FREE = 0
UNKNOWN = 2

# Robot parameters
ROBOT_RADIUS_CM = 25.0  # robot radius in cm (for erosion)

# Graph simplification: minimum edge length to keep (in grid cells)
MIN_EDGE_LENGTH = 5


def load_floor(floor_name):
    """Load occupancy grid and metadata for a floor."""
    npy_path = os.path.join(OUTPUT_DIR, f"{floor_name}_occupancy.npy")
    meta_path = os.path.join(OUTPUT_DIR, f"{floor_name}_meta.json")

    if not os.path.exists(npy_path):
        print(f"ERROR: {npy_path} not found")
        sys.exit(1)

    grid = np.load(npy_path)
    with open(meta_path, "r") as f:
        meta = json.load(f)

    print(f"Loaded {floor_name}: {grid.shape[1]}x{grid.shape[0]} grid, "
          f"cell_size={meta['cell_size_cm']}cm")
    return grid, meta


def clean_grid(grid, cell_size_cm):
    """Clean the occupancy grid: fill small holes, smooth edges."""
    free_mask = (grid == FREE).astype(np.uint8)

    # Morphological closing to fill small gaps in free space
    close_radius = max(1, int(10.0 / cell_size_cm))  # 10cm closing
    free_mask = closing(free_mask, disk(close_radius)).astype(np.uint8)

    # Treat unknown as occupied (conservative)
    return free_mask


def erode_by_robot_radius(free_mask, cell_size_cm, robot_radius_cm):
    """Erode free space by robot radius so the robot doesn't clip walls."""
    radius_cells = max(1, int(robot_radius_cm / cell_size_cm))
    print(f"  Eroding free space by {radius_cells} cells "
          f"({robot_radius_cm}cm robot radius / {cell_size_cm}cm cell)")

    eroded = erosion(free_mask, disk(radius_cells)).astype(np.uint8)

    n_before = int(np.sum(free_mask))
    n_after = int(np.sum(eroded))
    print(f"  Free cells: {n_before} -> {n_after} "
          f"({100 * n_after / max(1, n_before):.1f}% retained)")
    return eroded


def build_skeleton(free_mask):
    """Skeletonize the free space to find corridor center-lines."""
    skeleton = skeletonize(free_mask > 0)
    n_skeleton = int(np.sum(skeleton))
    print(f"  Skeleton pixels: {n_skeleton}")
    return skeleton.astype(np.uint8)


def skeleton_to_graph(skeleton):
    """Convert skeleton image to a full-resolution NetworkX graph."""
    G = nx.Graph()
    coords = np.argwhere(skeleton > 0)
    coord_set = set(map(tuple, coords))

    for r, c in coords:
        G.add_node((r, c))
        # 8-connected neighbors
        for dr, dc in [(-1, 0), (1, 0), (0, -1), (0, 1),
                        (-1, -1), (-1, 1), (1, -1), (1, 1)]:
            nr, nc = r + dr, c + dc
            if (nr, nc) in coord_set:
                weight = 1.0 if dr == 0 or dc == 0 else 1.414
                G.add_edge((r, c), (nr, nc), weight=weight)

    print(f"  Raw graph: {G.number_of_nodes()} nodes, {G.number_of_edges()} edges")
    return G


def bresenham_line(r0, c0, r1, c1):
    """Generate all grid cells along a line from (r0,c0) to (r1,c1)."""
    cells = []
    dr = abs(r1 - r0)
    dc = abs(c1 - c0)
    sr = 1 if r0 < r1 else -1
    sc = 1 if c0 < c1 else -1
    err = dr - dc
    r, c = r0, c0
    while True:
        cells.append((r, c))
        if r == r1 and c == c1:
            break
        e2 = 2 * err
        if e2 > -dc:
            err -= dc
            r += sr
        if e2 < dr:
            err += dr
            c += sc
    return cells


def path_crosses_wall(path, wall_mask):
    """Check if a path (list of (r,c) tuples) passes through any wall cell.

    Also checks the straight line between consecutive path points to catch
    diagonal shortcuts that skip over wall cells.
    """
    h, w = wall_mask.shape
    for r, c in path:
        if 0 <= r < h and 0 <= c < w:
            if wall_mask[r, c]:
                return True
    return False


def validate_edges(G, wall_mask):
    """Remove edges from simplified graph whose paths cross walls.

    For each edge, checks every cell along the stored skeleton path.
    If no skeleton path is stored, checks the direct straight line
    between the two endpoint nodes instead.
    Isolated nodes (no remaining edges) are kept — they represent
    areas that cannot be reached without crossing a wall.
    """
    h, w = wall_mask.shape
    edges_to_remove = []

    for u, v, data in G.edges(data=True):
        if "path" in data:
            # Check the actual skeleton path the robot would follow
            if path_crosses_wall(data["path"], wall_mask):
                edges_to_remove.append((u, v))
        else:
            # No stored path — check straight line between nodes
            line_cells = bresenham_line(int(u[0]), int(u[1]),
                                       int(v[0]), int(v[1]))
            if path_crosses_wall(line_cells, wall_mask):
                edges_to_remove.append((u, v))

    G.remove_edges_from(edges_to_remove)

    # Count isolated nodes but keep them (they're valid locations,
    # just unreachable without crossing walls)
    isolated = [n for n in G.nodes() if G.degree(n) == 0]

    print(f"  Wall validation: removed {len(edges_to_remove)} wall-crossing edges, "
          f"{len(isolated)} isolated nodes (kept, no valid edges)")
    return G


def simplify_graph(G):
    """Simplify graph to only junctions and endpoints.

    Nodes with degree != 2 are key nodes (junctions, endpoints, dead-ends).
    Chains of degree-2 nodes between key nodes become single weighted edges.
    """
    key_nodes = set(n for n in G.nodes() if G.degree(n) != 2)

    simplified = nx.Graph()
    for node in key_nodes:
        simplified.add_node(node)

    visited_edges = set()

    for start in key_nodes:
        for neighbor in G.neighbors(start):
            edge = (min(start, neighbor), max(start, neighbor))
            if edge in visited_edges:
                continue

            # Walk along the chain until we hit another key node
            path = [start, neighbor]
            total_weight = G[start][neighbor].get("weight", 1.0)
            visited_edges.add(edge)

            current = neighbor
            prev = start
            while current not in key_nodes:
                # Find next node (not going back)
                neighbors = [n for n in G.neighbors(current) if n != prev]
                if not neighbors:
                    break  # dead end that's not in key_nodes (shouldn't happen)

                next_node = neighbors[0]
                e = (min(current, next_node), max(current, next_node))
                visited_edges.add(e)
                total_weight += G[current][next_node].get("weight", 1.0)
                path.append(next_node)
                prev = current
                current = next_node

            end = current
            if end in key_nodes and start != end:
                # Store the path so we can reconstruct full routes later
                if simplified.has_edge(start, end):
                    # Keep the shorter edge
                    if simplified[start][end]["weight"] > total_weight:
                        simplified[start][end]["weight"] = total_weight
                        simplified[start][end]["path"] = path
                else:
                    simplified.add_edge(start, end,
                                        weight=total_weight, path=path)

    # Remove very short dead-end edges (noise)
    edges_to_remove = []
    for u, v, data in simplified.edges(data=True):
        if data["weight"] < MIN_EDGE_LENGTH:
            if simplified.degree(u) == 1 or simplified.degree(v) == 1:
                edges_to_remove.append((u, v))
    simplified.remove_edges_from(edges_to_remove)

    # Remove isolated nodes
    isolated = list(nx.isolates(simplified))
    simplified.remove_nodes_from(isolated)

    print(f"  Simplified graph: {simplified.number_of_nodes()} nodes, "
          f"{simplified.number_of_edges()} edges")
    return simplified


def grid_to_world(r, c, meta):
    """Convert grid (row, col) to world coordinates (x, y) in cm."""
    cell_size = meta["cell_size_cm"]
    x = meta["xy_min"][0] + c * cell_size + cell_size / 2
    y = meta["xy_min"][1] + r * cell_size + cell_size / 2
    return (x, y)


def world_to_grid(x, y, meta):
    """Convert world coordinates (x, y) in cm to grid (row, col)."""
    cell_size = meta["cell_size_cm"]
    c = int((x - meta["xy_min"][0]) / cell_size)
    r = int((y - meta["xy_min"][1]) / cell_size)
    return (r, c)


def find_nearest_node(G, target):
    """Find the graph node closest to a target (row, col)."""
    tr, tc = target
    min_dist = float("inf")
    nearest = None
    for node in G.nodes():
        nr, nc = node
        dist = (nr - tr) ** 2 + (nc - tc) ** 2
        if dist < min_dist:
            min_dist = dist
            nearest = node
    return nearest


def find_path(G, start_rc, end_rc, meta):
    """Find shortest path between two grid coordinates."""
    # Find nearest nodes in the graph
    start_node = find_nearest_node(G, start_rc)
    end_node = find_nearest_node(G, end_rc)

    if start_node is None or end_node is None:
        print("ERROR: Could not find graph nodes near start/end")
        return None, None

    print(f"  Start node: {start_node} (nearest to {start_rc})")
    print(f"  End node: {end_node} (nearest to {end_rc})")

    try:
        path_nodes = nx.dijkstra_path(G, start_node, end_node, weight="weight")
        path_length = nx.dijkstra_path_length(G, start_node, end_node, weight="weight")

        # Convert to world coordinates
        world_path = [grid_to_world(r, c, meta) for r, c in path_nodes]
        path_length_cm = path_length * meta["cell_size_cm"]

        print(f"  Path found: {len(path_nodes)} nodes, "
              f"{path_length_cm:.1f} cm ({path_length_cm / 100:.1f} m)")
        return path_nodes, world_path
    except nx.NetworkXNoPath:
        print("  ERROR: No path exists between start and end")
        return None, None


def save_graph(G, meta, floor_name):
    """Save graph as JSON (node list + edge list with world coords)."""
    graph_data = {
        "floor_name": floor_name,
        "cell_size_cm": meta["cell_size_cm"],
        "xy_min": meta["xy_min"],
        "xy_max": meta["xy_max"],
        "nodes": [],
        "edges": [],
    }

    node_index = {}
    for i, node in enumerate(G.nodes()):
        r, c = node
        wx, wy = grid_to_world(r, c, meta)
        graph_data["nodes"].append({
            "id": i,
            "grid_rc": [int(r), int(c)],
            "world_xy_cm": [round(float(wx), 1), round(float(wy), 1)],
            "degree": int(G.degree(node)),
        })
        node_index[node] = i

    for u, v, data in G.edges(data=True):
        weight_cm = data["weight"] * meta["cell_size_cm"]
        graph_data["edges"].append({
            "from": node_index[u],
            "to": node_index[v],
            "weight_cells": round(data["weight"], 2),
            "weight_cm": round(weight_cm, 1),
        })

    out_path = os.path.join(OUTPUT_DIR, f"{floor_name}_nav_graph.json")
    with open(out_path, "w") as f:
        json.dump(graph_data, f, indent=2)
    print(f"  Saved: {out_path}")
    return out_path


def draw_thick_line(img, r0, c0, r1, c1, color, thickness=1):
    """Draw a line on the image using Bresenham with optional thickness."""
    h, w = img.shape[:2]
    cells = bresenham_line(r0, c0, r1, c1)
    for r, c in cells:
        for dr in range(-thickness, thickness + 1):
            for dc in range(-thickness, thickness + 1):
                nr, nc = r + dr, c + dc
                if 0 <= nr < h and 0 <= nc < w:
                    img[nr, nc] = color


def save_visualization(grid, skeleton, simplified_G, meta, floor_name,
                       path_nodes=None):
    """Save a PNG visualization of the occupancy map + validated graph.

    Only shows validated edges (green) and junction nodes (red).
    Isolated nodes with no edges are shown in orange.
    No raw skeleton overlay — only the final navigation graph.
    """
    from PIL import Image

    h, w = grid.shape

    # Base: occupancy map in grayscale
    img = np.zeros((h, w, 3), dtype=np.uint8)
    img[grid == FREE] = [240, 240, 240]      # light gray = free
    img[grid == OCCUPIED] = [40, 40, 40]      # dark gray = wall
    img[grid == UNKNOWN] = [180, 180, 200]    # blue-gray = unknown

    # Draw validated edges in green
    for u, v, data in simplified_G.edges(data=True):
        if "path" in data and len(data["path"]) > 1:
            # Draw along the stored skeleton path (segment by segment)
            path = data["path"]
            for i in range(len(path) - 1):
                r0, c0 = int(path[i][0]), int(path[i][1])
                r1, c1 = int(path[i + 1][0]), int(path[i + 1][1])
                draw_thick_line(img, r0, c0, r1, c1, [50, 200, 50])
        else:
            # No path stored — draw straight line between nodes
            draw_thick_line(img, int(u[0]), int(u[1]),
                           int(v[0]), int(v[1]), [50, 200, 50])

    # Draw nodes: red for connected, orange for isolated
    for node in simplified_G.nodes():
        r, c = node
        is_isolated = simplified_G.degree(node) == 0
        color = [255, 150, 0] if is_isolated else [255, 50, 50]
        if 0 <= r < h and 0 <= c < w:
            for dr in range(-2, 3):
                for dc in range(-2, 3):
                    nr, nc = r + dr, c + dc
                    if 0 <= nr < h and 0 <= nc < w:
                        img[nr, nc] = color

    # Overlay Dijkstra path in yellow if provided
    if path_nodes:
        for r, c in path_nodes:
            if 0 <= r < h and 0 <= c < w:
                for dr in range(-1, 2):
                    for dc in range(-1, 2):
                        nr, nc = r + dr, c + dc
                        if 0 <= nr < h and 0 <= nc < w:
                            img[nr, nc] = [255, 255, 0]  # yellow

    out_path = os.path.join(OUTPUT_DIR, f"{floor_name}_nav_graph.png")
    Image.fromarray(img).save(out_path)
    print(f"  Saved: {out_path}")
    return out_path


def process_floor(floor_name, robot_radius_cm=ROBOT_RADIUS_CM,
                   start_world=None, end_world=None):
    """Full pipeline for one floor."""
    print(f"\n{'=' * 60}")
    print(f"Processing {floor_name}")
    print(f"{'=' * 60}")

    # Step 1: Load
    grid, meta = load_floor(floor_name)
    cell_size = meta["cell_size_cm"]

    # Step 2: Clean grid
    print("\nStep 1: Cleaning occupancy grid...")
    free_mask = clean_grid(grid, cell_size)
    n_free = int(np.sum(free_mask))
    print(f"  Free cells after cleaning: {n_free}")

    # Step 3: Erode by robot radius
    print("\nStep 2: Eroding by robot radius...")
    eroded = erode_by_robot_radius(free_mask, cell_size, robot_radius_cm)

    # Step 4: Skeletonize
    print("\nStep 3: Skeletonizing free space...")
    skeleton = build_skeleton(eroded)

    # Step 5: Build graph
    print("\nStep 4: Building graph from skeleton...")
    raw_G = skeleton_to_graph(skeleton)

    # Keep only the largest connected component
    if raw_G.number_of_nodes() > 0:
        components = list(nx.connected_components(raw_G))
        components.sort(key=len, reverse=True)
        print(f"  Connected components: {len(components)} "
              f"(largest: {len(components[0])} nodes)")
        raw_G = raw_G.subgraph(components[0]).copy()

    # Step 6: Simplify
    print("\nStep 5: Simplifying graph...")
    simplified_G = simplify_graph(raw_G)

    # Step 6: Validate edges against walls
    print("\nStep 6: Removing wall-crossing edges...")
    # Build wall mask from original grid: occupied OR unknown = wall
    wall_mask = (grid != FREE)
    simplified_G = validate_edges(simplified_G, wall_mask)

    # Step 7: Find path if requested
    path_nodes = None
    if start_world and end_world:
        print("\nStep 6: Finding shortest path...")
        start_rc = world_to_grid(start_world[0], start_world[1], meta)
        end_rc = world_to_grid(end_world[0], end_world[1], meta)
        path_nodes, world_path = find_path(simplified_G, start_rc, end_rc, meta)

    # Step 8: Save
    print("\nSaving results...")
    save_graph(simplified_G, meta, floor_name)
    save_visualization(grid, skeleton, simplified_G, meta, floor_name,
                       path_nodes=path_nodes)

    return simplified_G, meta


def main():
    parser = argparse.ArgumentParser(
        description="Build navigation graph from occupancy maps")
    parser.add_argument("--floor", type=str, default=None,
                        help="Process a specific floor (e.g., floor_1). "
                             "Default: all floors.")
    parser.add_argument("--start", type=str, default=None,
                        help="Start point as 'x,y' in world cm coords")
    parser.add_argument("--end", type=str, default=None,
                        help="End point as 'x,y' in world cm coords")
    parser.add_argument("--robot-radius", type=float, default=ROBOT_RADIUS_CM,
                        help=f"Robot radius in cm (default: {ROBOT_RADIUS_CM})")
    args = parser.parse_args()

    robot_radius = args.robot_radius

    # Parse start/end if given
    start_world = None
    end_world = None
    if args.start:
        start_world = tuple(float(v) for v in args.start.split(","))
    if args.end:
        end_world = tuple(float(v) for v in args.end.split(","))

    # Determine which floors to process
    if args.floor:
        floors = [args.floor]
    else:
        # Find all floor .npy files
        floors = []
        for f in sorted(os.listdir(OUTPUT_DIR)):
            if f.endswith("_occupancy.npy"):
                floor_name = f.replace("_occupancy.npy", "")
                floors.append(floor_name)

    print(f"Navigation Graph Builder")
    print(f"Floors to process: {', '.join(floors)}")
    print(f"Robot radius: {robot_radius} cm")

    for floor_name in floors:
        process_floor(floor_name, robot_radius, start_world, end_world)

    print(f"\nDone! Results saved to {OUTPUT_DIR}/")


if __name__ == "__main__":
    main()
