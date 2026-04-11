#!/usr/bin/env python3
"""
auto_graph_generator.py

Automated navigation graph generation for any floor of the MSOE building.
Supports all four V4 floor plans (floor_1 – floor_4) from test_building/V4_floor_plans/.

Pipeline:
  Stage 1 — Load:            read occupancy PNG + meta.json
  Stage 2 — Safety erosion:  erode free space by robot clearance
  Stage 3 — Skeletonization: medial axis via morphological thinning (scikit-image)
  Stage 4 — Skeleton graph:  8-connected pixel graph + junction/endpoint detection
  Stage 5 — Node sampling:   junctions, endpoints, and spaced intervals along branches
  Stage 6 — Waypoint graph:  sparse nav graph with Euclidean edge weights
  Stage 7 — Export:          JSON + GML (waypoint graph) + YAML (map metadata) + PNG overlay

NOTE on SWAGGER:
  ~/isaacsim/SWAGGER/ is cloned and available. SWAGGER uses GPU-accelerated cucim/cupy
  for skeletonization. To use it on a GPU node:
      pip install cupy-cuda12x cucim-cu12 && pip install -e ~/isaacsim/SWAGGER/
  Then replace the _skeletonize() call below with:
      from swagger import WaypointGraphGenerator, WaypointGraphGeneratorConfig
  The output is a networkx.Graph with x/y node attributes in world meters.

Usage:
    python3 floorplan_vision/auto_graph_generator.py --floor floor_1
    python3 floorplan_vision/auto_graph_generator.py --floor floor_3 --sample-dist 1.5
    python3 floorplan_vision/auto_graph_generator.py --floor floor_2 --safety-dist 0.50

Authors: MSOE AI Research Team
"""

import argparse
import json
import os
import sys

import cv2
import networkx as nx
import numpy as np
from scipy.ndimage import label as scipy_label
from skimage.morphology import disk, erosion, skeletonize

# ---------------------------------------------------------------------------
# Paths
# ---------------------------------------------------------------------------
REPO_DIR        = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
V4_DIR          = os.path.join(REPO_DIR, "test_building", "V4_floor_plans")
DEFAULT_OUT_DIR = os.path.join(REPO_DIR, "floorplan_vision", "results")

V4_FLOORS = ["floor_1", "floor_2", "floor_3", "floor_4"]

# ---------------------------------------------------------------------------
# Robot / pipeline defaults
# ---------------------------------------------------------------------------
DEFAULT_SAFETY_DIST_M = 0.36   # Nova Carter half-width + margin (metres)
DEFAULT_SAMPLE_DIST_M = 2.0    # spacing between skeleton nodes (metres)
DEFAULT_MERGE_DIST_M  = 0.8    # collapse junction clusters within this radius
DEFAULT_MIN_STUB_M    = 2.0    # prune dead-end edges shorter than this
OCCUPANCY_THRESHOLD   = 127    # pixels above this are free space

FREE = 255
WALL = 0


# ---------------------------------------------------------------------------
# Stage 1 — Load
# ---------------------------------------------------------------------------

def load_v4_floor(floor_name):
    """
    Load a V4 floor plan PNG and its meta.json from test_building/V4_floor_plans/.

    V4 coordinate convention (important for pixel↔world mapping):
      • cell_size_cm field is in METRES (legacy field name)
      • row 0 of the PNG = world y = xy_min[1]  (y INCREASES with row)
      • col 0 of the PNG = world x = xy_min[0]  (x increases with col)
    """
    img_path  = os.path.join(V4_DIR, f"{floor_name}_occupancy.png")
    meta_path = os.path.join(V4_DIR, f"{floor_name}_meta.json")

    for p in (img_path, meta_path):
        if not os.path.exists(p):
            sys.exit(f"ERROR: file not found:\n  {p}")

    img = cv2.imread(img_path, cv2.IMREAD_GRAYSCALE)
    if img is None:
        sys.exit(f"ERROR: cv2 could not read {img_path}")

    img_h, img_w = img.shape

    with open(meta_path) as f:
        raw = json.load(f)

    resolution_m = raw["cell_size_cm"]   # metres (field name is a legacy misnomer)
    xy_min       = raw["xy_min"]         # [x_min, y_min] in metres
    xy_max       = raw["xy_max"]         # [x_max, y_max] in metres

    meta = {
        "floor":        floor_name,
        "resolution_m": resolution_m,
        "cell_size_cm": resolution_m,    # keep for JSON compat
        "xy_min":       xy_min,
        "xy_max":       xy_max,
        "grid_width":   img_w,
        "grid_height":  img_h,
    }

    print(f"Loaded {floor_name}: {img_w}×{img_h} px  "
          f"res={resolution_m:.4f} m/px  "
          f"world=[({xy_min[0]:.2f},{xy_min[1]:.2f})→({xy_max[0]:.2f},{xy_max[1]:.2f})] m")
    return img, meta


def pixel_to_world(row, col, meta):
    """
    Convert pixel (row, col) → world (x, y) in metres.
    V4 convention: row 0 = xy_min[1], y increases with row.
    """
    res   = meta["resolution_m"]
    x     = meta["xy_min"][0] + col * res
    y     = meta["xy_min"][1] + row * res
    return x, y


def world_to_pixel(x, y, meta):
    """Inverse of pixel_to_world — returns (row, col)."""
    res = meta["resolution_m"]
    col = int((x - meta["xy_min"][0]) / res)
    row = int((y - meta["xy_min"][1]) / res)
    return row, col


# ---------------------------------------------------------------------------
# Stage 2 — Safety erosion
# ---------------------------------------------------------------------------

def erode_free_space(img, meta, safety_dist_m):
    """Erode free space by robot safety distance."""
    res       = meta["resolution_m"]
    radius_px = max(1, int(round(safety_dist_m / res)))
    print(f"  Safety erosion: {safety_dist_m:.2f} m → {radius_px} px radius")

    free_mask = (img > OCCUPANCY_THRESHOLD).astype(np.uint8)
    eroded    = erosion(free_mask, disk(radius_px))

    n_before = int(free_mask.sum())
    n_after  = int(eroded.sum())
    print(f"  Free pixels: {n_before:,} → {n_after:,} "
          f"({100 * n_after / max(1, n_before):.1f}% retained)")
    return eroded


# ---------------------------------------------------------------------------
# Stage 3 — Skeletonization
# ---------------------------------------------------------------------------

def _skeletonize(eroded_mask):
    """1-px-wide medial axis via morphological thinning (CPU equivalent of SWAGGER)."""
    return skeletonize(eroded_mask > 0).astype(np.uint8)


# ---------------------------------------------------------------------------
# Stage 4 — Skeleton graph + key pixel detection
# ---------------------------------------------------------------------------

def _neighbor_count(skel, r, c):
    h, w  = skel.shape
    count = 0
    for dr in (-1, 0, 1):
        for dc in (-1, 0, 1):
            if dr == 0 and dc == 0:
                continue
            nr, nc = r + dr, c + dc
            if 0 <= nr < h and 0 <= nc < w and skel[nr, nc]:
                count += 1
    return count


def build_skeleton_graph(skeleton):
    """Build an 8-connected NetworkX graph of every skeleton pixel."""
    G          = nx.Graph()
    coords     = np.argwhere(skeleton > 0)
    coord_set  = {(int(r), int(c)) for r, c in coords}

    for r, c in coords:
        node = (int(r), int(c))
        G.add_node(node)
        for dr, dc in [(-1,0),(1,0),(0,-1),(0,1),(-1,-1),(-1,1),(1,-1),(1,1)]:
            nb = (r + dr, c + dc)
            if nb in coord_set:
                w_edge = 1.0 if (dr == 0 or dc == 0) else 1.414
                G.add_edge(node, nb, weight=w_edge)

    print(f"  Raw skeleton graph: {G.number_of_nodes():,} nodes, "
          f"{G.number_of_edges():,} edges")
    return G


def find_key_pixels(skeleton):
    """Return junction (degree≥3) and endpoint (degree==1) pixel sets."""
    junctions = set()
    endpoints = set()
    for r, c in np.argwhere(skeleton > 0):
        n = _neighbor_count(skeleton, int(r), int(c))
        if n >= 3:
            junctions.add((int(r), int(c)))
        elif n == 1:
            endpoints.add((int(r), int(c)))
    print(f"  Key pixels: {len(junctions)} junctions, {len(endpoints)} endpoints")
    return junctions, endpoints


# ---------------------------------------------------------------------------
# Stage 5 — Node sampling
# ---------------------------------------------------------------------------

def sample_nodes(skeleton_G, skeleton, junctions, endpoints, meta, sample_dist_m):
    """
    Place waypoint nodes at junctions, endpoints, and every sample_dist_m along branches.
    Returns a list of (row, col) node positions.
    """
    sample_dist_px = sample_dist_m / meta["resolution_m"]
    key_pixels     = junctions | endpoints

    if skeleton_G.number_of_nodes() == 0:
        return []

    components = sorted(nx.connected_components(skeleton_G), key=len, reverse=True)
    main_cc    = skeleton_G.subgraph(components[0]).copy()
    print(f"  Largest CC: {main_cc.number_of_nodes():,} nodes "
          f"(of {skeleton_G.number_of_nodes():,} total)")

    nodes_set  = set()
    main_nodes = set(main_cc.nodes())

    for px in key_pixels:
        if px in main_nodes:
            nodes_set.add(px)

    visited_edges = set()

    def walk_branch(start, via):
        path = [start, via]
        dist = main_cc[start][via]["weight"] if main_cc.has_edge(start, via) else 1.0
        prev, cur = start, via
        while cur not in key_pixels:
            nbrs = [n for n in main_cc.neighbors(cur) if n != prev]
            if not nbrs:
                break
            nxt = nbrs[0]
            path.append(nxt)
            if main_cc.has_edge(cur, nxt):
                dist += main_cc[cur][nxt]["weight"]
            prev, cur = cur, nxt
        return path, dist

    for start in key_pixels:
        if start not in main_nodes:
            continue
        for via in list(main_cc.neighbors(start)):
            edge_key = (min(start, via), max(start, via))
            if edge_key in visited_edges:
                continue
            visited_edges.add(edge_key)
            path, total_px = walk_branch(start, via)
            if total_px > sample_dist_px:
                acc         = 0.0
                last_placed = 0.0
                for i in range(1, len(path)):
                    seg = main_cc[path[i-1]][path[i]]["weight"] \
                          if main_cc.has_edge(path[i-1], path[i]) else 1.0
                    acc += seg
                    if acc - last_placed >= sample_dist_px:
                        nodes_set.add(path[i])
                        last_placed = acc

    print(f"  Sampled nodes: {len(nodes_set)} total")
    return list(nodes_set)


# ---------------------------------------------------------------------------
# Stage 6 — Sparse waypoint graph
# ---------------------------------------------------------------------------

def build_waypoint_graph(node_pixels, skeleton_G, meta, merge_dist_m, min_stub_m):
    """Build a sparse navigation graph from sampled node positions."""
    if not node_pixels:
        return nx.Graph()

    res = meta["resolution_m"]
    G   = nx.Graph()

    for i, (r, c) in enumerate(node_pixels):
        x, y = pixel_to_world(r, c, meta)
        G.add_node(i, row=int(r), col=int(c), x=round(x, 4), y=round(y, 4))

    pixel_to_id    = {(d["row"], d["col"]): nid for nid, d in G.nodes(data=True)}
    node_pixels_set = set(node_pixels)
    main_nodes     = set(skeleton_G.nodes())
    added_edges    = 0

    print("  Building edges via skeleton path search...")
    for i, (r1, c1) in enumerate(node_pixels):
        if (r1, c1) not in main_nodes:
            continue
        visited  = {(r1, c1)}
        frontier = [(r1, c1)]
        dist_px  = {(r1, c1): 0.0}

        while frontier:
            next_frontier = []
            for cur in frontier:
                for nb in skeleton_G.neighbors(cur):
                    if nb in visited:
                        continue
                    visited.add(nb)
                    dist_px[nb] = dist_px[cur] + skeleton_G[cur][nb].get("weight", 1.0)
                    if nb in node_pixels_set and nb != (r1, c1):
                        j = pixel_to_id[nb]
                        if i < j and not G.has_edge(i, j):
                            d1, d2 = G.nodes[i], G.nodes[j]
                            euc = ((d1["x"] - d2["x"])**2 + (d1["y"] - d2["y"])**2)**0.5
                            G.add_edge(i, j, weight=round(euc, 4))
                            added_edges += 1
                        continue
                    next_frontier.append(nb)
            frontier = next_frontier

    print(f"  Edges added: {added_edges}")

    # Merge close nodes
    merge_dist_px = merge_dist_m / res
    nodes_to_remove = set()
    node_ids = list(G.nodes())
    for i in range(len(node_ids)):
        for j in range(i + 1, len(node_ids)):
            ni, nj = node_ids[i], node_ids[j]
            if ni in nodes_to_remove or nj in nodes_to_remove:
                continue
            di, dj = G.nodes[ni], G.nodes[nj]
            if ((di["row"]-dj["row"])**2 + (di["col"]-dj["col"])**2)**0.5 < merge_dist_px:
                for nb in list(G.neighbors(nj)):
                    if nb != ni and not G.has_edge(ni, nb):
                        euc = ((di["x"]-G.nodes[nb]["x"])**2 +
                               (di["y"]-G.nodes[nb]["y"])**2)**0.5
                        G.add_edge(ni, nb, weight=round(euc, 4))
                nodes_to_remove.add(nj)
    G.remove_nodes_from(nodes_to_remove)
    print(f"  After merge ({merge_dist_m} m): {G.number_of_nodes()} nodes")

    # Prune short dead-end stubs
    min_stub_px = min_stub_m / res
    changed = True
    while changed:
        changed = False
        for nid in list(G.nodes()):
            if G.degree(nid) == 1:
                nb       = next(iter(G.neighbors(nid)))
                edge_len = G[nid][nb]["weight"] / res
                if edge_len < min_stub_px:
                    G.remove_node(nid)
                    changed = True

    # Remove small disconnected fragments
    for comp in list(nx.connected_components(G)):
        sub = G.subgraph(comp)
        total_len = sum(d["weight"] for _, _, d in sub.edges(data=True))
        if len(comp) < 3 or total_len < min_stub_m * 3:
            if len(comp) < max(nx.connected_components(G), key=len):
                G.remove_nodes_from(comp)

    G = nx.convert_node_labels_to_integers(G)
    print(f"  Final graph: {G.number_of_nodes()} nodes, {G.number_of_edges()} edges")
    return G


# ---------------------------------------------------------------------------
# Stage 7 — Export: JSON, GML, YAML, PNG
# ---------------------------------------------------------------------------

def export_json(G, meta, out_dir, floor_name):
    """Export navigation graph to pipeline JSON format."""
    os.makedirs(out_dir, exist_ok=True)
    path = os.path.join(out_dir, f"{floor_name}_auto_navgraph.json")

    data = {
        "floor":        floor_name,
        "generated_by": "auto_graph_generator.py",
        "resolution_m": meta["resolution_m"],
        "xy_min":       meta["xy_min"],
        "xy_max":       meta["xy_max"],
        "total_nodes":  G.number_of_nodes(),
        "total_edges":  G.number_of_edges(),
        "nodes": [
            {
                "id":          int(nid),
                "world_xy_m":  [d["x"], d["y"]],
                "grid_rc":     [d["row"], d["col"]],
            }
            for nid, d in G.nodes(data=True)
        ],
        "edges": [
            {"from": int(u), "to": int(v), "distance_m": d["weight"]}
            for u, v, d in G.edges(data=True)
        ],
    }
    with open(path, "w") as f:
        json.dump(data, f, indent=2)
    print(f"  Saved JSON: {path}")
    return path


def export_gml(G, out_dir, floor_name):
    """Export navigation graph as GML (SWAGGER waypoint graph format)."""
    os.makedirs(out_dir, exist_ok=True)
    path = os.path.join(out_dir, f"{floor_name}_auto_navgraph.gml")
    nx.write_gml(G, path)
    print(f"  Saved GML:  {path}")
    return path


def export_yaml(meta, out_dir, floor_name):
    """
    Export map metadata as YAML (ROS map_server compatible).

    origin: world (x, y) of the pixel at row=0, col=0 — which in V4 convention
    is the top-left corner of the image = (xy_min[0], xy_min[1]).
    """
    os.makedirs(out_dir, exist_ok=True)
    path = os.path.join(out_dir, f"{floor_name}_map_metadata.yaml")
    res  = meta["resolution_m"]
    xmin, ymin = meta["xy_min"]
    xmax, ymax = meta["xy_max"]

    lines = [
        f"# Map metadata — {floor_name}",
        f"# Generated by auto_graph_generator.py",
        f"# V4 image convention: row 0 = world y_min, y increases downward in image",
        f"",
        f"image: {floor_name}_occupancy.png",
        f"resolution: {res}         # metres per pixel",
        f"origin: [{xmin:.6f}, {ymin:.6f}, 0.0]"
        f"  # world (x, y, theta) of top-left pixel (row=0, col=0)",
        f"negate: 0",
        f"occupied_thresh: 0.65",
        f"free_thresh: 0.196",
        f"",
        f"floor: {floor_name}",
        f"grid_width:  {meta['grid_width']}",
        f"grid_height: {meta['grid_height']}",
        f"xy_min: [{xmin:.6f}, {ymin:.6f}]   # metres",
        f"xy_max: [{xmax:.6f}, {ymax:.6f}]   # metres",
    ]
    with open(path, "w") as f:
        f.write("\n".join(lines) + "\n")
    print(f"  Saved YAML: {path}")
    return path


def save_visualization(img, G, meta, out_dir, floor_name):
    """Overlay the navigation graph on the occupancy map PNG."""
    os.makedirs(out_dir, exist_ok=True)
    vis = cv2.cvtColor(img, cv2.COLOR_GRAY2BGR)

    # Edges (cyan)
    for u, v in G.edges():
        du, dv = G.nodes[u], G.nodes[v]
        r1, c1 = world_to_pixel(du["x"], du["y"], meta)
        r2, c2 = world_to_pixel(dv["x"], dv["y"], meta)
        cv2.line(vis, (c1, r1), (c2, r2), (255, 200, 0), 2)

    # Nodes (red)
    for nid, data in G.nodes(data=True):
        r, c = world_to_pixel(data["x"], data["y"], meta)
        cv2.circle(vis, (c, r), 5, (0, 50, 255), -1)

    path = os.path.join(out_dir, f"{floor_name}_auto_navgraph.png")
    cv2.imwrite(path, vis)
    print(f"  Saved PNG:  {path}")
    return path


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser(
        description="Automated navigation graph from V4 occupancy maps")
    parser.add_argument(
        "--floor", type=str, default="floor_1", choices=V4_FLOORS,
        help=f"Which floor to process (default: floor_1). Choices: {V4_FLOORS}")
    parser.add_argument(
        "--safety-dist", type=float, default=DEFAULT_SAFETY_DIST_M, metavar="M",
        help=f"Robot safety clearance in metres (default: {DEFAULT_SAFETY_DIST_M})")
    parser.add_argument(
        "--sample-dist", type=float, default=DEFAULT_SAMPLE_DIST_M, metavar="M",
        help=f"Node spacing along skeleton in metres (default: {DEFAULT_SAMPLE_DIST_M})")
    parser.add_argument(
        "--merge-dist", type=float, default=DEFAULT_MERGE_DIST_M, metavar="M",
        help=f"Merge nodes closer than this (default: {DEFAULT_MERGE_DIST_M})")
    parser.add_argument(
        "--min-stub", type=float, default=DEFAULT_MIN_STUB_M, metavar="M",
        help=f"Prune dead-ends shorter than this (default: {DEFAULT_MIN_STUB_M})")
    parser.add_argument(
        "--out-dir", type=str, default=DEFAULT_OUT_DIR,
        help=f"Output directory (default: {DEFAULT_OUT_DIR})")
    args = parser.parse_args()

    print("=" * 60)
    print(f"Navigation Graph Generator — {args.floor}")
    print("=" * 60)
    print(f"  safety_dist = {args.safety_dist} m")
    print(f"  sample_dist = {args.sample_dist} m")
    print(f"  merge_dist  = {args.merge_dist} m")
    print(f"  min_stub    = {args.min_stub} m")
    print(f"  out_dir     = {args.out_dir}")
    print()

    # Stage 1
    print("Stage 1: Loading occupancy map...")
    img, meta = load_v4_floor(args.floor)

    # Stage 2
    print("\nStage 2: Safety erosion...")
    eroded = erode_free_space(img, meta, args.safety_dist)

    # Stage 3
    print("\nStage 3: Skeletonization (medial axis)...")
    skeleton = _skeletonize(eroded)
    print(f"  Skeleton pixels: {int(skeleton.sum()):,}")

    # Stage 4
    print("\nStage 4: Building skeleton graph...")
    skeleton_G          = build_skeleton_graph(skeleton)
    junctions, endpoints = find_key_pixels(skeleton)

    # Stage 5
    print("\nStage 5: Sampling waypoint nodes...")
    node_pixels = sample_nodes(
        skeleton_G, skeleton, junctions, endpoints, meta, args.sample_dist)
    if not node_pixels:
        sys.exit("ERROR: No nodes sampled — check occupancy map and parameters.")

    # Stage 6
    print("\nStage 6: Building sparse waypoint graph...")
    G = build_waypoint_graph(
        node_pixels, skeleton_G, meta, args.merge_dist, args.min_stub)

    # Stage 7
    print("\nStage 7: Exporting results...")
    export_json(G, meta, args.out_dir, args.floor)
    export_gml(G, args.out_dir, args.floor)
    export_yaml(meta, args.out_dir, args.floor)
    save_visualization(img, G, meta, args.out_dir, args.floor)

    print()
    print("=" * 60)
    print(f"Done — {args.floor}: {G.number_of_nodes()} nodes, "
          f"{G.number_of_edges()} edges")
    print("=" * 60)


if __name__ == "__main__":
    main()
