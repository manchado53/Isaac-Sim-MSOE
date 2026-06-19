#!/usr/bin/env python3
"""
visualize_door_tsp.py — generate a 4-panel slideshow visual of the
door-coverage TSP + Dijkstra algorithm on Floor 4.

Renders, on top of floor_4_occupancy.png:
  Panel 1: Substrate     — full 219-node nav graph (all edges + nodes)
  Panel 2: Visit set     — graph faded, 54 door nodes highlighted
  Panel 3: TSP order     — straight arrows between doors in optimal order
                            (what TSP actually optimizes over — ignores corridors)
  Panel 4: Expanded path — purple polyline along Dijkstra-shortest-paths
                            between consecutive doors (what the robot actually drives)

Outputs:
    docs/tsp_dijkstra_explainer.png        — combined 2x2 figure
    docs/tsp_dijkstra_step{1..4}.png       — individual panels for slide reveals

Usage:
    cd ~/isaac-sim-msoe-demo-video
    python3 floorplan_vision/visualize_door_tsp.py
"""

import heapq
import json
import os
import random
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.collections as mc
import matplotlib.image as mpimg
import matplotlib.patches as mpatches
import matplotlib.pyplot as plt
import numpy as np
import yaml


# ---------------------------------------------------------------------------
# Paths
# ---------------------------------------------------------------------------
REPO_ROOT  = Path(__file__).resolve().parent.parent
NAV_GRAPH  = REPO_ROOT / "assets/environments/InnovationCenter/navgraphs/floor_4_auto_navgraph.json"
DOOR_MAP   = REPO_ROOT / "assets/environments/InnovationCenter/navgraphs/floor_4_door_node_map.json"
MAP_META   = REPO_ROOT / "assets/environments/InnovationCenter/navgraphs/floor_4_map_metadata.yaml"
OCC_PNG    = REPO_ROOT / "assets/environments/InnovationCenter/maps/floor_4_occupancy.png"
OUT_DIR    = REPO_ROOT / "docs"

SPAWN_XY   = (-37.20, 82.09)   # floor 4 default spawn from building_info.json


# ---------------------------------------------------------------------------
# Algorithm — inlined so the script runs standalone (mirrors coverage.py)
# ---------------------------------------------------------------------------

def build_adj(graph):
    nodes = {n["id"]: n for n in graph["nodes"]}
    adj   = {nid: [] for nid in nodes}
    for e in graph["edges"]:
        a, b = e["from"], e["to"]
        d    = e["distance_m"]
        adj[a].append((b, d))
        adj[b].append((a, d))
    return nodes, adj


def dijkstra(start, nodes, adj):
    dist = {n: float("inf") for n in nodes}
    prev = {n: None for n in nodes}
    dist[start] = 0.0
    pq, seen = [(0.0, start)], set()
    while pq:
        d, u = heapq.heappop(pq)
        if u in seen: continue
        seen.add(u)
        for v, w in adj[u]:
            nd = d + w
            if nd < dist[v]:
                dist[v] = nd
                prev[v] = u
                heapq.heappush(pq, (nd, v))
    return dist, prev


def reconstruct(prev, end):
    path, cur = [], end
    while cur is not None:
        path.append(cur)
        cur = prev[cur]
    return list(reversed(path))


def tsp_two_opt(visit_set, start, all_dist, num_restarts=10, seed=0):
    random.seed(seed)
    visit = list(visit_set)

    def tour_dist(order):
        return sum(all_dist[order[i]][order[i+1]] for i in range(len(order)-1))

    def greedy(s):
        unvisited = set(visit) - {s}
        order, cur = [s], s
        while unvisited:
            best = min(unvisited, key=lambda n: all_dist[cur][n])
            if all_dist[cur][best] == float("inf"): break
            order.append(best); unvisited.discard(best); cur = best
        return order

    def two_opt(order):
        best, n = order[:], len(order)
        improved = True
        while improved:
            improved = False
            for i in range(1, n-1):
                for j in range(i+1, n):
                    jn = (j+1) % n
                    before = all_dist[best[i-1]][best[i]] + all_dist[best[j]][best[jn]]
                    after  = all_dist[best[i-1]][best[j]] + all_dist[best[i]][best[jn]]
                    if after < before - 1e-9:
                        best[i:j+1] = best[i:j+1][::-1]
                        improved = True
        return best

    best_order = two_opt(greedy(start))
    best_d     = tour_dist(best_order)
    candidates = random.sample([n for n in visit if n != start],
                                min(num_restarts-1, len(visit)-1))
    for s in candidates:
        order = two_opt(greedy(s))
        if start in order:
            i = order.index(start)
            order = order[i:] + order[:i]
        if tour_dist(order) < best_d:
            best_order, best_d = order, tour_dist(order)
    return best_order


def find_nearest_node(graph, xy):
    target = np.array(xy)
    best, best_d = None, float("inf")
    for n in graph["nodes"]:
        d = np.linalg.norm(np.array(n["world_xy_m"]) - target)
        if d < best_d:
            best_d, best = d, n["id"]
    return best


# ---------------------------------------------------------------------------
# Load
# ---------------------------------------------------------------------------

print(f"Loading nav graph:  {NAV_GRAPH}")
graph = json.load(open(NAV_GRAPH))

print(f"Loading door map:   {DOOR_MAP}")
door_map = json.load(open(DOOR_MAP))

print(f"Loading map meta:   {MAP_META}")
meta = yaml.safe_load(open(MAP_META))

print(f"Loading occupancy:  {OCC_PNG}")
occ = mpimg.imread(OCC_PNG)
IMG_H, IMG_W = occ.shape[:2]

ORIGIN_X, ORIGIN_Y, _ = meta["origin"]
RES = float(meta["resolution"])

def world_to_px(x, y):
    col = (x - ORIGIN_X) / RES
    row = IMG_H - (y - ORIGIN_Y) / RES
    return col, row


# ---------------------------------------------------------------------------
# Compute the algorithm output
# ---------------------------------------------------------------------------

nodes, adj = build_adj(graph)
print(f"\nGraph: {len(nodes)} nodes, {len(graph['edges'])} edges")

door_ids = sorted({m["closest_node_id"]
                   for m in door_map["mappings"]
                   if m["closest_node_id"] is not None})
print(f"Door nodes: {len(door_ids)}")

start = find_nearest_node(graph, SPAWN_XY)
print(f"Spawn-nearest node: {start} (xy={nodes[start]['world_xy_m']})")

# Defer to the production pipeline so the visualization always matches what
# the demo actually drives. coverage.generate_door_route filters unreachable
# doors from the visit set (important when the graph has disconnected CCs)
# before running TSP + multi-restart, then Dijkstra-expands the result.
import sys
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from navigation.coverage import (
    generate_door_route,
    _compute_all_pairs as _cov_compute_all_pairs,
    _run_tsp as _cov_run_tsp,
)

# We also want the raw TSP visit order (pre-expansion) for Panel 3.
_cov_all_dist, _cov_all_prev, _cov_reachable = _cov_compute_all_pairs(graph, start)
visit_set = {start} | (set(door_ids) & _cov_reachable)
dropped = set(door_ids) - _cov_reachable
print(f"Visit set (start + doors): {len(visit_set)} nodes "
      f"(dropped {len(dropped)} unreachable doors)")
tsp_order = _cov_run_tsp(visit_set, start, _cov_all_dist,
                          num_restarts=10, seed=0)
print(f"TSP order: {len(tsp_order)} nodes (start + {len(tsp_order) - 1} doors)")
print(f"  First 8: {tsp_order[:8]}")

# Phase 3: Dijkstra-expand into full robot path (same as production).
expanded = generate_door_route(graph, start, door_ids, num_restarts=10, seed=0)
print(f"Expanded path: {len(expanded)} nodes "
      f"({len(expanded)/max(1, len(tsp_order)):.1f}x door count)")


# ---------------------------------------------------------------------------
# Pre-compute pixel coordinates
# ---------------------------------------------------------------------------

node_px  = {nid: world_to_px(*n["world_xy_m"]) for nid, n in nodes.items()}
edge_segs = [[node_px[e["from"]], node_px[e["to"]]] for e in graph["edges"]]

door_px      = np.array([node_px[d] for d in door_ids])
all_px       = np.array(list(node_px.values()))
tsp_path_px  = np.array([node_px[n] for n in tsp_order])
expand_px    = np.array([node_px[n] for n in expanded])
spawn_px     = node_px[start]

# Bounding box of activity (all doors + expansion) with padding
PAD = 60
xmin = max(0,     min(expand_px[:, 0].min(), door_px[:, 0].min()) - PAD)
xmax = min(IMG_W, max(expand_px[:, 0].max(), door_px[:, 0].max()) + PAD)
ymin = max(0,     min(expand_px[:, 1].min(), door_px[:, 1].min()) - PAD)
ymax = min(IMG_H, max(expand_px[:, 1].max(), door_px[:, 1].max()) + PAD)
print(f"\nView window: ({xmin:.0f},{ymin:.0f}) - ({xmax:.0f},{ymax:.0f})")


# ---------------------------------------------------------------------------
# Plotting
# ---------------------------------------------------------------------------

OUT_DIR.mkdir(exist_ok=True, parents=True)


def setup_axis(ax, title):
    ax.set_xlim(xmin, xmax)
    ax.set_ylim(ymax, ymin)
    ax.set_aspect("equal")
    ax.axis("off")
    ax.imshow(occ, extent=[0, IMG_W, IMG_H, 0], aspect="auto", cmap="gray", zorder=0)
    ax.set_title(title, fontsize=18, fontweight="bold", pad=12)


def panel_1_substrate(ax):
    """Full nav graph — all nodes and edges."""
    setup_axis(ax,
        "Step 1 — Substrate\n"
        "Full nav graph: 219 nodes, 226 edges")
    ax.add_collection(mc.LineCollection(edge_segs, colors=(0.2, 0.4, 0.8, 0.55),
                                         linewidths=1.6, zorder=1))
    ax.scatter(all_px[:, 0], all_px[:, 1],
               s=24, c="#1f6feb", zorder=2, linewidths=0,
               label="219 nav-graph nodes")
    ax.scatter([spawn_px[0]], [spawn_px[1]],
               s=240, marker="*", c="#22c55e", zorder=4, linewidths=1.5,
               edgecolors="white", label=f"Robot spawn (node {start})")
    ax.legend(loc="lower right", fontsize=11, framealpha=0.92)


def panel_2_visit_set(ax):
    """Doors highlighted within the faded full graph."""
    setup_axis(ax,
        "Step 2 — Visit set\n"
        f"{len(door_ids)} door nodes — the only ones TSP cares about")
    ax.add_collection(mc.LineCollection(edge_segs, colors=(0.2, 0.4, 0.8, 0.18),
                                         linewidths=1.0, zorder=1))
    ax.scatter(all_px[:, 0], all_px[:, 1],
               s=12, c="#1f6feb", zorder=2, linewidths=0, alpha=0.35,
               label="Corridor / non-door nodes")
    ax.scatter(door_px[:, 0], door_px[:, 1],
               s=110, c="#dc2626", zorder=3, linewidths=1.4,
               edgecolors="white", label=f"{len(door_ids)} door nodes")
    ax.scatter([spawn_px[0]], [spawn_px[1]],
               s=240, marker="*", c="#22c55e", zorder=4, linewidths=1.5,
               edgecolors="white", label=f"Robot spawn (node {start})")
    ax.legend(loc="lower right", fontsize=11, framealpha=0.92)


def panel_3_tsp_order(ax):
    """TSP visit order — straight lines (TSP doesn't see corridors)."""
    setup_axis(ax,
        "Step 3 — TSP solution\n"
        "Optimal door visit order via 2-opt — straight-line view")
    ax.add_collection(mc.LineCollection(edge_segs, colors=(0.2, 0.4, 0.8, 0.10),
                                         linewidths=0.8, zorder=1))
    ax.scatter(door_px[:, 0], door_px[:, 1],
               s=70, c="#dc2626", zorder=3, linewidths=1.0,
               edgecolors="white", alpha=0.85)

    # Straight TSP edges (with arrows showing direction)
    n_arrows = len(tsp_path_px) - 1
    cmap = plt.cm.YlOrRd
    for i in range(n_arrows):
        c = cmap(0.35 + 0.55 * (i / max(1, n_arrows - 1)))
        x0, y0 = tsp_path_px[i]
        x1, y1 = tsp_path_px[i + 1]
        ax.annotate("",
            xy=(x1, y1), xytext=(x0, y0),
            arrowprops=dict(arrowstyle="->", color=c, lw=1.8,
                            shrinkA=4, shrinkB=4),
            zorder=4)

    # Mark spawn / first / last
    ax.scatter([tsp_path_px[0, 0]], [tsp_path_px[0, 1]],
               s=300, marker="*", c="#22c55e", zorder=5, linewidths=1.5,
               edgecolors="white", label="Start")
    ax.scatter([tsp_path_px[-1, 0]], [tsp_path_px[-1, 1]],
               s=180, marker="X", c="#7c3aed", zorder=5, linewidths=1.5,
               edgecolors="white", label="End")

    legend_handles = [
        mpatches.Patch(color="#dc2626", label=f"{len(door_ids)} doors"),
        plt.Line2D([0], [0], color=cmap(0.55), lw=2.2,
                   label="TSP visit order (straight lines)"),
        plt.Line2D([0], [0], marker="*", color="w", markerfacecolor="#22c55e",
                   markersize=14, lw=0, label="Start"),
        plt.Line2D([0], [0], marker="X", color="w", markerfacecolor="#7c3aed",
                   markersize=12, lw=0, label="End"),
    ]
    ax.legend(handles=legend_handles, loc="lower right",
              fontsize=11, framealpha=0.92)


def panel_4_expanded(ax):
    """Final robot route — Dijkstra-expanded along nav graph."""
    setup_axis(ax,
        "Step 4 — Dijkstra expansion\n"
        f"Robot route: {len(expanded)} waypoints along the nav graph")
    ax.add_collection(mc.LineCollection(edge_segs, colors=(0.2, 0.4, 0.8, 0.10),
                                         linewidths=0.8, zorder=1))
    # Faint TSP lines for context
    ax.plot(tsp_path_px[:, 0], tsp_path_px[:, 1],
            color="#fb923c", lw=1.2, ls="--", alpha=0.55, zorder=2,
            label="TSP straight-line order (reference)")
    # Expanded path
    ax.plot(expand_px[:, 0], expand_px[:, 1],
            color="#7c3aed", lw=2.4, zorder=3,
            label=f"Robot route ({len(expanded)} waypoints)")
    # Doors on top
    ax.scatter(door_px[:, 0], door_px[:, 1],
               s=85, c="#dc2626", zorder=4, linewidths=1.0,
               edgecolors="white")
    ax.scatter([tsp_path_px[0, 0]], [tsp_path_px[0, 1]],
               s=300, marker="*", c="#22c55e", zorder=5, linewidths=1.5,
               edgecolors="white", label="Start")
    legend_handles = [
        mpatches.Patch(color="#dc2626", label=f"{len(door_ids)} doors (visit set)"),
        plt.Line2D([0], [0], color="#fb923c", lw=1.4, ls="--",
                   label="TSP straight-line order"),
        plt.Line2D([0], [0], color="#7c3aed", lw=2.4,
                   label=f"Final robot path ({len(expanded)} waypoints)"),
        plt.Line2D([0], [0], marker="*", color="w", markerfacecolor="#22c55e",
                   markersize=14, lw=0, label="Start"),
    ]
    ax.legend(handles=legend_handles, loc="lower right",
              fontsize=11, framealpha=0.92)


# ---------------------------------------------------------------------------
# Render combined 2×2
# ---------------------------------------------------------------------------

fig, axes = plt.subplots(2, 2, figsize=(20, 14), dpi=130)
panel_1_substrate(axes[0, 0])
panel_2_visit_set(axes[0, 1])
panel_3_tsp_order(axes[1, 0])
panel_4_expanded(axes[1, 1])
fig.suptitle(
    "Door-coverage routing: TSP + Dijkstra on the Innovation Center floor 4",
    fontsize=22, fontweight="bold", y=0.995)
fig.tight_layout(rect=[0, 0, 1, 0.97])

combined_path = OUT_DIR / "tsp_dijkstra_explainer.png"
fig.savefig(combined_path, bbox_inches="tight", dpi=130)
print(f"\n  Saved combined: {combined_path}")
plt.close(fig)


# ---------------------------------------------------------------------------
# Render individual panels (for slide-by-slide reveal)
# ---------------------------------------------------------------------------

for i, (panel_fn, name) in enumerate([
    (panel_1_substrate, "step1_substrate"),
    (panel_2_visit_set, "step2_visit_set"),
    (panel_3_tsp_order, "step3_tsp_order"),
    (panel_4_expanded,  "step4_expanded"),
], start=1):
    fig, ax = plt.subplots(1, 1, figsize=(11, 9), dpi=140)
    panel_fn(ax)
    fig.tight_layout()
    out = OUT_DIR / f"tsp_dijkstra_{name}.png"
    fig.savefig(out, bbox_inches="tight", dpi=140)
    print(f"  Saved panel {i}: {out}")
    plt.close(fig)

print("\nDone.")
