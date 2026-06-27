#!/usr/bin/env python3
"""Fig 4 — occupancy map, SWAGGER navigation graph, and the door-inspection route.

Two paper-clean panels on the real floor-4 occupancy map:
  (a) full nav graph (219 nodes / 226 edges) with the 47 door nodes highlighted
  (b) the Dijkstra-expanded robot route through every door node.
Computation reuses the production coverage pipeline (navigation.coverage), so the
route is exactly what the robot drives.
"""
import json
import sys
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.collections as mc
import matplotlib.image as mpimg
import matplotlib.patches as mpatches
import matplotlib.pyplot as plt
import numpy as np
import yaml

ROOT = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(ROOT))
from navigation.coverage import generate_door_route  # noqa: E402

NAV = ROOT / "assets/environments/InnovationCenter/navgraphs"
MAPS = ROOT / "assets/environments/InnovationCenter/maps"
OUT = Path(__file__).resolve().parent.parent / "latex" / "figures"
OUT.mkdir(parents=True, exist_ok=True)

plt.rcParams.update({"font.family": "serif", "font.size": 9})

graph = json.load(open(NAV / "floor_4_auto_navgraph.json"))
door_map = json.load(open(NAV / "floor_4_door_node_map.json"))
meta = yaml.safe_load(open(NAV / "floor_4_map_metadata.yaml"))
occ = mpimg.imread(MAPS / "floor_4_occupancy.png")
IMG_H, IMG_W = occ.shape[:2]
ORIGIN_X, ORIGIN_Y, _ = meta["origin"]
RES = float(meta["resolution"])

SPAWN_XY = (-37.20, 82.09)


def world_to_px(x, y):
    return (x - ORIGIN_X) / RES, IMG_H - (y - ORIGIN_Y) / RES


nodes = {n["id"]: n for n in graph["nodes"]}
# door node IDs (nearest graph node per door) drive the TSP route ...
door_ids = sorted({m["closest_node_id"] for m in door_map["mappings"]
                   if m["closest_node_id"] is not None})
# ... but the door markers are drawn at the doors' TRUE world positions
# (door_world_xy_m), which sit in the wall gaps, not at the corridor node.
doors_xy = [m["door_world_xy_m"] for m in door_map["mappings"]
            if m.get("door_world_xy_m")]

start = min(nodes, key=lambda nid: np.hypot(
    nodes[nid]["world_xy_m"][0] - SPAWN_XY[0],
    nodes[nid]["world_xy_m"][1] - SPAWN_XY[1]))
expanded = generate_door_route(graph, start, door_ids, num_restarts=10, seed=0)

node_px = {nid: world_to_px(*n["world_xy_m"]) for nid, n in nodes.items()}
edge_segs = [[node_px[e["from"]], node_px[e["to"]]] for e in graph["edges"]]
all_px = np.array(list(node_px.values()))
door_px = np.array([world_to_px(*xy) for xy in doors_xy])
expand_px = np.array([node_px[n] for n in expanded])
spawn_px = node_px[start]
N_DOORS = len(doors_xy)

PAD = 55
xmin = max(0, min(expand_px[:, 0].min(), door_px[:, 0].min()) - PAD)
xmax = min(IMG_W, max(expand_px[:, 0].max(), door_px[:, 0].max()) + PAD)
ymin = max(0, min(expand_px[:, 1].min(), door_px[:, 1].min()) - PAD)
ymax = min(IMG_H, max(expand_px[:, 1].max(), door_px[:, 1].max()) + PAD)


def base(ax, sub):
    ax.imshow(occ, extent=[0, IMG_W, IMG_H, 0], aspect="equal", cmap="gray", zorder=0)
    ax.set_xlim(xmin, xmax)
    ax.set_ylim(ymax, ymin)
    ax.axis("off")
    ax.text(0.5, -0.03, sub, transform=ax.transAxes, ha="center", va="top",
            fontsize=9)


fig, (axA, axB) = plt.subplots(1, 2, figsize=(7.0, 3.0))

# Panel (a): graph + doors
base(axA, "(a) Navigation graph + doors")
axA.add_collection(mc.LineCollection(edge_segs, colors=(0.13, 0.43, 0.92, 0.55),
                                     linewidths=0.8, zorder=1))
axA.scatter(all_px[:, 0], all_px[:, 1], s=4, c="#1f6feb", zorder=2, linewidths=0)
axA.scatter(door_px[:, 0], door_px[:, 1], s=26, c="#dc2626", zorder=3,
            linewidths=0.5, edgecolors="white")
axA.scatter([spawn_px[0]], [spawn_px[1]], s=130, marker="*", c="#22c55e",
            zorder=4, linewidths=0.8, edgecolors="white")
axA.legend(handles=[
    plt.Line2D([0], [0], color="#1f6feb", lw=1.4, marker="o", markersize=4,
               label="219 nodes / 226 edges"),
    plt.Line2D([0], [0], marker="o", color="w", markerfacecolor="#dc2626",
               markersize=6, lw=0, label=f"{N_DOORS} doors"),
    plt.Line2D([0], [0], marker="*", color="w", markerfacecolor="#22c55e",
               markersize=10, lw=0, label="robot spawn"),
], loc="lower right", fontsize=6.6, framealpha=0.92, handlelength=1.4,
   borderpad=0.4, labelspacing=0.3)

# Panel (b): expanded route
base(axB, "(b) Door-inspection route (TSP + Dijkstra)")
axB.add_collection(mc.LineCollection(edge_segs, colors=(0.13, 0.43, 0.92, 0.12),
                                     linewidths=0.6, zorder=1))
axB.plot(expand_px[:, 0], expand_px[:, 1], color="#7c3aed", lw=1.7, zorder=3)
axB.scatter(door_px[:, 0], door_px[:, 1], s=22, c="#dc2626", zorder=4,
            linewidths=0.5, edgecolors="white")
axB.scatter([spawn_px[0]], [spawn_px[1]], s=130, marker="*", c="#22c55e",
            zorder=5, linewidths=0.8, edgecolors="white")
axB.legend(handles=[
    plt.Line2D([0], [0], color="#7c3aed", lw=2.0,
               label=f"robot route ({len(expanded)} waypoints)"),
    plt.Line2D([0], [0], marker="o", color="w", markerfacecolor="#dc2626",
               markersize=6, lw=0, label=f"{N_DOORS} doors inspected"),
], loc="lower right", fontsize=6.6, framealpha=0.92, handlelength=1.4,
   borderpad=0.4, labelspacing=0.3)

fig.subplots_adjust(left=0.005, right=0.995, top=0.99, bottom=0.06, wspace=0.03)
for ext in ("pdf", "png"):
    fig.savefig(OUT / f"fig_navgraph.{ext}", bbox_inches="tight", dpi=300)
print("saved", OUT / "fig_navgraph.pdf",
      f"| start={start} doors={len(door_ids)} waypoints={len(expanded)}")
