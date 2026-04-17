"""
visualize_coverage.py

Generates a PNG of the full floor coverage route overlaid on the floor plan.

Reads:
  test_building/floor4_nodes_overlay.png                                — floor plan background
  test_building/test_nodes/Final_Data/floor_4_fullres_graph.json        — all nodes + edges
  navigation/full_floor_coverage/coverage.json                          — solved coverage route

Outputs:
  navigation/full_floor_coverage/coverage_map.png    — annotated floor plan

Color scheme:
  Reachable nodes  — colored by type (room=blue, hallway=grey, intersection=amber,
                     stairwell=red, elevator=purple, entrance=green)
  Unreachable nodes — magenta X marker
  Coverage route   — bright orange line
  Start node       — large green circle
  Graph edges      — faint grey lines (background reference)

Usage:
  python3 navigation/full_floor_coverage/visualize_coverage.py
"""

import json
import os

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from matplotlib.lines import Line2D
from PIL import Image
import numpy as np

REPO_ROOT     = os.path.join(os.path.dirname(__file__), "..", "..")
FLOOR_IMG     = os.path.join(REPO_ROOT, "test_building", "floor4_nodes_overlay.png")
NODES_FILE    = os.path.join(REPO_ROOT, "test_building", "test_nodes", "Final_Data", "floor_4_fullres_graph.json")
COVERAGE_FILE = os.path.join(os.path.dirname(__file__), "coverage.json")
OUT_FILE      = os.path.join(os.path.dirname(__file__), "coverage_map.png")

NODE_COLOR        = "#4A90D9"    # blue (all nodes)
UNREACHABLE_COLOR = "#FF00FF"    # magenta
ROUTE_COLOR       = "#FF8C00"    # dark orange
START_COLOR       = "#00FF00"    # bright green
EDGE_COLOR        = "#666666"


def main():
    # ── Load data ──────────────────────────────────────────────────────────────
    with open(NODES_FILE) as f:
        node_data = json.load(f)
    with open(COVERAGE_FILE) as f:
        coverage = json.load(f)

    nodes = {n["id"]: n for n in node_data["nodes"]}
    edges = node_data["edges"]

    unreachable_ids = {entry["id"] for entry in coverage["unreachable"]}
    full_path       = coverage["full_path"]
    start_node      = coverage["start_node"]

    def nx(n): return n["grid_col"]
    def ny(n): return n["grid_row"]

    # ── Set up figure ──────────────────────────────────────────────────────────
    img = Image.open(FLOOR_IMG)
    fig, ax = plt.subplots(figsize=(16, 18))
    ax.imshow(img, origin="upper")
    ax.set_title(
        f"Floor 4 — Full Coverage Route\n"
        f"{coverage['nodes_visited']} reachable nodes visited  |  "
        f"{coverage['unreachable_nodes']} unreachable  |  "
        f"total distance = {coverage['total_distance_px']:,.0f} px",
        fontsize=12, fontweight="bold", pad=10
    )
    ax.axis("off")

    # ── Draw graph edges (faint background) ────────────────────────────────────
    for e in edges:
        n0, n1 = nodes.get(e["from"]), nodes.get(e["to"])
        if n0 and n1:
            ax.plot([nx(n0), nx(n1)], [ny(n0), ny(n1)],
                    color=EDGE_COLOR, lw=0.6, alpha=0.3, zorder=2)

    # ── Draw coverage route ────────────────────────────────────────────────────
    rx = [nx(nodes[w]) for w in full_path if w in nodes]
    ry = [ny(nodes[w]) for w in full_path if w in nodes]
    ax.plot(rx, ry, color=ROUTE_COLOR, lw=2.0, alpha=0.75, zorder=4,
            label="Coverage route")

    # ── Draw reachable nodes ───────────────────────────────────────────────────
    for nid, n in nodes.items():
        if nid in unreachable_ids:
            continue
        ax.scatter(nx(n), ny(n), s=35, color=NODE_COLOR,
                   edgecolors="white", linewidths=0.5, alpha=0.9, zorder=5)

    # ── Draw unreachable nodes (magenta X) ────────────────────────────────────
    for nid in unreachable_ids:
        n = nodes.get(nid)
        if n:
            ax.scatter(nx(n), ny(n), s=60, color=UNREACHABLE_COLOR,
                       marker="x", linewidths=1.5, alpha=1.0, zorder=6)

    # ── Draw start node (large green circle) ──────────────────────────────────
    if start_node in nodes:
        sn = nodes[start_node]
        ax.scatter(nx(sn), ny(sn), s=200, color=START_COLOR,
                   edgecolors="white", linewidths=1.5, zorder=7, marker="o")
        ax.annotate(f"START\n{start_node}",
                    xy=(nx(sn), ny(sn)),
                    xytext=(8, 8), textcoords="offset points",
                    fontsize=7, color="white",
                    bbox=dict(boxstyle="round,pad=0.2", fc="#1a1a1a", alpha=0.8),
                    zorder=8)

    # ── Legend ─────────────────────────────────────────────────────────────────
    legend_handles = [
        mpatches.Patch(color=NODE_COLOR, label="Node"),
        mpatches.Patch(color=UNREACHABLE_COLOR, label="Unreachable (no edges)"),
        Line2D([0], [0], color=ROUTE_COLOR, lw=2.5, label="Coverage route"),
        Line2D([0], [0], color=START_COLOR, marker="o", markersize=8,
               linestyle="None", label=f"Start: {start_node}"),
    ]
    ax.legend(handles=legend_handles, loc="lower right",
              fontsize=8, framealpha=0.88, ncol=2)

    # ── Save ───────────────────────────────────────────────────────────────────
    fig.savefig(OUT_FILE, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"Saved: {OUT_FILE}")


if __name__ == "__main__":
    main()
