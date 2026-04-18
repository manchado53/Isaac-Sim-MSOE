#!/usr/bin/env python3
"""
Plot a floor nav graph on top of its occupancy map with labeled world-frame axes.

Useful for visually picking a spawn point: read the (x, y) coordinates off the
axes, then look up the nearest node ID to use.

Usage:
    python3 floorplan_vision/plot_graph_with_axes.py --floor floor_4
    python3 floorplan_vision/plot_graph_with_axes.py --floor floor_1 --show-ids
"""

import argparse
import json
import os

import matplotlib.pyplot as plt
import numpy as np
from PIL import Image


REPO_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
MAP_DIR = os.path.join(REPO_DIR, "assets", "environments", "InnovationCenter", "maps")
GRAPH_DIR = os.path.join(REPO_DIR, "assets", "environments", "InnovationCenter", "navgraphs")
OUT_DIR = GRAPH_DIR


def plot_floor(floor_name: str, show_ids: bool = False):
    img_path = os.path.join(MAP_DIR, f"{floor_name}_occupancy.png")
    graph_path = os.path.join(GRAPH_DIR, f"{floor_name}_auto_navgraph.json")

    # Occupancy PNG is now stored in ROS map_server convention: col 0 = x_min,
    # row 0 = y_max. Both flips are applied at the generator source.
    img = np.array(Image.open(img_path).convert("L"))
    with open(graph_path) as f:
        graph = json.load(f)

    x_min, y_min = graph["xy_min"]
    x_max, y_max = graph["xy_max"]

    fig, ax = plt.subplots(figsize=(16, 10))

    # row 0 of the image = world y_max, y decreases with row. With
    # origin="upper" and extent=[x_min, x_max, y_min, y_max], matplotlib
    # puts row 0 at the top of the axes where y_max lives, giving a natural
    # "north up" view with no flip tricks.
    ax.imshow(
        img,
        cmap="gray",
        extent=[x_min, x_max, y_min, y_max],
        origin="upper",
    )

    # Plot edges
    node_by_id = {n["id"]: n for n in graph["nodes"]}
    for edge in graph["edges"]:
        a = node_by_id[edge["from"]]
        b = node_by_id[edge["to"]]
        ax.plot(
            [a["world_xy_m"][0], b["world_xy_m"][0]],
            [a["world_xy_m"][1], b["world_xy_m"][1]],
            color="cyan",
            linewidth=0.8,
            alpha=0.7,
        )

    # Plot nodes
    xs = [n["world_xy_m"][0] for n in graph["nodes"]]
    ys = [n["world_xy_m"][1] for n in graph["nodes"]]
    ax.scatter(xs, ys, c="red", s=12, zorder=3)

    if show_ids:
        for n in graph["nodes"]:
            ax.annotate(
                str(n["id"]),
                (n["world_xy_m"][0], n["world_xy_m"][1]),
                fontsize=5,
                color="blue",
                ha="center",
                va="center",
            )

    # Gridlines at 5m intervals
    x_ticks = np.arange(np.floor(x_min / 5) * 5, np.ceil(x_max / 5) * 5 + 1, 5)
    y_ticks = np.arange(np.floor(y_min / 5) * 5, np.ceil(y_max / 5) * 5 + 1, 5)
    ax.set_xticks(x_ticks)
    ax.set_yticks(y_ticks)
    ax.grid(True, which="both", color="lime", linewidth=0.4, alpha=0.5)

    ax.set_xlabel("World X (meters)")
    ax.set_ylabel("World Y (meters)")
    ax.set_title(
        f"{floor_name}: {graph['total_nodes']} nodes, {graph['total_edges']} edges\n"
        f"bounds: x=[{x_min:.1f}, {x_max:.1f}] y=[{y_min:.1f}, {y_max:.1f}]"
    )
    ax.set_aspect("equal")

    out_path = os.path.join(OUT_DIR, f"{floor_name}_graph_with_axes.png")
    fig.tight_layout()
    fig.savefig(out_path, dpi=150)
    print(f"Saved: {out_path}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--floor", default="floor_4",
                        choices=["floor_1", "floor_2", "floor_3", "floor_4"])
    parser.add_argument("--show-ids", action="store_true",
                        help="Label every node with its ID (dense)")
    args = parser.parse_args()
    plot_floor(args.floor, args.show_ids)
