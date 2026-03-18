"""
visualize_routes.py

Generates PNG visualizations of the navigation graph and solved task routes,
overlaid on the floor plan image.

Outputs:
  navigation/viz_graph.png           — full graph (all nodes + edges)
  navigation/viz_tasks_easy.png      — all 6 easy routes
  navigation/viz_tasks_medium.png    — all 6 medium routes
  navigation/viz_tasks_hard.png      — all 6 hard routes
  navigation/viz_task_<id>.png       — one PNG per task (18 total)

Usage:
  python3 visualize_routes.py
"""

import json
import os
import sys

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from matplotlib.lines import Line2D
from PIL import Image
import numpy as np

REPO_ROOT   = os.path.dirname(__file__)
FLOOR_IMG   = os.path.join(REPO_ROOT, "floor_plans_resized", "floor1", "floor1.jpg")
NODES_FILE  = os.path.join(REPO_ROOT, "floor_plans_resized", "floor1", "floor1_nodes.json")
ROUTES_FILE = os.path.join(REPO_ROOT, "navigation", "routes.json")
OUT_DIR     = os.path.join(REPO_ROOT, "navigation")

# ── colour scheme ──────────────────────────────────────────────────────────────
NODE_COLORS = {
    "room":         "#4A90D9",   # blue
    "hallway":      "#CCCCCC",   # light grey
    "intersection": "#F5A623",   # amber
    "stairwell":    "#E74C3C",   # red
    "elevator":     "#9B59B6",   # purple
    "entrance":     "#2ECC71",   # green
}
ROUTE_COLORS = [
    "#FF4136", "#FF851B", "#FFDC00", "#2ECC40", "#0074D9",
    "#B10DC9", "#01FF70", "#F012BE", "#3D9970", "#7FDBFF",
    "#85144b", "#39CCCC", "#001f3f", "#FF6F00", "#7B68EE",
    "#DC143C", "#00FA9A", "#FF69B4",
]


def load_graph():
    with open(NODES_FILE) as f:
        data = json.load(f)
    nodes = {n["id"]: n for n in data["nodes"]}
    edges = data["edges"]
    return nodes, edges


def load_routes():
    with open(ROUTES_FILE) as f:
        data = json.load(f)
    return data["tasks"]


def make_fig(title, figsize=(14, 12)):
    img = Image.open(FLOOR_IMG)
    fig, ax = plt.subplots(figsize=figsize)
    ax.imshow(img, origin="upper")
    ax.set_title(title, fontsize=13, fontweight="bold", pad=10)
    ax.axis("off")
    return fig, ax


def draw_graph_base(ax, nodes, edges, alpha_nodes=0.85, alpha_edges=0.5,
                    node_size=40, edge_lw=0.8):
    """Draw all nodes and edges of the navigation graph."""
    # edges first (under nodes)
    for e in edges:
        n0, n1 = nodes.get(e[0]), nodes.get(e[1])
        if n0 and n1:
            ax.plot([n0["x"], n1["x"]], [n0["y"], n1["y"]],
                    color="#888888", lw=edge_lw, alpha=alpha_edges, zorder=2)

    # nodes
    for nid, n in nodes.items():
        c = NODE_COLORS.get(n["type"], "#FFFFFF")
        ax.scatter(n["x"], n["y"], s=node_size, color=c,
                   edgecolors="white", linewidths=0.5,
                   alpha=alpha_nodes, zorder=3)


def draw_route(ax, waypoints, nodes, color, lw=3.0, ms=10):
    """Draw a solved route as a coloured line with start/goal markers."""
    xs = [nodes[w]["x"] for w in waypoints if w in nodes]
    ys = [nodes[w]["y"] for w in waypoints if w in nodes]
    if len(xs) < 2:
        return
    ax.plot(xs, ys, color=color, lw=lw, alpha=0.85, zorder=5)
    # start: circle
    ax.scatter(xs[0], ys[0], s=ms * 14, color=color,
               edgecolors="white", linewidths=1.5, zorder=7, marker="o")
    # goal: star
    ax.scatter(xs[-1], ys[-1], s=ms * 20, color=color,
               edgecolors="white", linewidths=1.5, zorder=7, marker="*")


def node_type_legend():
    return [
        mpatches.Patch(color=v, label=k.capitalize())
        for k, v in NODE_COLORS.items()
        if k not in ("stairwell", "elevator", "entrance")
    ]


# ── 1. Full graph overview ─────────────────────────────────────────────────────
def render_graph_overview(nodes, edges):
    fig, ax = make_fig("Floor 1 — Navigation Graph (all nodes & edges)")
    draw_graph_base(ax, nodes, edges, node_size=50, edge_lw=1.0,
                    alpha_nodes=1.0, alpha_edges=0.7)
    ax.legend(handles=node_type_legend(), loc="lower right",
              fontsize=9, framealpha=0.8)
    out = os.path.join(OUT_DIR, "viz_graph.png")
    fig.savefig(out, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"  Saved: {out}")


# ── 2. All routes by difficulty (one image per difficulty) ─────────────────────
def render_difficulty_group(tasks, nodes, edges, difficulty, color_list):
    group = [t for t in tasks if t["difficulty"] == difficulty and t["waypoints"]]
    if not group:
        return
    title = f"Floor 1 — {difficulty.capitalize()} Routes ({len(group)} tasks)"
    fig, ax = make_fig(title)
    draw_graph_base(ax, nodes, edges, alpha_nodes=0.5, alpha_edges=0.35,
                    node_size=25, edge_lw=0.6)

    legend_handles = []
    for i, task in enumerate(group):
        c = color_list[i % len(color_list)]
        draw_route(ax, task["waypoints"], nodes, color=c, lw=2.8)
        legend_handles.append(Line2D([0], [0], color=c, lw=2.5,
            label=f"{task['id']}: {task['start_name']} → {task['goal_name']}"))

    ax.legend(handles=legend_handles, loc="lower right",
              fontsize=7.5, framealpha=0.88, ncol=1)

    out = os.path.join(OUT_DIR, f"viz_tasks_{difficulty}.png")
    fig.savefig(out, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"  Saved: {out}")


# ── 3. Individual task images ──────────────────────────────────────────────────
def render_individual_tasks(tasks, nodes, edges):
    for task in tasks:
        if not task["waypoints"]:
            continue
        title = (f"{task['id']} ({task['difficulty']}) — "
                 f"{task['start_name']} → {task['goal_name']}\n"
                 f"{len(task['waypoints'])} waypoints  |  "
                 f"distance = {task['total_distance']:.0f} px")
        fig, ax = make_fig(title, figsize=(12, 10))
        draw_graph_base(ax, nodes, edges, alpha_nodes=0.45, alpha_edges=0.3,
                        node_size=22, edge_lw=0.55)
        draw_route(ax, task["waypoints"], nodes, color="#FF4136", lw=3.5, ms=12)

        # Label each waypoint along the path
        for j, wid in enumerate(task["waypoints"]):
            n = nodes.get(wid)
            if n is None:
                continue
            label = n["name"]
            if j == 0:
                label = f"START\n{label}"
            elif j == len(task["waypoints"]) - 1:
                label = f"GOAL\n{label}"
            ax.annotate(label,
                        xy=(n["x"], n["y"]),
                        xytext=(5, 5), textcoords="offset points",
                        fontsize=6.5, color="white",
                        bbox=dict(boxstyle="round,pad=0.2",
                                  fc="#333333", alpha=0.7),
                        zorder=8)

        out = os.path.join(OUT_DIR, f"viz_task_{task['id']}.png")
        fig.savefig(out, dpi=150, bbox_inches="tight")
        plt.close(fig)
        print(f"  Saved: {out}")


def main():
    os.makedirs(OUT_DIR, exist_ok=True)
    nodes, edges = load_graph()
    tasks = load_routes()

    print("Rendering graph overview...")
    render_graph_overview(nodes, edges)

    print("Rendering difficulty group images...")
    for diff in ("easy", "medium", "hard"):
        render_difficulty_group(tasks, nodes, edges, diff, ROUTE_COLORS)

    print("Rendering individual task images...")
    render_individual_tasks(tasks, nodes, edges)

    print(f"\nDone — all images saved to {OUT_DIR}/")


if __name__ == "__main__":
    main()
