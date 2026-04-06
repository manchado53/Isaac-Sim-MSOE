"""Crowd simulation analysis — heatmaps and traffic statistics.

Reads a crowd_log_<timestamp>.csv produced by crowd_simulation.py and outputs:
  1. A position heatmap (agent density across the floor)
  2. A node visit bar chart (most-trafficked nav nodes)
  3. A printed summary of key traffic statistics

Usage:
    python simulations/crowd/analyze_crowd.py                        # latest log
    python simulations/crowd/analyze_crowd.py logs/crowd_log_X.csv  # specific log

Output:
    simulations/crowd/figures/heatmap_<timestamp>.png
    simulations/crowd/figures/node_traffic_<timestamp>.png
"""

import os
import sys
import glob
import json
from collections import Counter
from datetime import datetime

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.colors as mcolors
from matplotlib.patches import Circle

# ── Paths ─────────────────────────────────────────────────────────────────────

PROJECT_DIR  = os.path.abspath(os.path.join(os.path.dirname(__file__), "../.."))
LOG_DIR      = os.path.join(PROJECT_DIR, "simulations/crowd/logs")
FIG_DIR      = os.path.join(PROJECT_DIR, "simulations/crowd/figures")
GRAPH_PATH   = os.path.join(PROJECT_DIR, "assets/Graph/floor_4_nav_graph_world_xflipped.json")

# ── Load data ──────────────────────────────────────────────────────────────────

def pick_log(argv) -> str:
    if len(argv) > 1 and os.path.isfile(argv[1]):
        return argv[1]
    logs = sorted(glob.glob(os.path.join(LOG_DIR, "crowd_log_*.csv")))
    if not logs:
        sys.exit("[ERROR] No crowd logs found. Run crowd_simulation.py first.")
    return logs[-1]  # most recent


def load_nav_graph(path: str):
    with open(path) as f:
        data = json.load(f)
    z_m = data["z_height_cm"] / 100.0
    nodes = {}
    for n in data["nodes"]:
        x_m = n["world_xy_cm"][0] / 100.0
        y_m = n["world_xy_cm"][1] / 100.0
        nodes[n["id"]] = (x_m, y_m)
    edges = [(e["from"], e["to"]) for e in data["edges"]]
    return nodes, edges


# ── Analysis ───────────────────────────────────────────────────────────────────

def node_traffic(df: pd.DataFrame, nodes: dict) -> pd.DataFrame:
    """Count how many log rows each node appears in (proxy for dwell time)."""
    counts = Counter(df["node"])
    rows = []
    for nid, (x, y) in nodes.items():
        rows.append({"node": nid, "x_m": x, "y_m": y, "visits": counts.get(nid, 0)})
    return pd.DataFrame(rows).sort_values("visits", ascending=False)


def print_stats(df: pd.DataFrame, traffic: pd.DataFrame, log_path: str):
    n_agents  = df["agent_id"].nunique()
    n_steps   = df["step"].nunique()
    duration  = df["time_s"].max()

    print("=" * 60)
    print("  CROWD SIMULATION SUMMARY")
    print(f"  Log file : {os.path.basename(log_path)}")
    print(f"  Agents   : {n_agents}")
    print(f"  Duration : {duration:.0f} s  ({n_steps} logged steps)")
    print("=" * 60)

    print("\nTop 10 most-visited nodes:")
    top = traffic.head(10)
    for _, row in top.iterrows():
        bar = "█" * int(row["visits"] / traffic["visits"].max() * 30)
        print(f"  {row['node']:20s} {int(row['visits']):6d}  {bar}")

    # Spatial density — divide floor into a coarse grid and find hotspots
    x_bins = np.linspace(df["x_m"].min(), df["x_m"].max(), 10)
    y_bins = np.linspace(df["y_m"].min(), df["y_m"].max(), 10)
    H, _, _ = np.histogram2d(df["x_m"], df["y_m"], bins=[x_bins, y_bins])
    peak_cell = np.unravel_index(H.argmax(), H.shape)
    peak_x = (x_bins[peak_cell[0]] + x_bins[peak_cell[0]+1]) / 2
    peak_y = (y_bins[peak_cell[1]] + y_bins[peak_cell[1]+1]) / 2
    print(f"\nDensest floor zone (center): x={peak_x:.1f}m, y={peak_y:.1f}m")
    print(f"  → Good location for signage / poster placement")
    print()


# ── Figures ────────────────────────────────────────────────────────────────────

def plot_heatmap(df: pd.DataFrame, nodes: dict, edges: list, out_path: str):
    fig, ax = plt.subplots(figsize=(12, 9))

    # 2-D histogram as heatmap
    x_vals = df["x_m"].values
    y_vals = df["y_m"].values
    h = ax.hist2d(
        x_vals, y_vals,
        bins=80,
        cmap="hot",
        norm=mcolors.PowerNorm(gamma=0.4),   # compress extreme outliers
    )
    plt.colorbar(h[3], ax=ax, label="Agent density (log counts)")

    # Overlay nav graph edges
    for src, dst in edges:
        if src in nodes and dst in nodes:
            xs = [nodes[src][0], nodes[dst][0]]
            ys = [nodes[src][1], nodes[dst][1]]
            ax.plot(xs, ys, color="cyan", alpha=0.3, linewidth=0.8, zorder=2)

    # Overlay nodes
    node_xs = [v[0] for v in nodes.values()]
    node_ys = [v[1] for v in nodes.values()]
    ax.scatter(node_xs, node_ys, c="white", s=15, zorder=3, label="Nav nodes")

    ax.set_xlabel("X (m)")
    ax.set_ylabel("Y (m)")
    ax.set_title("Floor 4 — Agent Density Heatmap")
    ax.legend(loc="upper right", fontsize=8)
    ax.set_aspect("equal")

    fig.tight_layout()
    fig.savefig(out_path, dpi=150)
    plt.close(fig)
    print(f"[INFO] Heatmap saved to: {out_path}")


def plot_node_traffic(traffic: pd.DataFrame, nodes: dict, edges: list, out_path: str):
    """Bubble map — node size/color proportional to visit count."""
    fig, axes = plt.subplots(1, 2, figsize=(16, 7))

    # ── Left: spatial bubble map ──────────────────────────────────────────────
    ax = axes[0]
    # Draw edges first
    for src, dst in edges:
        if src in nodes and dst in nodes:
            xs = [nodes[src][0], nodes[dst][0]]
            ys = [nodes[src][1], nodes[dst][1]]
            ax.plot(xs, ys, color="lightgray", linewidth=0.8, zorder=1)

    max_v = max(traffic["visits"].max(), 1)
    sc = ax.scatter(
        traffic["x_m"], traffic["y_m"],
        s=(traffic["visits"] / max_v * 400 + 20),
        c=traffic["visits"],
        cmap="YlOrRd",
        alpha=0.85,
        zorder=2,
        edgecolors="black",
        linewidths=0.4,
    )
    plt.colorbar(sc, ax=ax, label="Visit count")

    # Label top-5 nodes
    for _, row in traffic.head(5).iterrows():
        ax.annotate(
            row["node"],
            (row["x_m"], row["y_m"]),
            fontsize=6,
            ha="center",
            va="bottom",
            color="black",
            zorder=3,
        )

    ax.set_xlabel("X (m)")
    ax.set_ylabel("Y (m)")
    ax.set_title("Node Traffic Map (bubble = visit count)")
    ax.set_aspect("equal")

    # ── Right: top-20 bar chart ───────────────────────────────────────────────
    ax2 = axes[1]
    top20 = traffic.head(20)
    colors = plt.cm.YlOrRd(top20["visits"] / max_v)
    ax2.barh(top20["node"][::-1], top20["visits"][::-1], color=colors[::-1])
    ax2.set_xlabel("Visit count")
    ax2.set_title("Top 20 Most-Visited Nodes")
    ax2.tick_params(axis="y", labelsize=8)

    fig.suptitle("Floor 4 — Navigation Node Traffic Analysis", fontsize=13, fontweight="bold")
    fig.tight_layout()
    fig.savefig(out_path, dpi=150)
    plt.close(fig)
    print(f"[INFO] Node traffic figure saved to: {out_path}")


# ── Entry point ────────────────────────────────────────────────────────────────

def main():
    os.makedirs(FIG_DIR, exist_ok=True)

    log_path = pick_log(sys.argv)
    print(f"[INFO] Analyzing: {log_path}")

    df = pd.read_csv(log_path)
    nodes, edges = load_nav_graph(GRAPH_PATH)

    traffic = node_traffic(df, nodes)
    print_stats(df, traffic, log_path)

    # Timestamp suffix from log filename (e.g. crowd_log_20250405_143022.csv)
    tag = os.path.basename(log_path).replace("crowd_log_", "").replace(".csv", "")

    plot_heatmap(df, nodes, edges,
                 os.path.join(FIG_DIR, f"heatmap_{tag}.png"))
    plot_node_traffic(traffic, nodes, edges,
                      os.path.join(FIG_DIR, f"node_traffic_{tag}.png"))

    print("[INFO] Analysis complete.")


if __name__ == "__main__":
    main()
