#!/usr/bin/env python3
"""
Floor 4 Map Tracker

Renders the Floor 4 navigation graph overlaid on the building floor plan,
marks door positions, and animates the robot trajectory as a red dot.

Outputs (written to <project_root>/data/):
  floor4_map_overview.png     — static map with nav graph + doors + all trajectories
  floor4_episode_N.mp4        — per-episode video of robot red-dot traversal

Run after nova_carter_door_collection.py has produced data/door_labels.npz:
  cd ~/Digital-twin-v2/isaac-sim-msoe
  python3 simulations/robot/floor4_map_tracker.py
  # OR via Isaac Sim Python if matplotlib not in system env:
  # ./python.sh simulations/robot/floor4_map_tracker.py

No Isaac Sim required — pure matplotlib/numpy.
"""

import os
import json
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import matplotlib.image as mpimg
from matplotlib.collections import LineCollection

# ---------------------------------------------------------------------------
# Paths
# ---------------------------------------------------------------------------

_script_dir  = os.path.dirname(os.path.abspath(__file__))
_project_dir = os.path.abspath(os.path.join(_script_dir, "..", ".."))

NAV_JSON  = os.path.join(_project_dir, "floor_4_nav_graph_manual.json")
FLOOR_PNG = os.path.join(_project_dir, "floor4_navigation-graph.png")
NPZ_PATH  = os.path.join(_project_dir, "data", "door_labels.npz")
OUT_DIR   = os.path.join(_project_dir, "data")
os.makedirs(OUT_DIR, exist_ok=True)

# ---------------------------------------------------------------------------
# Load nav graph
# ---------------------------------------------------------------------------

with open(NAV_JSON) as f:
    nav = json.load(f)

# Convert node world_xy_cm → meters
nodes = {n["id"]: np.array([n["world_xy_cm"][0] / 100.0,
                             n["world_xy_cm"][1] / 100.0])
         for n in nav["nodes"]}

cell_cm  = nav["cell_size_cm"]
xy_min   = [v / 100.0 for v in nav["xy_min"]]   # meters
xy_max   = [v / 100.0 for v in nav["xy_max"]]

# Edges as (x0,y0)→(x1,y1) pairs
edge_segs = []
for e in nav["edges"]:
    p0 = nodes[e["from"]]
    p1 = nodes[e["to"]]
    edge_segs.append([p0, p1])

# ---------------------------------------------------------------------------
# Coordinate mapping: world meters → image pixels
#
# The floor plan PNG (1125 × 696 px) was generated from the occupancy grid.
# grid_rc[1] (col) corresponds to world X, grid_rc[0] (row) corresponds to
# world Y.  Row 0 = southern-most point (lowest Y), so in image space the
# image is stored with south at pixel-row 0. We flip Y when plotting so that
# north is up on-screen.
#
# Verified: node world_xy_cm[0]/cell_size = grid_col, [1]/cell_size = grid_row
#   node_36 grid_col = (-1877.1 - xy_min_cm[0]) / 3 → col index
#   → pixel_x = col / total_cols * img_width
#   → pixel_y = img_height - row / total_rows * img_height  (flip for north-up)
#
# In matplotlib we bypass the manual pixel math by using imshow(extent=...)
# which maps world coords directly to axis space so we can plot in meters.
# ---------------------------------------------------------------------------

img = mpimg.imread(FLOOR_PNG)
img_h, img_w = img.shape[:2]

# The imshow extent [left, right, bottom, top] in world coords (meters).
# origin='lower' → image row 0 is at the bottom → south at bottom = correct.
# If nodes appear misaligned, adjust _Y_FLIP below.
_Y_FLIP = True   # True = image row 0 is at south (normal map convention)

extent = [xy_min[0], xy_max[0], xy_min[1], xy_max[1]]  # [W, E, S, N] in meters

def world_to_ax(x_m, y_m):
    """Return (x, y) directly — matplotlib axes are already in world meters."""
    return x_m, y_m


# ---------------------------------------------------------------------------
# Load robot trajectory + door positions from npz
# ---------------------------------------------------------------------------

robot_traj  = None  # (N_episodes, MAX_FRAMES, 2) float32, NaN where not playing
door_pos_m  = None  # (N_doors, 2) float32

if os.path.isfile(NPZ_PATH):
    data = np.load(NPZ_PATH, allow_pickle=False)
    robot_traj = data["robot_traj"] if "robot_traj" in data else None
    door_pos_m = data["door_pos_m"] if "door_pos_m" in data else None
    print(f"Loaded {NPZ_PATH}")
    if robot_traj is not None:
        print(f"  robot_traj shape: {robot_traj.shape}")
else:
    print(f"NOTE: {NPZ_PATH} not found — will render map without trajectories.")

# Fallback: use hard-coded door positions from diagnostic job 237551
if door_pos_m is None:
    door_pos_m = np.array([
        [-5.85,  54.21],   # Doors_1913690
        [-41.41, 85.81],   # Doors_4190126
        [-48.39, 78.80],   # Doors_12112322
        [-48.39, 79.76],   # Doors_12112444
        [-41.41, 82.60],   # Doors_13185454
        [-20.48, 80.15],   # Doors_13606713  ← closest to robot path
        [-5.88,  55.17],   # Doors_13935342
    ], dtype=np.float32)

TRIGGER_RADIUS = 8.0  # meters — matches nova_carter_door_collection.py

# Door names for the legend
DOOR_LABELS = [
    "Doors_1913690",
    "Doors_4190126",
    "Doors_12112322",
    "Doors_12112444",
    "Doors_13185454",
    "Doors_13606713 ★",   # closest to robot
    "Doors_13935342",
]

# ---------------------------------------------------------------------------
# Helper: build one axes with floor plan + nav graph + doors
# ---------------------------------------------------------------------------

def make_base_ax(fig, ax):
    """Draw floor plan, nav graph, and door markers onto ax."""
    # Floor plan background
    if _Y_FLIP:
        ax.imshow(img, extent=extent, origin="lower", aspect="auto", zorder=0,
                  alpha=0.6)
    else:
        ax.imshow(img, extent=extent, origin="upper", aspect="auto", zorder=0,
                  alpha=0.6)

    # Nav graph edges
    lc = LineCollection(edge_segs, colors="royalblue", linewidths=0.8,
                        alpha=0.6, zorder=1)
    ax.add_collection(lc)

    # Nav graph nodes
    node_xy = np.array(list(nodes.values()))
    ax.scatter(node_xy[:, 0], node_xy[:, 1], s=12, c="royalblue",
               zorder=2, label="Nav nodes")

    # Door positions + trigger-radius circles
    for i, dpos in enumerate(door_pos_m):
        is_target = ("★" in DOOR_LABELS[i])
        color = "darkorange" if is_target else "gold"
        ax.scatter(dpos[0], dpos[1], s=80 if is_target else 40,
                   c=color, marker="X", zorder=4,
                   label=DOOR_LABELS[i] if i < len(DOOR_LABELS) else f"Door {i}")
        circle = plt.Circle((dpos[0], dpos[1]), TRIGGER_RADIUS,
                             color=color, fill=False, linestyle="--",
                             linewidth=1.0, alpha=0.5, zorder=3)
        ax.add_patch(circle)

    # Spawn position
    spawn = np.array([-20.93, 73.12])  # node_41
    ax.scatter(*spawn, s=120, c="limegreen", marker="^", zorder=5,
               label="Robot spawn (node_41)")

    ax.set_xlim(xy_min[0], xy_max[0])
    ax.set_ylim(xy_min[1], xy_max[1])
    ax.set_xlabel("USD X (m)")
    ax.set_ylabel("USD Y (m)")
    ax.set_aspect("equal")


# ---------------------------------------------------------------------------
# Figure 1: Static overview — all episodes overlaid
# ---------------------------------------------------------------------------

fig_ov, ax_ov = plt.subplots(figsize=(14, 12))
make_base_ax(fig_ov, ax_ov)
ax_ov.set_title("Floor 4 — Robot Trajectories Overview", fontsize=14)

EPISODE_COLORS = ["crimson", "deepskyblue", "limegreen", "darkorange", "purple"]

if robot_traj is not None:
    for ep, traj in enumerate(robot_traj):
        color = EPISODE_COLORS[ep % len(EPISODE_COLORS)]
        valid = ~np.isnan(traj[:, 0])
        xy = traj[valid]
        if len(xy) > 1:
            ax_ov.plot(xy[:, 0], xy[:, 1], color=color, linewidth=1.2,
                       alpha=0.7, label=f"Episode {ep+1}")
        if len(xy) > 0:
            # Final position marker
            ax_ov.scatter(xy[-1, 0], xy[-1, 1], s=60, c=color,
                          marker="o", zorder=6)

ax_ov.legend(loc="upper right", fontsize=7, framealpha=0.8)
overview_path = os.path.join(OUT_DIR, "floor4_map_overview.png")
fig_ov.savefig(overview_path, dpi=150, bbox_inches="tight")
plt.close(fig_ov)
print(f"Saved: {overview_path}")

# ---------------------------------------------------------------------------
# Figure 2: Per-episode animated MP4
# Uses imageio if available; falls back to saving PNG frames.
# ---------------------------------------------------------------------------

if robot_traj is not None:
    try:
        import imageio
        _have_imageio = True
    except ImportError:
        _have_imageio = False
        print("imageio not available — saving PNG frames only (no MP4).")

    STEP_STRIDE = 10   # render every Nth step (keeps video manageable)

    for ep, traj in enumerate(robot_traj):
        valid = ~np.isnan(traj[:, 0])
        xy_valid = traj[valid]
        if len(xy_valid) == 0:
            continue

        frames_out = []
        steps_to_render = list(range(0, len(xy_valid), STEP_STRIDE))
        if steps_to_render[-1] != len(xy_valid) - 1:
            steps_to_render.append(len(xy_valid) - 1)

        print(f"Rendering episode {ep+1} ({len(steps_to_render)} frames)...")

        for si in steps_to_render:
            fig_ep, ax_ep = plt.subplots(figsize=(11, 9))
            make_base_ax(fig_ep, ax_ep)
            ax_ep.set_title(f"Floor 4 — Episode {ep+1}  step {si*STEP_STRIDE}", fontsize=12)

            # Trail (path so far)
            if si > 0:
                ax_ep.plot(xy_valid[:si, 0], xy_valid[:si, 1],
                           color="crimson", linewidth=1.5, alpha=0.5, zorder=5)

            # Red dot (current position)
            ax_ep.scatter(xy_valid[si, 0], xy_valid[si, 1],
                          s=120, c="crimson", marker="o", zorder=7,
                          label="Robot")

            ax_ep.legend(loc="upper right", fontsize=7, framealpha=0.8)

            # Render to numpy RGBA array
            fig_ep.canvas.draw()
            buf = np.frombuffer(fig_ep.canvas.tostring_rgb(), dtype=np.uint8)
            buf = buf.reshape(fig_ep.canvas.get_width_height()[::-1] + (3,))
            frames_out.append(buf)
            plt.close(fig_ep)

        if _have_imageio:
            # Try MP4 first; fall back to GIF if imageio-ffmpeg is not installed.
            vid_path = os.path.join(OUT_DIR, f"floor4_episode_{ep+1}.mp4")
            try:
                imageio.mimsave(vid_path, frames_out, fps=10)
                print(f"  Saved: {vid_path}")
            except (ImportError, Exception) as _e:
                gif_path = vid_path.replace(".mp4", ".gif")
                print(f"  MP4 failed ({_e.__class__.__name__}: {_e})")
                print(f"  Falling back to GIF: {gif_path}")
                print(f"  (Install imageio-ffmpeg for MP4: pip install imageio-ffmpeg)")
                # GIF: use reduced frame set to keep file size manageable
                gif_frames = frames_out[::3]  # every 3rd frame
                imageio.mimsave(gif_path, gif_frames, duration=0.3)
                print(f"  Saved: {gif_path} ({len(gif_frames)} frames)")
        else:
            # Save first and last frames as PNG
            for fname, frame in [("first", frames_out[0]), ("last", frames_out[-1])]:
                png_path = os.path.join(OUT_DIR, f"floor4_episode_{ep+1}_{fname}.png")
                plt.imsave(png_path, frame)
                print(f"  Saved: {png_path}")

print("\nDone. Check data/ for output files.")
