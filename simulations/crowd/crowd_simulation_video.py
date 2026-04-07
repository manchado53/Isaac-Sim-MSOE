"""Crowd simulation video — third-person follow camera.

Pattern copied from simulations/environment/msoe_building_tour.py (gonzalo-camera branch):
  - add_reference_to_stage (not open_stage)
  - simulation_app.update() after world.reset() before camera.initialize()
  - 30 warm-up frames before recording
  - Camera pose updated BEFORE world.step() (critical for async pipeline)
  - Exponentially smoothed heading to prevent jittery camera

Output: videos/crowd_follow_<timestamp>.mp4
"""

import os
import sys
import json
import random
import heapq
from datetime import datetime

os.environ["OMNI_KIT_ACCEPT_EULA"] = "YES"
os.environ["PRIVACY_CONSENT"] = "Y"

from isaacsim import SimulationApp
simulation_app = SimulationApp({"headless": True, "width": 1280, "height": 720})

import math
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.backends.backend_agg import FigureCanvasAgg
from scipy.spatial.transform import Rotation as R
import isaacsim.core.utils.numpy.rotations as rot_utils
from isaacsim.core.api import World
from isaacsim.sensors.camera import Camera
from omni.isaac.core.objects import VisualCapsule
from pxr import UsdLux, UsdGeom, Sdf, Gf
import omni.usd

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "../..")))
from sim_utils.video_capture import VideoRecorder

# ── Config ────────────────────────────────────────────────────────────────────

PROJECT_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), "../.."))
USD_PATH    = os.path.join(PROJECT_DIR, "assets/environments/msoe_building/innovation_center_finalV1.usd")
GRAPH_PATH  = os.path.join(PROJECT_DIR, "assets/Graph/floor_4_nav_graph_world_xflipped.json")

NUM_AGENTS     = 30
SIM_DURATION_S = 20.0
AGENT_SPEED_MS = 1.2
AGENT_RADIUS   = 0.2
AGENT_HEIGHT   = 1.4    # ~1.8m total with 2*radius caps
STEP_SIZE      = 1.0 / 30.0
WARMUP_FRAMES  = 60     # let renderer stabilize before recording

# Follow-cam: behind and above the followed agent
FOLLOW_OFFSET  = np.array([-3.0, 0.0, 1.8])

# Heading smoothing — lower = smoother turns, more lag
# 0.04 ≈ 1 second ease at 30fps
HEADING_ALPHA  = 0.04

# ── Nav graph ─────────────────────────────────────────────────────────────────

def load_graph(path):
    with open(path) as f:
        data = json.load(f)
    z_m = data["z_height_cm"] / 100.0
    nodes = {}
    for n in data["nodes"]:
        nodes[n["id"]] = np.array([n["world_xy_cm"][0] / 100.0,
                                    n["world_xy_cm"][1] / 100.0, z_m])
    adj = {nid: [] for nid in nodes}
    for e in data["edges"]:
        a, b = e["from"], e["to"]
        dist = np.linalg.norm(nodes[a][:2] - nodes[b][:2])
        adj[a].append((b, dist))
        adj[b].append((a, dist))
    return nodes, adj


def dijkstra(adj, start, goal):
    dist = {start: 0.0}
    prev = {}
    pq = [(0.0, start)]
    while pq:
        d, u = heapq.heappop(pq)
        if u == goal:
            break
        if d > dist.get(u, float("inf")):
            continue
        for v, w in adj[u]:
            nd = d + w
            if nd < dist.get(v, float("inf")):
                dist[v] = nd
                prev[v] = u
                heapq.heappush(pq, (nd, v))
    path, cur = [], goal
    while cur in prev:
        path.append(cur)
        cur = prev[cur]
    path.append(start)
    path.reverse()
    return path if path[0] == start else []


# ── Agent ─────────────────────────────────────────────────────────────────────

class PedestrianAgent:
    def __init__(self, agent_id, nodes, adj, world):
        self.agent_id     = agent_id
        self.nodes        = nodes
        self.adj          = adj
        self.current_node = random.choice(list(nodes.keys()))
        self.pos          = nodes[self.current_node].copy().astype(float)
        self.heading      = np.array([0.0, 1.0])
        color = np.array([
            random.uniform(0.2, 0.95),
            random.uniform(0.2, 0.95),
            random.uniform(0.2, 0.95),
        ])
        # Capsule standing upright — center at waist height
        self.prim = world.scene.add(
            VisualCapsule(
                prim_path=f"/World/Agents/agent_{agent_id}",
                name=f"agent_{agent_id}",
                position=self.pos + np.array([0, 0, 0.9]),
                radius=AGENT_RADIUS,
                height=AGENT_HEIGHT,
                color=color,
            )
        )
        self.path = []
        self.path_idx = 0
        self._pick_new_destination()

    def _pick_new_destination(self):
        node_ids = list(self.nodes.keys())
        dest = random.choice(node_ids)
        while dest == self.current_node:
            dest = random.choice(node_ids)
        self.path     = dijkstra(self.adj, self.current_node, dest)
        self.path_idx = 1

    def step(self, dt):
        if not self.path or self.path_idx >= len(self.path):
            self._pick_new_destination()
            return
        target_pos = self.nodes[self.path[self.path_idx]]
        direction  = target_pos[:2] - self.pos[:2]
        dist       = np.linalg.norm(direction)
        if dist < 0.1:
            self.current_node = self.path[self.path_idx]
            self.path_idx += 1
            if self.path_idx >= len(self.path):
                self._pick_new_destination()
        else:
            unit         = direction / dist
            self.heading = unit
            self.pos[:2] += unit * AGENT_SPEED_MS * dt
            self.prim.set_world_pose(position=self.pos + np.array([0, 0, 0.9]))


class Minimap:
    """Pre-renders static nav graph once; updates only agent dots each frame."""

    def __init__(self, nodes, edges, size=280):
        self.size = size
        self.fig, self.ax = plt.subplots(figsize=(size/100, size/100), dpi=100)
        self.fig.patch.set_facecolor("#111111")
        self.ax.set_facecolor("#111111")

        nxs = [v[0] for v in nodes.values()]
        nys = [v[1] for v in nodes.values()]

        # Static: edges + nodes (drawn once)
        for src, dst in edges:
            self.ax.plot([nodes[src][0], nodes[dst][0]],
                         [nodes[src][1], nodes[dst][1]],
                         color="#444444", linewidth=0.8, zorder=1)
        self.ax.scatter(nxs, nys, c="#888888", s=6, zorder=2)

        self.ax.set_xlim(min(nxs) - 2, max(nxs) + 2)
        self.ax.set_ylim(min(nys) - 2, max(nys) + 2)
        self.ax.set_aspect("equal")
        self.ax.axis("off")
        self.fig.tight_layout(pad=0)

        # Dynamic: agent scatter objects (updated each frame, not recreated)
        self._agents_sc  = self.ax.scatter([], [], c="white",  s=10, zorder=3)
        self._follow_sc  = self.ax.scatter([], [], c="#FFD700", s=30, zorder=4,
                                            edgecolors="white", linewidths=0.5)
        self._canvas = FigureCanvasAgg(self.fig)

    def render(self, agents, follow_agent):
        """Update agent positions and return RGB numpy array."""
        xs = np.array([a.pos[0] for a in agents])
        ys = np.array([a.pos[1] for a in agents])
        self._agents_sc.set_offsets(np.c_[xs, ys])
        self._follow_sc.set_offsets([[follow_agent.pos[0], follow_agent.pos[1]]])
        self._canvas.draw()
        buf = self._canvas.buffer_rgba()
        img = np.frombuffer(buf, dtype=np.uint8).reshape(self.size, self.size, 4)
        return img[:, :, :3].copy()

    def close(self):
        plt.close(self.fig)


def composite_minimap(frame_rgba, minimap_rgb, margin=10):
    """Paste minimap into bottom-right corner of frame (in-place)."""
    if frame_rgba is None:
        return frame_rgba
    h, w = frame_rgba.shape[:2]
    mh, mw = minimap_rgb.shape[:2]
    y0 = h - mh - margin
    x0 = w - mw - margin

    # Semi-transparent dark border
    frame_rgba[y0-2:y0+mh+2, x0-2:x0+mw+2, :3] = 20
    frame_rgba[y0:y0+mh, x0:x0+mw, :3] = minimap_rgb
    return frame_rgba


def heading_to_quat(heading_2d):
    """2D unit vector → Isaac Sim [w, x, y, z] quaternion (yaw only)."""
    yaw = float(np.arctan2(heading_2d[1], heading_2d[0]))
    q   = R.from_euler("z", yaw).as_quat()  # scipy [x,y,z,w]
    return np.array([q[3], q[0], q[1], q[2]])


# ── Main ──────────────────────────────────────────────────────────────────────

def main():
    timestamp  = datetime.now().strftime("%Y%m%d_%H%M%S")
    video_name = f"crowd_follow_{timestamp}"

    print("[INFO] Loading nav graph...", flush=True)
    nodes, adj = load_graph(GRAPH_PATH)
    with open(GRAPH_PATH) as f:
        _gdata = __import__("json").load(f)
    edges = [(_e["from"], _e["to"]) for _e in _gdata["edges"]]
    print(f"[INFO] Loaded {len(nodes)} nodes, {len(edges)} edges", flush=True)

    # ── World + building ──────────────────────────────────────────────────────
    # open_stage shows the building geometry (confirmed working in fbd2b58)
    print("[INFO] Opening USD stage...", flush=True)
    omni.usd.get_context().open_stage(USD_PATH)
    world = World(stage_units_in_meters=1.0)

    # ── Lighting ──────────────────────────────────────────────────────────────
    # DistantLight + DomeLight for global fill (50000/10000 confirmed visible)
    # RectLights at floor 4 ceiling height (z≈31m) for interior illumination
    # since the roof blocks DistantLight inside the building
    # DomeLight is omnidirectional — penetrates through geometry unlike DistantLight
    # which gets blocked by the roof. No RectLights needed.
    stage = omni.usd.get_context().get_stage()
    dome = UsdLux.DomeLight.Define(stage, Sdf.Path("/World/DomeLight"))
    dome.CreateIntensityAttr(100000)
    # SphereLight at agent level for additional fill inside enclosed spaces
    sphere = UsdLux.SphereLight.Define(stage, Sdf.Path("/World/FillLight"))
    sphere.CreateIntensityAttr(500000)
    sphere.CreateRadiusAttr(1.0)
    UsdGeom.XformCommonAPI(sphere).SetTranslate(Gf.Vec3d(-7.0, 67.0, 29.5))
    print("[INFO] Lighting added (DomeLight 100k + SphereLight 500k)", flush=True)

    # ── Agents ────────────────────────────────────────────────────────────────
    print(f"[INFO] Spawning {NUM_AGENTS} agents...", flush=True)
    agents = [PedestrianAgent(i, nodes, adj, world) for i in range(NUM_AGENTS)]
    follow_agent   = agents[0]
    smooth_heading = follow_agent.heading.copy()
    minimap = Minimap(nodes, edges, size=280)
    print("[INFO] Minimap initialized", flush=True)

    # ── Camera ────────────────────────────────────────────────────────────────
    init_pos = follow_agent.pos + np.array([-3.0, 0.0, 1.8])
    follow_cam = Camera(
        prim_path="/World/FollowCam",
        position=init_pos,
        orientation=rot_utils.euler_angles_to_quats(np.array([0, 0, 0]), degrees=True),
        frequency=30,
        resolution=(1280, 720),
    )

    # ── Reset + init — exact order from msoe_building_tour.py ────────────────
    world.reset()
    simulation_app.update()   # flush stage after reset (critical)
    follow_cam.initialize()
    print("[INFO] Camera initialized", flush=True)

    # ── Warm-up frames — let renderer fully initialize before capturing ────────
    print(f"[INFO] Warming up ({WARMUP_FRAMES} frames)...", flush=True)
    for _ in range(WARMUP_FRAMES):
        world.step(render=True)

    # Diagnose first frame
    test_rgba = follow_cam.get_rgba()
    if test_rgba is not None:
        print(f"[INFO] Warm-up check — RGBA shape={test_rgba.shape} mean={test_rgba.mean():.1f}", flush=True)
    else:
        print("[WARN] Warm-up check — get_rgba() returned None", flush=True)

    total_steps = int(SIM_DURATION_S / STEP_SIZE)
    print(f"[INFO] Recording {total_steps} steps ({SIM_DURATION_S:.0f}s)...", flush=True)

    frame = 0
    with VideoRecorder(video_name, fps=30) as recorder:
        while frame < total_steps:

            # ── 1. Move agents ────────────────────────────────────────────────
            for agent in agents:
                agent.step(STEP_SIZE)

            # ── 2. Smooth heading + update camera BEFORE world.step ───────────
            raw = follow_agent.heading
            smooth_heading = HEADING_ALPHA * raw + (1 - HEADING_ALPHA) * smooth_heading
            n = np.linalg.norm(smooth_heading)
            if n > 1e-6:
                smooth_heading /= n

            yaw = float(np.arctan2(smooth_heading[1], smooth_heading[0]))
            rot = R.from_euler("z", yaw)
            cam_pos = follow_agent.pos + rot.apply(FOLLOW_OFFSET)

            follow_cam.set_world_pose(
                position    = cam_pos,
                orientation = heading_to_quat(smooth_heading),
            )

            # ── 3. Render ─────────────────────────────────────────────────────
            world.step(render=True)

            # ── 4. Capture + minimap overlay ──────────────────────────────────
            rgba = follow_cam.get_rgba()
            if rgba is not None:
                rgba = rgba.copy()
                composite_minimap(rgba, minimap.render(agents, follow_agent))
                recorder.capture_rgba(rgba)

            if frame % 100 == 0:
                mean_px = rgba.mean() if rgba is not None else -1
                print(f"[INFO] Frame {frame}/{total_steps}  px_mean={mean_px:.1f}", flush=True)

            frame += 1

    minimap.close()
    print(f"[INFO] Done. Video: videos/{video_name}.mp4", flush=True)
    simulation_app.close()


if __name__ == "__main__":
    main()
