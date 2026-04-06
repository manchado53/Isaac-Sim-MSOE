"""Crowd simulation video — third-person follow camera.

Mirrors gonzalo's nova_carter_onboard_cam.py pattern exactly:
  1. Camera pose updated BEFORE world.step() (critical for async pipeline)
  2. world.is_playing() guard before get_rgba()
  3. Explicit lighting added to the stage

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

import numpy as np
from scipy.spatial.transform import Rotation as R
import isaacsim.core.utils.numpy.rotations as rot_utils
from isaacsim.core.api import World
from isaacsim.sensors.camera import Camera
from omni.isaac.core.objects import DynamicCapsule
from pxr import UsdLux, Sdf
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
AGENT_RADIUS   = 0.25
STEP_SIZE      = 1.0 / 30.0

# Follow-cam: 3m behind agent, 1.8m above — rotated by heading each frame
FOLLOW_OFFSET = np.array([-3.0, 0.0, 1.8])

# Camera heading smoothing — exponential moving average (lower = smoother, more lag)
HEADING_ALPHA = 0.06  # ~0.06 gives a smooth ~0.5s ease into turns

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
        self.heading      = np.array([0.0, 1.0])  # updated each step
        # Clothing-like colors — avoid pure white/black so agents are visible
        color = np.array([
            random.uniform(0.2, 0.9),
            random.uniform(0.2, 0.9),
            random.uniform(0.2, 0.9),
        ])
        # Tall capsule standing upright — looks like a person silhouette
        self.sphere = world.scene.add(
            DynamicCapsule(
                prim_path=f"/World/Agents/agent_{agent_id}",
                name=f"agent_{agent_id}",
                position=self.pos + np.array([0, 0, 0.9]),  # capsule center at waist height
                radius=0.2,
                height=1.4,   # ~1.8m total (height + 2*radius)
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
            self.sphere.set_world_pose(position=self.pos + np.array([0, 0, 0.9]))


def heading_to_quat(heading_2d):
    """2D heading → Isaac Sim [w, x, y, z] quaternion (yaw only)."""
    yaw = float(np.arctan2(heading_2d[1], heading_2d[0]))
    q   = R.from_euler("z", yaw).as_quat()   # scipy: [x, y, z, w]
    return np.array([q[3], q[0], q[1], q[2]])


# ── Main ──────────────────────────────────────────────────────────────────────

def main():
    timestamp  = datetime.now().strftime("%Y%m%d_%H%M%S")
    video_name = f"crowd_follow_{timestamp}"

    print("[INFO] Loading nav graph...")
    nodes, adj = load_graph(GRAPH_PATH)
    print(f"[INFO] Loaded {len(nodes)} nodes")

    # Open USD as root stage (same order that works in crowd_simulation.py)
    print("[INFO] Opening USD stage...")
    omni.usd.get_context().open_stage(USD_PATH)
    world = World(stage_units_in_meters=1.0)

    # Add lighting — building USD has no guaranteed interior lights
    stage = omni.usd.get_context().get_stage()
    # Bright interior lighting — DistantLight alone is too dim inside buildings
    dl = UsdLux.DistantLight.Define(stage, Sdf.Path("/World/DistantLight"))
    dl.CreateIntensityAttr(50000)
    dome = UsdLux.DomeLight.Define(stage, Sdf.Path("/World/DomeLight"))
    dome.CreateIntensityAttr(10000)
    print("[INFO] Lighting added")

    # Spawn agents
    print(f"[INFO] Spawning {NUM_AGENTS} agents...")
    agents = [PedestrianAgent(i, nodes, adj, world) for i in range(NUM_AGENTS)]
    follow_agent   = agents[0]
    smooth_heading = follow_agent.heading.copy()  # smoothed camera heading

    # Follow camera — initial position doesn't matter, updated every frame
    follow_cam = Camera(
        prim_path="/World/FollowCam",
        position=np.array([follow_agent.pos[0], follow_agent.pos[1] - 2.0,
                           follow_agent.pos[2] + 1.6]),
        orientation=rot_utils.euler_angles_to_quats(np.array([0, 0, 0]), degrees=True),
        frequency=30,
        resolution=(1280, 720),
    )

    world.reset()
    follow_cam.initialize()

    total_steps = int(SIM_DURATION_S / STEP_SIZE)
    print(f"[INFO] Recording {total_steps} steps ({SIM_DURATION_S:.0f}s) → {video_name}.mp4")

    frame = 0
    with VideoRecorder(video_name, fps=30) as recorder:
        while frame < total_steps:

            if world.is_playing():
                # ── 1. Move agents ────────────────────────────────────────
                for agent in agents:
                    agent.step(STEP_SIZE)

                # ── 2. Update camera BEFORE world.step (critical) ─────────
                pos = follow_agent.pos.copy()

                # Smooth heading — exponential moving average to avoid
                # jittery camera snapping at every nav node turn
                raw = follow_agent.heading
                smooth_heading = HEADING_ALPHA * raw + (1 - HEADING_ALPHA) * smooth_heading
                n = np.linalg.norm(smooth_heading)
                if n > 1e-6:
                    smooth_heading /= n

                yaw = float(np.arctan2(smooth_heading[1], smooth_heading[0]))
                rot = R.from_euler("z", yaw)
                rotated_offset = rot.apply(FOLLOW_OFFSET)

                follow_cam.set_world_pose(
                    position    = pos + rotated_offset,
                    orientation = heading_to_quat(smooth_heading),
                )

            # ── 3. Render step AFTER pose updates ─────────────────────────
            world.step(render=True)

            # ── 4. Capture (skip frame 0 like gonzalo) ────────────────────
            if world.is_playing() and frame > 0 and frame % 2 == 0:
                rgba = follow_cam.get_rgba()
                if rgba is not None:
                    recorder.capture_rgba(rgba)

            if frame % 100 == 0:
                print(f"[INFO] Frame {frame}/{total_steps}  playing={world.is_playing()}")

            frame += 1

    print("[INFO] Done.")
    simulation_app.close()


if __name__ == "__main__":
    main()
