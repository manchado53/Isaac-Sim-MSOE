"""Crowd simulation video — third-person follow camera.

Follows agent 0 from behind using a Camera sensor + capture_rgba(),
the same pattern used in simulations/robot/nova_carter_onboard_cam.py.

Usage:
    sbatch simulations/crowd/submit_crowd_video.sh

Output:
    videos/crowd_follow_<timestamp>.mp4
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
from isaacsim.core.utils.stage import add_reference_to_stage
from isaacsim.sensors.camera import Camera
from omni.isaac.core.objects import DynamicSphere

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

# Follow-cam offset from agent: 2m behind, 1.6m up
FOLLOW_OFFSET = np.array([-2.0, 0.0, 1.6])   # rotated each frame by agent heading

# ── Nav graph ─────────────────────────────────────────────────────────────────

def load_graph(path):
    with open(path) as f:
        data = json.load(f)
    z_m = data["z_height_cm"] / 100.0
    nodes = {}
    for n in data["nodes"]:
        nodes[n["id"]] = np.array([n["world_xy_cm"][0] / 100.0,
                                    n["world_xy_cm"][1] / 100.0,
                                    z_m])
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
        self.agent_id  = agent_id
        self.nodes     = nodes
        self.adj       = adj
        self.current_node = random.choice(list(nodes.keys()))
        self.pos       = nodes[self.current_node].copy().astype(float)
        self.heading   = np.array([1.0, 0.0])  # unit vector, updated each step
        color = np.array([random.random(), random.random(), random.random()])
        self.sphere = world.scene.add(
            DynamicSphere(
                prim_path=f"/World/Agents/agent_{agent_id}",
                name=f"agent_{agent_id}",
                position=self.pos,
                radius=AGENT_RADIUS,
                color=color,
            )
        )
        self.path     = []
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
            unit = direction / dist
            self.heading = unit
            self.pos[:2] += unit * AGENT_SPEED_MS * dt
            self.sphere.set_world_pose(position=self.pos)


def heading_to_quat(heading_2d):
    """Convert a 2D heading vector to an Isaac Sim [w,x,y,z] quaternion (yaw only)."""
    yaw = float(np.arctan2(heading_2d[1], heading_2d[0]))
    # scipy gives [x,y,z,w]; Isaac Sim wants [w,x,y,z]
    q = R.from_euler("z", yaw).as_quat()
    return np.array([q[3], q[0], q[1], q[2]])


# ── Main ──────────────────────────────────────────────────────────────────────

def main():
    timestamp  = datetime.now().strftime("%Y%m%d_%H%M%S")
    video_name = f"crowd_follow_{timestamp}"

    print("[INFO] Loading nav graph...")
    nodes, adj = load_graph(GRAPH_PATH)
    print(f"[INFO] Loaded {len(nodes)} nodes")

    # ── Load building USD via add_reference_to_stage (gonzalo's pattern) ──
    print("[INFO] Loading USD scene...")
    world = World(stage_units_in_meters=1.0)
    add_reference_to_stage(usd_path=USD_PATH, prim_path="/World/Building")

    # ── Spawn agents ──
    print(f"[INFO] Spawning {NUM_AGENTS} agents...")
    agents = [PedestrianAgent(i, nodes, adj, world) for i in range(NUM_AGENTS)]

    # ── Follow camera — initialized at agent 0's start position ──
    follow_agent = agents[0]
    init_orient  = heading_to_quat(follow_agent.heading)

    follow_cam = Camera(
        prim_path="/World/FollowCam",
        position=follow_agent.pos + np.array([-2.0, 0.0, 1.6]),
        orientation=init_orient,
        frequency=30,
        resolution=(1280, 720),
    )

    world.reset()
    follow_cam.initialize()

    total_steps = int(SIM_DURATION_S / STEP_SIZE)
    print(f"[INFO] Recording {total_steps} steps ({SIM_DURATION_S:.0f}s) → {video_name}.mp4")

    with VideoRecorder(video_name, fps=30) as recorder:
        for step in range(total_steps):
            # 1. Step physics + rendering
            world.step(render=True)

            # 2. Move agents
            for agent in agents:
                agent.step(STEP_SIZE)

            # 3. Update camera pose to follow agent 0
            pos     = follow_agent.pos.copy()
            heading = follow_agent.heading          # 2D unit vector
            orient  = heading_to_quat(heading)

            # Rotate the behind-and-up offset by agent heading
            rot            = R.from_euler("z", np.arctan2(heading[1], heading[0]))
            rotated_offset = rot.apply(FOLLOW_OFFSET)

            follow_cam.set_world_pose(
                position    = pos + rotated_offset,
                orientation = orient,
            )

            # 4. Capture frame
            rgba = follow_cam.get_rgba()
            if rgba is not None:
                recorder.capture_rgba(rgba)

            if step % 100 == 0:
                print(f"[INFO] Step {step}/{total_steps}")

    print("[INFO] Done.")
    simulation_app.close()


if __name__ == "__main__":
    main()
