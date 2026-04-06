"""Crowd simulation video — hallway camera view.

Places a hallway-level camera at node_21 looking north toward the busiest
corridor (node_37 → node_43) and records agents walking past.

Usage:
    sbatch simulations/crowd/submit_crowd_video.sh

Output:
    videos/crowd_hallway_<timestamp>.mp4
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
from omni.isaac.core import World
from omni.isaac.core.objects import DynamicSphere
import omni.usd

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "../..")))
from sim_utils.video_capture import VideoRecorder

# ── Config ────────────────────────────────────────────────────────────────────

PROJECT_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), "../.."))
USD_PATH    = os.path.join(PROJECT_DIR, "assets/environments/msoe_building/innovation_center_finalV1.usd")
GRAPH_PATH  = os.path.join(PROJECT_DIR, "assets/Graph/floor_4_nav_graph_world_xflipped.json")

NUM_AGENTS      = 30       # fewer agents keeps rendering snappy
SIM_DURATION_S  = 20.0     # 20s clip
AGENT_SPEED_MS  = 1.2
AGENT_RADIUS    = 0.25
STEP_SIZE       = 1.0 / 30.0

# Camera — node_21 position, looking north toward node_37/node_43
# node_21: x=-7.4, y=62.3   node_43: x=-7.2, y=73.4   floor z=28.25m
CAM_EYE    = np.array([-7.4,  60.0, 29.85])  # slightly south of node_21, eye height
CAM_TARGET = np.array([-7.3,  74.0, 29.0 ])  # looking toward node_43, slight downward tilt

# ── Nav graph ─────────────────────────────────────────────────────────────────

def load_graph(path):
    with open(path) as f:
        data = json.load(f)
    z_m = data["z_height_cm"] / 100.0
    nodes = {}
    for n in data["nodes"]:
        x_m = n["world_xy_cm"][0] / 100.0
        y_m = n["world_xy_cm"][1] / 100.0
        nodes[n["id"]] = np.array([x_m, y_m, z_m])
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
        self.agent_id = agent_id
        self.nodes = nodes
        self.adj = adj
        self.current_node = random.choice(list(nodes.keys()))
        self.pos = nodes[self.current_node].copy().astype(float)
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
        self.path = []
        self.path_idx = 0
        self._pick_new_destination()

    def _pick_new_destination(self):
        node_ids = list(self.nodes.keys())
        dest = random.choice(node_ids)
        while dest == self.current_node:
            dest = random.choice(node_ids)
        self.path = dijkstra(self.adj, self.current_node, dest)
        self.path_idx = 1

    def step(self, dt):
        if not self.path or self.path_idx >= len(self.path):
            self._pick_new_destination()
            return
        target_pos = self.nodes[self.path[self.path_idx]]
        direction = target_pos[:2] - self.pos[:2]
        dist = np.linalg.norm(direction)
        if dist < 0.1:
            self.current_node = self.path[self.path_idx]
            self.path_idx += 1
            if self.path_idx >= len(self.path):
                self._pick_new_destination()
        else:
            self.pos[:2] += (direction / dist) * AGENT_SPEED_MS * dt
            self.sphere.set_world_pose(position=self.pos)


# ── Main ──────────────────────────────────────────────────────────────────────

def main():
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    video_name = f"crowd_hallway_{timestamp}"

    print(f"[INFO] Loading nav graph...")
    nodes, adj = load_graph(GRAPH_PATH)
    print(f"[INFO] Loaded {len(nodes)} nodes")

    print(f"[INFO] Loading USD scene...")
    omni.usd.get_context().open_stage(USD_PATH)
    world = World(stage_units_in_meters=1.0)
    world.reset()

    # Position the viewport camera inside the hallway
    from isaacsim.core.utils.viewports import set_camera_view
    set_camera_view(
        eye=CAM_EYE,
        target=CAM_TARGET,
        camera_prim_path="/OmniverseKit_Persp",
    )
    print(f"[INFO] Camera: eye={CAM_EYE}, target={CAM_TARGET}")

    print(f"[INFO] Spawning {NUM_AGENTS} agents...")
    agents = [PedestrianAgent(i, nodes, adj, world) for i in range(NUM_AGENTS)]

    total_steps = int(SIM_DURATION_S / STEP_SIZE)
    print(f"[INFO] Recording {total_steps} steps ({SIM_DURATION_S:.0f}s)...")

    with VideoRecorder(video_name, fps=30) as recorder:
        for step in range(total_steps):
            world.step(render=True)
            recorder.capture()
            for agent in agents:
                agent.step(STEP_SIZE)
            if step % 100 == 0:
                print(f"[INFO] Step {step}/{total_steps}")

    print(f"[INFO] Done.")
    simulation_app.close()


if __name__ == "__main__":
    main()
