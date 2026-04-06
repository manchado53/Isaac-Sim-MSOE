"""Crowd simulation in the Kern Center digital twin.

Spawns N pedestrian agents on floor 4 and moves them between
navigation graph nodes using Dijkstra routing. Logs agent positions
every step for heatmap and traffic analysis.

Usage (on a GPU node):
    cd ~/IsaacLab
    ./isaaclab.sh -p ~/isaac-sim-msoe/simulations/crowd/crowd_simulation.py

Output:
    simulations/crowd/logs/crowd_log_<timestamp>.csv
"""

import os
import sys
import csv
import json
import random
import heapq
from datetime import datetime

os.environ["OMNI_KIT_ACCEPT_EULA"] = "YES"
os.environ["PRIVACY_CONSENT"] = "Y"

from isaacsim import SimulationApp
simulation_app = SimulationApp({"headless": True})

import numpy as np
from omni.isaac.core import World
from omni.isaac.core.objects import DynamicSphere
from omni.isaac.core.prims import XFormPrim
import omni.usd

# ── Config ────────────────────────────────────────────────────────────────────

PROJECT_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), "../.."))
USD_PATH    = os.path.join(PROJECT_DIR, "assets/environments/msoe_building/innovation_center_finalV1.usd")
GRAPH_PATH  = os.path.join(PROJECT_DIR, "assets/Graph/floor_4_nav_graph_world_xflipped.json")
LOG_DIR     = os.path.join(PROJECT_DIR, "simulations/crowd/logs")

NUM_AGENTS       = 50       # number of pedestrians
SIM_DURATION_S   = 300.0    # simulation duration in seconds
AGENT_SPEED_MS   = 1.2      # walking speed m/s (average human ~1.2-1.4 m/s)
LOG_INTERVAL     = 10       # log every N steps (reduces file size)
AGENT_RADIUS     = 0.25     # visual sphere radius (metres)
STEP_SIZE        = 1.0 / 30.0  # 30 Hz simulation

# ── Load navigation graph ─────────────────────────────────────────────────────

def load_graph(path: str):
    with open(path) as f:
        data = json.load(f)

    z_m = data["z_height_cm"] / 100.0

    nodes = {}
    for n in data["nodes"]:
        x_m = n["world_xy_cm"][0] / 100.0
        y_m = n["world_xy_cm"][1] / 100.0
        nodes[n["id"]] = np.array([x_m, y_m, z_m])

    # Build adjacency list from edges
    adj = {nid: [] for nid in nodes}
    for e in data["edges"]:
        a, b = e["from"], e["to"]
        dist = np.linalg.norm(nodes[a][:2] - nodes[b][:2])
        adj[a].append((b, dist))
        adj[b].append((a, dist))

    return nodes, adj


def dijkstra(adj, start: str, goal: str):
    """Return list of node IDs from start to goal."""
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

    # Reconstruct path
    path = []
    cur = goal
    while cur in prev:
        path.append(cur)
        cur = prev[cur]
    path.append(start)
    path.reverse()
    return path if path[0] == start else []


# ── Agent class ───────────────────────────────────────────────────────────────

class PedestrianAgent:
    """Moves along nav graph edges toward a random destination."""

    def __init__(self, agent_id: int, nodes: dict, adj: dict, world: World):
        self.agent_id = agent_id
        self.nodes    = nodes
        self.adj      = adj
        node_ids      = list(nodes.keys())

        # Random start node
        self.current_node = random.choice(node_ids)
        self.pos = nodes[self.current_node].copy().astype(float)

        # Spawn a small sphere to represent the agent
        prim_path = f"/World/Agents/agent_{agent_id}"
        color = np.array([random.random(), random.random(), random.random()])
        self.sphere = world.scene.add(
            DynamicSphere(
                prim_path=prim_path,
                name=f"agent_{agent_id}",
                position=self.pos,
                radius=AGENT_RADIUS,
                color=color,
            )
        )

        self.path: list = []
        self.path_idx: int = 0
        self._pick_new_destination()

    def _pick_new_destination(self):
        node_ids = list(self.nodes.keys())
        dest = random.choice(node_ids)
        while dest == self.current_node:
            dest = random.choice(node_ids)
        self.path = dijkstra(self.adj, self.current_node, dest)
        self.path_idx = 1  # index 0 is current node

    def step(self, dt: float):
        """Move toward next waypoint. Pick new destination on arrival."""
        if not self.path or self.path_idx >= len(self.path):
            self._pick_new_destination()
            return

        target_node = self.path[self.path_idx]
        target_pos  = self.nodes[target_node]

        direction = target_pos[:2] - self.pos[:2]
        dist = np.linalg.norm(direction)

        if dist < 0.1:
            # Reached waypoint
            self.current_node = target_node
            self.path_idx += 1
            if self.path_idx >= len(self.path):
                self._pick_new_destination()
        else:
            move = (direction / dist) * AGENT_SPEED_MS * dt
            self.pos[:2] += move
            # Update sphere position
            self.sphere.set_world_pose(position=self.pos)


# ── Main ──────────────────────────────────────────────────────────────────────

def main():
    os.makedirs(LOG_DIR, exist_ok=True)
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    log_path  = os.path.join(LOG_DIR, f"crowd_log_{timestamp}.csv")

    print(f"[INFO] Loading nav graph from: {GRAPH_PATH}")
    nodes, adj = load_graph(GRAPH_PATH)
    print(f"[INFO] Loaded {len(nodes)} nodes, {sum(len(v) for v in adj.values())//2} edges")

    # Open the USD stage BEFORE creating World — World attaches to the active stage.
    # Calling open_stage() after World() replaces the stage and causes a crash.
    print(f"[INFO] Loading USD scene from: {USD_PATH}")
    omni.usd.get_context().open_stage(USD_PATH)
    world = World(stage_units_in_meters=1.0)
    world.reset()

    print(f"[INFO] Spawning {NUM_AGENTS} agents...")
    agents = [PedestrianAgent(i, nodes, adj, world) for i in range(NUM_AGENTS)]

    total_steps = int(SIM_DURATION_S / STEP_SIZE)
    print(f"[INFO] Running {total_steps} steps ({SIM_DURATION_S:.0f}s) ...")
    print(f"[INFO] Logging to: {log_path}")

    with open(log_path, "w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(["step", "time_s", "agent_id", "x_m", "y_m", "node"])

        for step in range(total_steps):
            world.step(render=False)

            for agent in agents:
                agent.step(STEP_SIZE)

            if step % LOG_INTERVAL == 0:
                t = step * STEP_SIZE
                for agent in agents:
                    writer.writerow([
                        step, f"{t:.2f}", agent.agent_id,
                        f"{agent.pos[0]:.3f}", f"{agent.pos[1]:.3f}",
                        agent.current_node,
                    ])

            if step % 1000 == 0:
                print(f"[INFO] Step {step}/{total_steps} ({step*STEP_SIZE:.0f}s elapsed)")

    print(f"[INFO] Done. Log saved to: {log_path}")
    simulation_app.close()


if __name__ == "__main__":
    main()
