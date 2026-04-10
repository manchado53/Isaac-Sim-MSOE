"""Crowd simulation — 20 agents navigating Floor 4 with third-person follow camera.

Building loaded via sublayer (same pattern as nova_carter_vlm_demo.py).
Agents are VisualCapsules placed at correct Floor 4 height (Z≈27.85 m).
Camera sits 3 m behind the followed agent at eye level (Z≈28.6 m),
corridor-locked heading with smooth real turns.

Output: videos/crowd_<timestamp>.mp4
"""

import os, sys, json, random, heapq
from datetime import datetime

os.environ["OMNI_KIT_ACCEPT_EULA"] = "YES"
os.environ["PRIVACY_CONSENT"]       = "Y"

from isaacsim import SimulationApp
simulation_app = SimulationApp({"headless": True, "width": 1280, "height": 720})

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

NUM_AGENTS     = 20
SIM_DURATION_S = 15.0
AGENT_SPEED    = 1.2          # m/s
STEP_SIZE      = 1.0 / 30.0
WARMUP_FRAMES  = 30

# Floor 4: slab at Z=26.8 m, walkable surface ~26.9 m (verified from robot scripts)
FLOOR_Z      = 26.9
AGENT_RADIUS = 0.25
AGENT_HEIGHT = 1.4
# capsule center = floor + bottom hemisphere + half cylinder
AGENT_CZ     = FLOOR_Z + AGENT_RADIUS + AGENT_HEIGHT / 2.0   # ≈ 27.85 m

# Camera: eye level ~1.7 m above floor, 3 m behind agent along heading
EYE_Z        = FLOOR_Z + 1.7     # ≈ 28.6 m
CAM_BACK     = 3.0               # m behind agent
CAMERA_PITCH = +0.20             # rad; positive = tilt DOWN (validated empirically)

TURN_THRESH  = np.deg2rad(20)    # ignore heading jitter below this
TURN_SPEED   = 0.10              # EMA blend rate once a real turn is detected


# ── Nav graph ─────────────────────────────────────────────────────────────────

def load_graph(path):
    with open(path) as f:
        data = json.load(f)
    nodes = {}
    for n in data["nodes"]:
        # XY only — Z is forced to FLOOR_Z for all agent movement
        nodes[n["id"]] = np.array([n["world_xy_cm"][0] / 100.0,
                                    n["world_xy_cm"][1] / 100.0])
    adj = {nid: [] for nid in nodes}
    for e in data["edges"]:
        a, b = e["from"], e["to"]
        if a not in nodes or b not in nodes:
            continue
        dist = np.linalg.norm(nodes[a] - nodes[b])
        adj[a].append((b, dist))
        adj[b].append((a, dist))
    return nodes, adj


def dijkstra(adj, start, goal):
    dist = {start: 0.0}
    prev = {}
    pq   = [(0.0, start)]
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
        self.xy           = nodes[self.current_node].copy().astype(float)
        self.heading      = np.array([1.0, 0.0])
        color = np.array([random.uniform(0.2, 0.95),
                          random.uniform(0.2, 0.95),
                          random.uniform(0.2, 0.95)])
        self.prim = world.scene.add(
            VisualCapsule(
                prim_path=f"/World/Agents/agent_{agent_id}",
                name=f"agent_{agent_id}",
                position=np.array([self.xy[0], self.xy[1], AGENT_CZ]),
                radius=AGENT_RADIUS,
                height=AGENT_HEIGHT,
                color=color,
            )
        )
        self.path     = []
        self.path_idx = 0
        self._pick_destination()

    def _pick_destination(self):
        node_ids = list(self.nodes.keys())
        dest = random.choice(node_ids)
        while dest == self.current_node:
            dest = random.choice(node_ids)
        self.path     = dijkstra(self.adj, self.current_node, dest)
        self.path_idx = 1

    def step(self, dt):
        if not self.path or self.path_idx >= len(self.path):
            self._pick_destination()
            return
        target_xy = self.nodes[self.path[self.path_idx]]
        direction = target_xy - self.xy
        dist      = np.linalg.norm(direction)
        if dist < 0.1:
            self.current_node = self.path[self.path_idx]
            self.path_idx    += 1
            if self.path_idx >= len(self.path):
                self._pick_destination()
        else:
            unit         = direction / dist
            self.heading = unit
            self.xy     += unit * AGENT_SPEED * dt
            self.prim.set_world_pose(
                position=np.array([self.xy[0], self.xy[1], AGENT_CZ])
            )


# ── Minimap ────────────────────────────────────────────────────────────────────

class Minimap:
    def __init__(self, nodes, edges, size=240):
        self.size = size
        self.fig, self.ax = plt.subplots(figsize=(size / 100, size / 100), dpi=100)
        self.fig.patch.set_facecolor("#111111")
        self.ax.set_facecolor("#111111")
        nxs = [v[0] for v in nodes.values()]
        nys = [v[1] for v in nodes.values()]
        for src, dst in edges:
            if src in nodes and dst in nodes:
                self.ax.plot([nodes[src][0], nodes[dst][0]],
                             [nodes[src][1], nodes[dst][1]],
                             color="#444444", linewidth=0.8, zorder=1)
        self.ax.scatter(nxs, nys, c="#888888", s=6, zorder=2)
        self.ax.set_xlim(min(nxs) - 2, max(nxs) + 2)
        self.ax.set_ylim(min(nys) - 2, max(nys) + 2)
        self.ax.set_aspect("equal")
        self.ax.axis("off")
        self.fig.tight_layout(pad=0)
        self._agents_sc = self.ax.scatter([], [], c="white",  s=10, zorder=3)
        self._follow_sc = self.ax.scatter([], [], c="#FFD700", s=30, zorder=4,
                                           edgecolors="white", linewidths=0.5)
        self._canvas = FigureCanvasAgg(self.fig)

    def render(self, agents, follow_agent):
        xs = np.array([a.xy[0] for a in agents])
        ys = np.array([a.xy[1] for a in agents])
        self._agents_sc.set_offsets(np.c_[xs, ys])
        self._follow_sc.set_offsets([[follow_agent.xy[0], follow_agent.xy[1]]])
        self._canvas.draw()
        buf = self._canvas.buffer_rgba()
        img = np.frombuffer(buf, dtype=np.uint8).reshape(self.size, self.size, 4)
        return img[:, :, :3].copy()

    def close(self):
        plt.close(self.fig)


def composite_minimap(frame_rgba, minimap_rgb, margin=10):
    if frame_rgba is None:
        return frame_rgba
    h, w = frame_rgba.shape[:2]
    mh, mw = minimap_rgb.shape[:2]
    y0, x0 = h - mh - margin, w - mw - margin
    frame_rgba[y0 - 2:y0 + mh + 2, x0 - 2:x0 + mw + 2, :3] = 20
    frame_rgba[y0:y0 + mh, x0:x0 + mw, :3] = minimap_rgb
    return frame_rgba


# ── Camera orientation helper ─────────────────────────────────────────────────

def cam_quat(yaw, pitch=CAMERA_PITCH):
    """Isaac Sim [w,x,y,z] quaternion for yaw then pitch.
    Positive pitch tilts the camera DOWN (validated empirically with validate_camera_pitch.py).
    """
    q = R.from_euler("zy", [yaw, pitch]).as_quat()   # scipy returns [x,y,z,w]
    return np.array([q[3], q[0], q[1], q[2]])         # Isaac expects [w,x,y,z]


# ── Main ──────────────────────────────────────────────────────────────────────

def main():
    timestamp  = datetime.now().strftime("%Y%m%d_%H%M%S")
    video_name = f"crowd_{timestamp}"

    print("[INFO] Loading nav graph...", flush=True)
    nodes, adj = load_graph(GRAPH_PATH)
    with open(GRAPH_PATH) as f:
        _gdata = json.load(f)
    edges = [(_e["from"], _e["to"]) for _e in _gdata["edges"]
             if _e["from"] in nodes and _e["to"] in nodes]
    print(f"[INFO] {len(nodes)} nodes, {len(edges)} edges", flush=True)

    # ── World + building ──────────────────────────────────────────────────────
    # Use sublayer insertion — the Revit USD has no defaultPrim so open_stage
    # returns an empty scene. Sublayer merges all prims directly.
    print("[INFO] Setting up world and loading building...", flush=True)
    world = World(stage_units_in_meters=1.0)
    stage = omni.usd.get_context().get_stage()
    stage.GetRootLayer().subLayerPaths.insert(0, USD_PATH)

    # Exterior sky light
    dome = UsdLux.DomeLight.Define(stage, Sdf.Path("/World/DomeLight"))
    dome.CreateIntensityAttr(30000)

    # Interior sphere lights at each nav node (at ceiling height above floor 4)
    _ceil_z = FLOOR_Z + 3.0
    for _i, (_nid, _nxy) in enumerate(nodes.items()):
        _s = UsdLux.SphereLight.Define(stage, Sdf.Path(f"/World/NavLight_{_i}"))
        _s.CreateIntensityAttr(300000)
        _s.CreateRadiusAttr(0.5)
        UsdGeom.XformCommonAPI(_s).SetTranslate(
            Gf.Vec3d(float(_nxy[0]), float(_nxy[1]), _ceil_z)
        )
    print(f"[INFO] Lighting: DomeLight + {len(nodes)} interior lights", flush=True)

    # Floor 4 ground plane — the USD slab may not have collision geometry, so
    # this physics plane at Z=26.75 acts as a safety floor (same as robot script)
    world.scene.add_default_ground_plane(
        z_position=26.75,
        name="floor4_ground_plane",
        prim_path="/World/floor4GroundPlane",
        static_friction=0.3,
        dynamic_friction=0.3,
        restitution=0.0,
    )

    # ── Agents ────────────────────────────────────────────────────────────────
    print(f"[INFO] Spawning {NUM_AGENTS} agents at Z={AGENT_CZ:.3f} m...", flush=True)
    agents       = [PedestrianAgent(i, nodes, adj, world) for i in range(NUM_AGENTS)]
    follow_agent = agents[0]
    minimap      = Minimap(nodes, edges, size=240)
    print(f"[INFO] Follow agent starts at XY=({follow_agent.xy[0]:.1f}, {follow_agent.xy[1]:.1f})",
          flush=True)

    # ── Camera ────────────────────────────────────────────────────────────────
    init_pos = np.array([follow_agent.xy[0] - CAM_BACK, follow_agent.xy[1], EYE_Z])
    follow_cam = Camera(
        prim_path="/World/FollowCam",
        position=init_pos,
        orientation=cam_quat(0.0),
        frequency=30,
        resolution=(1280, 720),
    )
    follow_cam.set_focal_length(9.0)   # wide FOV (~110° horizontal)

    world.reset()
    simulation_app.update()
    follow_cam.initialize()
    print(f"[INFO] Camera at Z={EYE_Z:.2f} m (floor {FLOOR_Z} + 1.7 m eye height)", flush=True)

    # Renderer warmup
    print(f"[INFO] Warming up ({WARMUP_FRAMES} frames)...", flush=True)
    for _ in range(WARMUP_FRAMES):
        world.step(render=True)

    rgba_test = follow_cam.get_rgba()
    if rgba_test is not None:
        print(f"[INFO] Warmup RGBA shape={rgba_test.shape} mean={rgba_test.mean():.1f}", flush=True)
    else:
        print("[WARN] get_rgba() returned None after warmup", flush=True)

    # ── Simulation loop ───────────────────────────────────────────────────────
    total_steps = int(SIM_DURATION_S / STEP_SIZE)
    print(f"[INFO] Recording {total_steps} steps ({SIM_DURATION_S:.0f} s @ 30 fps)...", flush=True)

    cam_heading = np.array(follow_agent.heading, dtype=float)

    frame = 0
    with VideoRecorder(video_name, fps=30) as recorder:
        while frame < total_steps:

            # 1. Move all agents
            for agent in agents:
                agent.step(STEP_SIZE)

            # 2. Corridor-locked camera heading — only rotate on real turns (>20 deg)
            raw_heading = np.array(follow_agent.heading, dtype=float)
            dot = float(np.clip(np.dot(cam_heading, raw_heading), -1.0, 1.0))
            if np.arccos(dot) > TURN_THRESH:
                cam_heading = TURN_SPEED * raw_heading + (1 - TURN_SPEED) * cam_heading
                cam_heading /= np.linalg.norm(cam_heading)

            # 3. Position camera: back off along heading, fixed eye height
            yaw  = float(np.arctan2(cam_heading[1], cam_heading[0]))
            back = R.from_euler("z", yaw).apply(np.array([-CAM_BACK, 0.0, 0.0]))
            cam_pos = np.array([
                follow_agent.xy[0] + back[0],
                follow_agent.xy[1] + back[1],
                EYE_Z,
            ])

            follow_cam.set_world_pose(
                position    = cam_pos,
                orientation = cam_quat(yaw),
            )

            # 4. Step physics + render
            world.step(render=True)

            # 5. Capture frame + minimap overlay
            rgba = follow_cam.get_rgba()
            if rgba is not None:
                rgba = rgba.copy()
                composite_minimap(rgba, minimap.render(agents, follow_agent))
                recorder.capture_rgba(rgba)

            if frame % 60 == 0:
                mean_px = rgba.mean() if rgba is not None else -1
                print(f"[INFO] Frame {frame:4d}/{total_steps}  "
                      f"px_mean={mean_px:.1f}  "
                      f"agent=({follow_agent.xy[0]:.1f},{follow_agent.xy[1]:.1f})  "
                      f"cam_z={cam_pos[2]:.2f}",
                      flush=True)

            frame += 1

    minimap.close()
    print(f"[INFO] Done. Video: videos/{video_name}.mp4", flush=True)
    simulation_app.close()


if __name__ == "__main__":
    main()
