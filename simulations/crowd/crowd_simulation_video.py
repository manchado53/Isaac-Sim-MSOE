"""Crowd simulation — 20 humanoid agents navigating Floor 4.

Humanoids use human_skeleton.usd + walk.skelanim.usd (confirmed paths from inspection).
Animation driven by omni.timeline playing — fixes the T-pose from the earlier attempt.
Building loaded via sublayer. Camera ahead-of-agent looking back (no wall clipping).

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
from isaacsim.core.utils.stage import add_reference_to_stage
from pxr import UsdLux, UsdGeom, UsdSkel, Sdf, Gf
import omni.usd
import omni.kit.app

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "../..")))
from sim_utils.video_capture import VideoRecorder

# ── Config ────────────────────────────────────────────────────────────────────

PROJECT_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), "../.."))
USD_PATH    = os.path.join(PROJECT_DIR, "assets/environments/msoe_building/innovation_center_finalV1.usd")
GRAPH_PATH  = os.path.join(PROJECT_DIR, "assets/Graph/floor_4_nav_graph_world_xflipped.json")

# Confirmed paths from inspect_humanoid.py output
_EXT_BASE    = os.path.expanduser("~/isaacsim/extscache")
_RETARGET    = os.path.join(_EXT_BASE, "omni.anim.retarget.core-107.3.3+107.3.3.lx64.r.cp311.u353",
                             "data/rigs/Human/assets")
SKELETON_USD = os.path.join(_RETARGET, "human_skeleton.usd")
WALK_USD     = os.path.join(_RETARGET, "walk.skelanim.usd")

# Confirmed from inspection:
#   SkelRoot  relative path: __RETARGETING_TEMPLATE__/Biped_Demo
#   Animation relative path: WalkAnim/__RETARGETING_WALK_PREVIEW__
#   Animation time range: 0–80 (81 frames)
WALK_TIME_END = 80.0
WALK_FPS      = 24.0    # animation authored at 24 fps

NUM_AGENTS     = 15     # humanoids are heavier than capsules
SIM_DURATION_S = 15.0
AGENT_SPEED    = 1.2    # m/s
STEP_SIZE      = 1.0 / 30.0
WARMUP_FRAMES  = 60

# Floor 4: slab Z=26.8m verified from robot scripts
FLOOR_Z    = 26.9
CHAR_FOOT_Z = FLOOR_Z   # characters stand at floor level

# Camera: ahead of agent looking back — corridor ahead is open, no wall clipping
EYE_Z        = FLOOR_Z + 1.5
CAM_AHEAD    = 4.0
CAMERA_PITCH = +0.15    # positive = tilt down (validated)

TURN_THRESH  = np.deg2rad(20)
TURN_SPEED   = 0.10

# Hide trees — they clip through walls and are distracting
TREE_KEYWORDS = ["tree", "plant", "vegetation", "palm", "shrub", "foliage"]

# ── Nav graph ─────────────────────────────────────────────────────────────────

def load_graph(path):
    with open(path) as f:
        data = json.load(f)
    nodes = {}
    for n in data["nodes"]:
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
    """Humanoid agent: human_skeleton.usd + walk.skelanim.usd."""

    def __init__(self, agent_id, nodes, adj, stage):
        self.agent_id     = agent_id
        self.nodes        = nodes
        self.adj          = adj
        self.current_node = random.choice(list(nodes.keys()))
        self.xy           = nodes[self.current_node].copy().astype(float)
        self.heading      = np.array([1.0, 0.0])
        self._stage       = stage
        self._root        = f"/World/Agents/agent_{agent_id}"

        # ── Load humanoid USD ─────────────────────────────────────────────────
        add_reference_to_stage(SKELETON_USD, self._root)

        # ── Reference walk animation at a child path ──────────────────────────
        # walk.skelanim.usd contains /__RETARGETING_WALK_PREVIEW__ [SkelAnimation]
        # with 81 joints, 81 time samples (0–80), confirmed by inspect_humanoid.py.
        _walk_prim_path = f"{self._root}/WalkAnim"
        _walk_prim = stage.DefinePrim(_walk_prim_path)
        _walk_prim.GetReferences().AddReference(WALK_USD)

        # ── Bind animation to SkelRoot ────────────────────────────────────────
        # SkelRoot confirmed at: <root>/__RETARGETING_TEMPLATE__/Biped_Demo
        # Animation confirmed at: <root>/WalkAnim/__RETARGETING_WALK_PREVIEW__
        _skel_root_path = f"{self._root}/__RETARGETING_TEMPLATE__/Biped_Demo"
        _anim_path      = f"{_walk_prim_path}/__RETARGETING_WALK_PREVIEW__"
        _skel_root_prim = stage.GetPrimAtPath(_skel_root_path)
        if _skel_root_prim.IsValid():
            _binding = UsdSkel.BindingAPI.Apply(_skel_root_prim)
            _binding.CreateAnimationSourceRel().SetTargets([Sdf.Path(_anim_path)])
            print(f"[INFO] Agent {agent_id}: animation bound to {_anim_path}", flush=True)
        else:
            print(f"[WARN] Agent {agent_id}: SkelRoot not found at {_skel_root_path}", flush=True)

        # ── XformCommonAPI for position/rotation updates ──────────────────────
        _root_prim = stage.GetPrimAtPath(self._root)
        self._xform = UsdGeom.XformCommonAPI(_root_prim)
        self._update_pose()

        self.path     = []
        self.path_idx = 0
        self._pick_destination()

    def _update_pose(self):
        yaw_deg = float(np.degrees(np.arctan2(self.heading[1], self.heading[0])))
        self._xform.SetTranslate(Gf.Vec3d(float(self.xy[0]), float(self.xy[1]), CHAR_FOOT_Z))
        self._xform.SetRotate(Gf.Vec3f(0.0, 0.0, yaw_deg),
                               UsdGeom.XformCommonAPI.RotationOrderXYZ)

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
            self._update_pose()


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


def cam_quat(yaw, pitch=CAMERA_PITCH):
    """Isaac Sim [w,x,y,z] quaternion: yaw then pitch down."""
    q = R.from_euler("zy", [yaw, pitch]).as_quat()
    return np.array([q[3], q[0], q[1], q[2]])


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

    # ── World + building via sublayer ─────────────────────────────────────────
    print("[INFO] Setting up world...", flush=True)
    world = World(stage_units_in_meters=1.0)
    stage = omni.usd.get_context().get_stage()
    stage.GetRootLayer().subLayerPaths.insert(0, USD_PATH)

    # ── Hide trees ────────────────────────────────────────────────────────────
    hidden_trees = 0
    for prim in stage.Traverse():
        name = prim.GetName().lower()
        if any(kw in name for kw in TREE_KEYWORDS):
            UsdGeom.Imageable(prim).MakeInvisible()
            hidden_trees += 1
    print(f"[INFO] Hidden {hidden_trees} tree/plant prims", flush=True)

    # ── Lighting ──────────────────────────────────────────────────────────────
    # Reduced dome (was 30000 — too bright). Interior sphere lights for even coverage.
    dome = UsdLux.DomeLight.Define(stage, Sdf.Path("/World/DomeLight"))
    dome.CreateIntensityAttr(3000)

    _ceil_z = FLOOR_Z + 3.0
    for _i, (_nid, _nxy) in enumerate(nodes.items()):
        _s = UsdLux.SphereLight.Define(stage, Sdf.Path(f"/World/NavLight_{_i}"))
        _s.CreateIntensityAttr(80000)   # reduced from 300000
        _s.CreateRadiusAttr(1.0)
        UsdGeom.XformCommonAPI(_s).SetTranslate(
            Gf.Vec3d(float(_nxy[0]), float(_nxy[1]), _ceil_z)
        )
    print(f"[INFO] Lighting: DomeLight 3k + {len(nodes)} interior lights at 80k", flush=True)

    # Floor 4 safety ground plane
    world.scene.add_default_ground_plane(
        z_position=26.75,
        name="floor4_ground_plane",
        prim_path="/World/floor4GroundPlane",
        static_friction=0.3,
        dynamic_friction=0.3,
        restitution=0.0,
    )

    # ── Flush stage before spawning agents ────────────────────────────────────
    # Required so prim tree from sublayer is resolved before add_reference_to_stage
    simulation_app.update()
    simulation_app.update()

    # ── Spawn humanoid agents ─────────────────────────────────────────────────
    print(f"[INFO] Spawning {NUM_AGENTS} humanoid agents...", flush=True)
    agents       = [PedestrianAgent(i, nodes, adj, stage) for i in range(NUM_AGENTS)]
    follow_agent = agents[0]

    # Flush again so all references are resolved before world.reset()
    simulation_app.update()
    simulation_app.update()

    minimap = Minimap(nodes, edges, size=240)

    # ── Camera ────────────────────────────────────────────────────────────────
    init_pos = np.array([follow_agent.xy[0] + CAM_AHEAD, follow_agent.xy[1], EYE_Z])
    follow_cam = Camera(
        prim_path="/World/FollowCam",
        position=init_pos,
        orientation=cam_quat(0.0),
        frequency=30,
        resolution=(1280, 720),
    )

    world.reset()
    simulation_app.update()
    follow_cam.initialize()

    # Focal length + clipping via USD API (same as nova_carter_vlm_demo.py)
    usd_cam = UsdGeom.Camera(stage.GetPrimAtPath("/World/FollowCam"))
    usd_cam.GetFocalLengthAttr().Set(5.0)
    usd_cam.GetClippingRangeAttr().Set((0.01, 10000.0))

    # ── Start timeline so UsdSkel deformation runs ────────────────────────────
    # stage.SetCurrentTime() alone doesn't drive the skeleton deformer in Isaac Sim —
    # the timeline must be playing for UsdSkel to evaluate joint transforms.
    timeline = omni.timeline.get_timeline_interface()
    timeline.set_start_time(0.0)
    timeline.set_end_time(WALK_TIME_END)
    timeline.set_time_codes_per_second(WALK_FPS)
    timeline.play()
    print("[INFO] Timeline playing for UsdSkel animation", flush=True)

    # ── Warmup ────────────────────────────────────────────────────────────────
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
    print(f"[INFO] Recording {total_steps} steps ({SIM_DURATION_S:.0f}s @ 30fps)...", flush=True)

    cam_heading = np.array(follow_agent.heading, dtype=float)
    sim_time    = 0.0   # track simulation time for timeline

    frame = 0
    with VideoRecorder(video_name, fps=30) as recorder:
        while frame < total_steps:

            # 1. Move agents
            for agent in agents:
                agent.step(STEP_SIZE)

            # 2. Advance timeline time — cycles walk animation (0–80 frames at 24fps)
            sim_time += STEP_SIZE
            anim_time = (sim_time * WALK_FPS) % (WALK_TIME_END + 1)
            timeline.set_current_time(anim_time / WALK_FPS)

            # 3. Corridor-locked camera heading
            raw_heading = np.array(follow_agent.heading, dtype=float)
            dot = float(np.clip(np.dot(cam_heading, raw_heading), -1.0, 1.0))
            if np.arccos(dot) > TURN_THRESH:
                cam_heading = TURN_SPEED * raw_heading + (1 - TURN_SPEED) * cam_heading
                cam_heading /= np.linalg.norm(cam_heading)

            # 4. Camera ahead of agent, looking back
            yaw   = float(np.arctan2(cam_heading[1], cam_heading[0]))
            ahead = R.from_euler("z", yaw).apply(np.array([CAM_AHEAD, 0.0, 0.0]))
            cam_pos = np.array([
                follow_agent.xy[0] + ahead[0],
                follow_agent.xy[1] + ahead[1],
                EYE_Z,
            ])
            follow_cam.set_world_pose(position=cam_pos, orientation=cam_quat(yaw + np.pi))

            # 5. Step physics + render
            world.step(render=True)

            # 6. Capture + minimap
            rgba = follow_cam.get_rgba()
            if rgba is not None:
                rgba = rgba.copy()
                composite_minimap(rgba, minimap.render(agents, follow_agent))
                recorder.capture_rgba(rgba)

            if frame % 60 == 0:
                mean_px = rgba.mean() if rgba is not None else -1
                print(f"[INFO] Frame {frame:4d}/{total_steps}  px_mean={mean_px:.1f}  "
                      f"anim_t={anim_time:.1f}  "
                      f"agent=({follow_agent.xy[0]:.1f},{follow_agent.xy[1]:.1f})",
                      flush=True)

            frame += 1

    minimap.close()
    timeline.stop()
    print(f"[INFO] Done. Video: videos/{video_name}.mp4", flush=True)
    simulation_app.close()


if __name__ == "__main__":
    main()
