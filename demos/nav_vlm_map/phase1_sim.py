#!/usr/bin/env python3
"""
demos/nav_vlm_map/phase1_sim.py — Step 1: Deterministic full-floor navigation demo.

Nova Carter navigates all reachable nodes on Floor 4 of the MSOE Innovation Center
using a coverage route generated at runtime: all-pairs Dijkstra + nearest-neighbor
TSP + 2-opt, starting from the nav graph node nearest to the robot's spawn position.

Outputs:
  videos/nav_demo_follow_TIMESTAMP.mp4   — chase camera following robot
  videos/nav_demo_onboard_TIMESTAMP.mp4  — robot's onboard camera view
  videos/nav_demo_map_TIMESTAMP.mp4      — real-time floor plan map with robot position
  nav_demo_cache_JOBID.npz               — robot XY trajectory + door frames for Phase 2

Step 1: No VLM, no map overlay. Verify robot navigates and cameras work.

Usage:
  sbatch submit_isaac_job.sh demos/nav_vlm_map/phase1_sim.py
"""

import argparse
import json
import math
import os
import sys
import time

import numpy as np

# ---------------------------------------------------------------------------
# CLI args — parsed before Isaac Sim starts
# ---------------------------------------------------------------------------

_parser = argparse.ArgumentParser()
_parser.add_argument(
    "--max-waypoints", type=int, default=10,
    help="Cap the route to this many waypoints (default: 10). Use -1 for full tour.",
)
_args, _ = _parser.parse_known_args()

# ---------------------------------------------------------------------------
# Project root on sys.path — needed for navigation.coverage and sim_utils
# ---------------------------------------------------------------------------

_demo_dir    = os.path.dirname(os.path.abspath(__file__))
_project_dir = os.path.abspath(os.path.join(_demo_dir, "..", ".."))
sys.path.insert(0, _project_dir)

from navigation.coverage import find_nearest_node, generate_route

# ---------------------------------------------------------------------------
# Config
# ---------------------------------------------------------------------------

# Robot spawn — southern cluster near node_11 area.
ROBOT_SPAWN_XY = np.array([-22.0, 53.7])
ROBOT_SPAWN_Z  = 27.3    # Floor 4 slab Z=26.8 + 0.5 m clearance

NAV_GRAPH_PATH = os.path.join(
    _project_dir, "test_building", "test_nodes",
    "Final_Data", "floor_4_nav_graph_world_xflipped.json",
)

# P-controller tuning
ARRIVE_THRESHOLD = 0.6    # m — waypoint reached when within this distance
LINEAR_SPEED     = 1.0    # m/s forward
ANGULAR_GAIN     = 1.5    # proportional heading correction gain
TURN_THRESHOLD   = 0.25   # rad — above this error, reduce forward speed

# Known door positions on Floor 4 — frames captured here feed Phase 2 VLM
# (none along the long east-west corridor; extend this list when doors are added)
DOOR_POSITIONS = []
DOOR_TRIGGER_DIST = 4.0   # m — save frames for VLM when robot is within this range

CAM_RESOLUTION = (640, 480)
FPS            = 15

JOB_ID    = os.environ.get("SLURM_JOB_ID", "local")
RUN_STAMP = time.strftime("%Y%m%d_%H%M%S")

# ---------------------------------------------------------------------------
# Navigation setup — run before Isaac Sim starts to avoid unnecessary GPU time
# ---------------------------------------------------------------------------

print("[Nav] Loading nav graph...")
with open(NAV_GRAPH_PATH) as f:
    graph_data = json.load(f)

# Node XY lookup in metres (graph stores coords in cm)
node_xy = {
    n["id"]: np.array([n["world_xy_cm"][0] / 100.0,
                       n["world_xy_cm"][1] / 100.0])
    for n in graph_data["nodes"]
}

start_node = find_nearest_node(graph_data, ROBOT_SPAWN_XY)
print(f"[Nav] Spawn nearest node: {start_node}  XY={node_xy[start_node]}")

print("[Nav] Generating coverage route (Dijkstra + TSP + 2-opt)...")
full_path = generate_route(graph_data, start_node, num_restarts=10, seed=0)
if _args.max_waypoints > 0:
    full_path = full_path[:_args.max_waypoints]
waypoints = [node_xy[nid] for nid in full_path]
MAX_STEPS = max(3000, len(waypoints) * 600)   # ~20 s per waypoint at 30 Hz
print(f"[Nav] Route: {len(waypoints)} waypoints covering {len(set(full_path))} unique nodes")

# Pre-compute which waypoint indices are near door positions
door_waypoint_set = set()
for i, wp in enumerate(waypoints):
    for door_xy in DOOR_POSITIONS:
        if np.linalg.norm(wp - door_xy) <= DOOR_TRIGGER_DIST:
            door_waypoint_set.add(i)
print(f"[Nav] Door-adjacent waypoints: {len(door_waypoint_set)}")

# ---------------------------------------------------------------------------
# Map video setup — matplotlib figure pre-rendered before Isaac Sim starts
# ---------------------------------------------------------------------------

import matplotlib
matplotlib.use("Agg")                     # non-interactive, safe on headless nodes
import matplotlib.pyplot as plt
import matplotlib.collections as mc
import matplotlib.image as _mpimg

# Coordinate transform: world metres → floor plan pixel coords
# floor_4_occupancy.png is the native-resolution occupancy grid:
# 3cm/pixel, xy_min from floor_1_meta.json (verified: 14/15 corridor nodes
# land on white/free pixels).
_OCC_XY_MIN  = np.array([-8640.60, -1050.94])  # world xy_min in cm
_OCC_CELL_CM = 3.0   # cm per pixel

def world_m_to_pixel(xy_m):
    """Convert world-space [x, y] metres → floor plan pixel (px, py).
    X is mirrored to match the occupancy image orientation."""
    x_cm = xy_m[0] * 100.0
    y_cm = xy_m[1] * 100.0
    col = (x_cm - _OCC_XY_MIN[0]) / _OCC_CELL_CM
    row = (y_cm - _OCC_XY_MIN[1]) / _OCC_CELL_CM
    return _IMG_W - col, row

_floor_plan_path = os.path.join(_demo_dir, "floor4_plan.png")
_floor_img = _mpimg.imread(_floor_plan_path)
_IMG_H, _IMG_W = _floor_img.shape[:2]   # 4750 × 4314

# Map video resolution — higher than camera feeds for clarity
_MAP_W, _MAP_H = 1280, 720

# Compute all node pixel positions first so we can derive the zoom window
_node_list = list(node_xy.keys())
_node_idx  = {nid: i for i, nid in enumerate(_node_list)}
_node_px   = np.array([world_m_to_pixel(node_xy[nid]) for nid in _node_list])

# Zoom to the bounding box of all nav nodes with 300px padding
_PAD = 300
_x_min = max(0,      _node_px[:, 0].min() - _PAD)
_x_max = min(_IMG_W, _node_px[:, 0].max() + _PAD)
_y_min = max(0,      _node_px[:, 1].min() - _PAD)
_y_max = min(_IMG_H, _node_px[:, 1].max() + _PAD)

_fig, _ax = plt.subplots(figsize=(_MAP_W / 100, _MAP_H / 100), dpi=100)
_fig.subplots_adjust(left=0, right=1, top=1, bottom=0)
_ax.set_xlim(_x_min, _x_max)
_ax.set_ylim(_y_max, _y_min)   # y-axis flipped: pixel 0 at top
_ax.axis("off")

# Background: floor plan image
_ax.imshow(_floor_img, extent=[0, _IMG_W, _IMG_H, 0], aspect="auto", zorder=0)

# Nav graph edges — green semi-transparent (single LineCollection draw call)
_edge_segs = [
    [_node_px[_node_idx[e["from"]]], _node_px[_node_idx[e["to"]]]]
    for e in graph_data["edges"]
    if e["from"] in _node_idx and e["to"] in _node_idx
]
_ax.add_collection(mc.LineCollection(
    _edge_segs, colors=(0, 0.63, 0, 0.5), linewidths=2.0, zorder=1
))

# Nav graph nodes — blue dots
_ax.scatter(_node_px[:, 0], _node_px[:, 1],
            s=40, c="#0064ff", zorder=2, linewidths=0)

# Planned route — orange line + dots (matches Joey's TSP colour)
_route_px = np.array([world_m_to_pixel(wp) for wp in waypoints])
_ax.plot(_route_px[:, 0], _route_px[:, 1], color="#ff8800", lw=2.5, zorder=3)
_ax.scatter(_route_px[:, 0], _route_px[:, 1],
            s=50, c="#ff8800", zorder=4, linewidths=0)

# Start marker — green
_ax.scatter([_route_px[0, 0]], [_route_px[0, 1]],
            s=120, c="#00dd00", zorder=5, linewidths=0)

# Dynamic artists updated each frame
_tail_line,  = _ax.plot([], [], color="#ff4444", lw=2.5, zorder=6)
_robot_dot,  = _ax.plot([], [], "o", color="#ff0000", ms=12, zorder=7,
                         markeredgecolor="#ffffff", markeredgewidth=1.5)
_wp_marker,  = _ax.plot([], [], "x", color="#ffff00", ms=14, mew=3.0, zorder=8)

_map_tail_px = []   # pixel-space trajectory history

print(f"[Map] Figure ready ({_MAP_W}×{_MAP_H}px | floor plan {_IMG_W}×{_IMG_H}px)")

# ---------------------------------------------------------------------------
# Isaac Sim init — must come before any omni/usd imports
# ---------------------------------------------------------------------------

os.environ["OMNI_KIT_ACCEPT_EULA"] = "YES"
from isaacsim import SimulationApp
simulation_app = SimulationApp({"headless": True})

import carb
import omni.usd
from pxr import Sdf, UsdGeom, UsdLux
from scipy.spatial.transform import Rotation as ScipyR

import isaacsim.core.utils.numpy.rotations as rot_utils
from isaacsim.core.api import World
from isaacsim.robot.wheeled_robots.controllers.differential_controller import DifferentialController
from isaacsim.robot.wheeled_robots.robots import WheeledRobot
from isaacsim.sensors.camera import Camera
from isaacsim.storage.native import get_assets_root_path

from sim_utils.follow_camera import FollowCamera
from sim_utils.video_capture import VideoRecorder

# ---------------------------------------------------------------------------
# World + building
# ---------------------------------------------------------------------------

assets_root = get_assets_root_path()
if assets_root is None:
    carb.log_error("Could not find Isaac Sim assets root")
    simulation_app.close()
    sys.exit(1)

world = World(stage_units_in_meters=1.0)

_building_usd = os.path.join(
    _project_dir, "assets", "environments", "msoe_building",
    "innovation_center_finalV1.usd",
)
if not os.path.isfile(_building_usd):
    print(f"ERROR: Building USD not found: {_building_usd}")
    simulation_app.close()
    sys.exit(1)

print(f"[Sim] Loading building: {_building_usd}")
stage = omni.usd.get_context().get_stage()
stage.GetRootLayer().subLayerPaths.insert(0, _building_usd)

# Hide vegetation — trees/plants block camera views inside the atrium
_VEG_KEYWORDS = (
    "tree", "plant", "veget", "shrub", "foli", "planting",
    "landscap", "palm", "bush", "hedge", "grass", "flora",
)

def _hide_vegetation(stg):
    from pxr import UsdGeom as _UG
    hidden = 0
    for prim in stg.Traverse():
        if any(k in prim.GetName().lower() for k in _VEG_KEYWORDS):
            _UG.Imageable(prim).MakeInvisible()
            hidden += 1
    return hidden

print(f"[Sim] Vegetation hidden: {_hide_vegetation(stage)} prims")

# Lighting
dome = UsdLux.DomeLight.Define(stage, Sdf.Path("/World/DomeLight"))
dome.CreateIntensityAttr(50000)

# Fill light travels with robot (inverse-square needs proximity for interior corridors)
fill = UsdLux.SphereLight.Define(stage, Sdf.Path("/World/FillLight"))
fill.CreateIntensityAttr(300000)
fill.CreateRadiusAttr(0.1)
fill_xform = UsdGeom.Xformable(fill.GetPrim())

# Separate fill for the follow camera (robot fill is too dim at 5 m offset)
follow_fill = UsdLux.SphereLight.Define(stage, Sdf.Path("/World/FollowFillLight"))
follow_fill.CreateIntensityAttr(600000)
follow_fill.CreateRadiusAttr(0.1)
follow_fill_xform = UsdGeom.Xformable(follow_fill.GetPrim())

# Floor 4 collision plane (slab Z=26.8 + small buffer)
world.scene.add_default_ground_plane(
    z_position=26.75,
    name="floor4_plane",
    prim_path="/World/Floor4Plane",
    static_friction=0.3,
    dynamic_friction=0.3,
    restitution=0.0,
)

# ---------------------------------------------------------------------------
# Robot
# ---------------------------------------------------------------------------

# Face +X initially — corridor runs east-west, robot starts at western end
spawn_orientation = rot_utils.euler_angles_to_quats(
    np.array([0.0, 0.0, 0.0]), degrees=True
)

carter = world.scene.add(
    WheeledRobot(
        prim_path="/World/NovaCarter",
        name="nova_carter",
        wheel_dof_names=["joint_wheel_left", "joint_wheel_right"],
        create_robot=True,
        usd_path=assets_root + "/Isaac/Robots/NVIDIA/NovaCarter/nova_carter.usd",
        position=np.array([ROBOT_SPAWN_XY[0], ROBOT_SPAWN_XY[1], ROBOT_SPAWN_Z]),
        orientation=spawn_orientation,
    )
)
controller = DifferentialController(name="ctrl", wheel_radius=0.14, wheel_base=0.4)

# ---------------------------------------------------------------------------
# Cameras
# ---------------------------------------------------------------------------

ONBOARD_OFFSET = np.array([0.20, 0.0, 0.65])   # front-centre, 0.65 m up
W, H = CAM_RESOLUTION

onboard_cam = Camera(
    prim_path="/World/OnboardCam",
    position=np.array([ROBOT_SPAWN_XY[0] + 0.20,
                       ROBOT_SPAWN_XY[1],
                       ROBOT_SPAWN_Z + 0.65]),
    frequency=FPS,
    resolution=(W, H),
    orientation=spawn_orientation,
    annotator_device="cpu",
)

# FollowCamera maintains a local offset behind/above the robot and always
# points at it — imported from sim_utils/follow_camera.py
follow_cam = FollowCamera(
    prim_path="/World/FollowCam",
    local_offset=[-5.0, 0.0, 2.0],   # 5 m behind, 2 m above robot
    resolution=(W, H),
    fps=FPS,
    z_clamp=(27.0, 32.0),            # stay on Floor 4
    look_height=0.3,
)

world.reset()
simulation_app.update()
onboard_cam.initialize()
follow_cam.initialize()

# Wide FOV lens for onboard camera
usd_cam = UsdGeom.Camera(stage.GetPrimAtPath("/World/OnboardCam"))
usd_cam.GetFocalLengthAttr().Set(5.0)
usd_cam.GetClippingRangeAttr().Set((0.01, 10000.0))

# Re-apply spawn pose after world.reset() — WheeledRobot constructor orientation
# is not reliably applied by Isaac Sim after reset()
carter.set_world_pose(
    position=np.array([ROBOT_SPAWN_XY[0], ROBOT_SPAWN_XY[1], ROBOT_SPAWN_Z]),
    orientation=spawn_orientation,
)

print("[Sim] Warming up renderer (20 frames)...")
for _ in range(20):
    world.step(render=True)

# ---------------------------------------------------------------------------
# Navigation helpers
# ---------------------------------------------------------------------------

def get_robot_yaw(quat):
    """Extract world-frame yaw (radians) from [w, x, y, z] quaternion."""
    r = ScipyR.from_quat([quat[1], quat[2], quat[3], quat[0]])
    return r.as_euler("xyz", degrees=False)[2]


def normalize_angle(a):
    while a >  math.pi: a -= 2 * math.pi
    while a < -math.pi: a += 2 * math.pi
    return a


def steer_toward(robot_xy, robot_quat, target_xy):
    """
    Proportional heading controller toward target_xy.
    Returns [linear_speed, angular_speed] for DifferentialController.forward().
    """
    direction  = target_xy - robot_xy
    target_yaw = math.atan2(direction[1], direction[0])
    yaw_error  = normalize_angle(target_yaw - get_robot_yaw(robot_quat))
    angular    = float(np.clip(ANGULAR_GAIN * yaw_error, -2.0, 2.0))
    # Scale forward speed down when heading error is large
    linear     = LINEAR_SPEED * max(0.0, 1.0 - abs(yaw_error) / math.pi)
    return [linear, angular]

# ---------------------------------------------------------------------------
# Simulation loop
# ---------------------------------------------------------------------------

fill_translate_op        = fill_xform.AddTranslateOp()
follow_fill_translate_op = follow_fill_xform.AddTranslateOp()

wp_idx          = 0
step            = 0
video_frame_idx = 0
robot_xy_log    = []    # (x, y) per captured video frame
door_frames     = []    # onboard RGBA frames captured near doors (for Phase 2 VLM)
door_frame_vidx = []    # which video frame index each door frame corresponds to

# Waypoint stall detection — skip if stuck for more than 7 seconds (210 steps @ 30 Hz)
STALL_STEPS     = 210
wp_stall_steps  = 0     # steps spent on current waypoint without progress
wp_best_dist    = float("inf")  # closest we've gotten to current waypoint

print(f"\n[Sim] Starting — {len(waypoints)} waypoints, max {MAX_STEPS} steps\n")

_videos_dir = os.path.join(_project_dir, "videos")
os.makedirs(_videos_dir, exist_ok=True)

with (
    VideoRecorder(f"nav_demo_follow_{RUN_STAMP}",  fps=FPS, output_dir=_videos_dir) as follow_rec,
    VideoRecorder(f"nav_demo_onboard_{RUN_STAMP}", fps=FPS, output_dir=_videos_dir) as onboard_rec,
    VideoRecorder(f"nav_demo_map_{RUN_STAMP}",     fps=FPS, output_dir=_videos_dir) as map_rec,
):
    while step < MAX_STEPS:
        pos, quat = carter.get_world_pose()
        robot_xy  = pos[:2]

        # Advance past any waypoints already within arrival threshold
        while wp_idx < len(waypoints):
            if np.linalg.norm(robot_xy - waypoints[wp_idx]) < ARRIVE_THRESHOLD:
                wp_idx += 1
                wp_stall_steps = 0
                wp_best_dist   = float("inf")
            else:
                break

        if wp_idx >= len(waypoints):
            print("[Sim] All waypoints reached!")
            break

        # Stall detection — skip waypoint if no progress for 7 seconds
        dist_to_wp = np.linalg.norm(robot_xy - waypoints[wp_idx])
        if dist_to_wp < wp_best_dist:
            wp_best_dist  = dist_to_wp
            wp_stall_steps = 0
        else:
            wp_stall_steps += 1
            if wp_stall_steps >= STALL_STEPS:
                print(f"[Sim] Waypoint {wp_idx} stalled — skipping (dist={dist_to_wp:.2f}m)")
                wp_idx        += 1
                wp_stall_steps = 0
                wp_best_dist   = float("inf")
                continue

        # Drive command toward current waypoint
        cmd = steer_toward(robot_xy, quat, waypoints[wp_idx])
        carter.apply_wheel_actions(controller.forward(command=cmd))

        # Move fill lights with robot
        lp = pos + np.array([0.0, 0.0, 1.0])
        fill_translate_op.Set((float(lp[0]), float(lp[1]), float(lp[2])))

        flp = pos + np.array([-2.5, 0.0, 2.0])
        follow_fill_translate_op.Set((float(flp[0]), float(flp[1]), float(flp[2])))

        # Update cameras — must happen BEFORE world.step(render=True)
        r_obj = ScipyR.from_quat([quat[1], quat[2], quat[3], quat[0]])
        cam_world_pos = pos + r_obj.apply(ONBOARD_OFFSET)
        onboard_cam.set_world_pose(position=cam_world_pos, orientation=quat)
        follow_cam.update(pos, quat)

        world.step(render=True)

        # Capture every 2 steps → effective FPS = sim_hz/2 ≈ 15 fps
        if step % 2 == 0:
            ob = onboard_cam.get_rgba()
            fl = follow_cam.get_rgba()

            if ob is not None and fl is not None:
                ob = np.clip(ob, 0, 255).astype(np.uint8)
                fl = np.clip(fl, 0, 255).astype(np.uint8)

                onboard_rec.capture_rgba(ob)
                follow_rec.capture_rgba(fl)
                robot_xy_log.append(robot_xy.copy())

                # Save onboard frame if near a door (for Phase 2 VLM inference)
                if wp_idx in door_waypoint_set:
                    door_frames.append(ob.copy())
                    door_frame_vidx.append(video_frame_idx)

                # Map video frame — update robot dot + trajectory tail in matplotlib fig
                _rx_px, _ry_px = world_m_to_pixel(robot_xy)
                _map_tail_px.append((_rx_px, _ry_px))
                if len(_map_tail_px) > 1:
                    _tail_line.set_data(
                        [p[0] for p in _map_tail_px],
                        [p[1] for p in _map_tail_px],
                    )
                _robot_dot.set_data([_rx_px], [_ry_px])
                if wp_idx < len(waypoints):
                    _wpx, _wpy = world_m_to_pixel(waypoints[wp_idx])
                    _wp_marker.set_data([_wpx], [_wpy])
                _fig.canvas.draw()
                _map_buf = np.frombuffer(_fig.canvas.tostring_argb(), dtype=np.uint8)
                _map_frame = _map_buf.reshape(_MAP_H, _MAP_W, 4).copy()  # 1280×720
                # ARGB → RGBA
                _map_frame = np.concatenate(
                    [_map_frame[:, :, 1:], _map_frame[:, :, :1]], axis=2
                )
                map_rec.capture_rgba(_map_frame)

                video_frame_idx += 1

        if step % 300 == 0:
            print(
                f"[Sim] step {step:5d} | wp {wp_idx:3d}/{len(waypoints)} | "
                f"XY=({robot_xy[0]:.1f}, {robot_xy[1]:.1f})"
            )

        step += 1

world.stop()
print(
    f"\n[Sim] Complete — {step} steps | "
    f"{wp_idx}/{len(waypoints)} waypoints | "
    f"{video_frame_idx} frames | "
    f"{len(door_frames)} door frames"
)

# ---------------------------------------------------------------------------
# Save cache for Phase 2 (VLM inference + map overlay)
# ---------------------------------------------------------------------------

_cache_path = os.path.join(_project_dir, f"nav_demo_cache_{JOB_ID}.npz")

np.savez_compressed(
    _cache_path,
    # Door frames for VLM inference (small subset)
    door_frames=np.array(door_frames, dtype=np.uint8)
        if door_frames else np.zeros((0, H, W, 4), dtype=np.uint8),
    door_frame_video_indices=np.array(door_frame_vidx, dtype=np.int32),
    # Robot trajectory for map overlay
    robot_xy_m=np.array(robot_xy_log, dtype=np.float32),
    # Nav graph for map rendering
    nav_node_xy=np.array(list(node_xy.values()), dtype=np.float32),
    nav_node_ids=np.array(list(node_xy.keys())),
    nav_edges=np.array(
        [[e["from"], e["to"]] for e in graph_data["edges"]]
    ),
    # Metadata
    config=np.array([FPS, W, H, video_frame_idx], dtype=np.int32),
    run_stamp=np.array([RUN_STAMP]),
    video_follow=np.array([f"nav_demo_follow_{RUN_STAMP}.mp4"]),
    video_onboard=np.array([f"nav_demo_onboard_{RUN_STAMP}.mp4"]),
)

print(f"[Sim] Cache saved → {_cache_path}")
print(f"[Sim] Videos → videos/nav_demo_follow_{RUN_STAMP}.mp4")
print(f"[Sim]           videos/nav_demo_onboard_{RUN_STAMP}.mp4")
print(f"[Sim]           videos/nav_demo_map_{RUN_STAMP}.mp4")

# Detach camera annotators before shutdown — prevents segfault in
# libomni.syntheticdata.plugin during Python atexit cleanup
for cam in [onboard_cam, follow_cam._camera]:
    try:
        cam.detach()
    except Exception:
        pass

simulation_app.close()
