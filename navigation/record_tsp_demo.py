#!/usr/bin/env python3
"""
navigation/record_tsp_demo.py — Nav2-driven TSP coverage tour with
three simultaneous video recordings (follow cam, onboard cam, top-down map).

Architecture:
    TSP route (offline, navigation.coverage)
        → Nav2 /navigate_through_poses action (one goal containing the
          full ordered PoseStamped list — DWB drives it continuously, no
          per-waypoint stop-and-go)
            → three parallel VideoRecorders ticking every N sim steps
                → trajectory cache for downstream VLM inference

Two route modes:
    --mode doors    (default) — TSP only on door nodes from
                                floor_N_door_node_map.json. Best for
                                door-coverage demos.
    --mode coverage           — TSP over every nav-graph node. Best for
                                generating training data with corridor
                                coverage too.

Requires Nav2 running in a separate terminal:
    ros2 launch kern_navigation kern_navigation.launch.py

Usage:
    ~/isaacsim/python.sh navigation/record_tsp_demo.py
    ~/isaacsim/python.sh navigation/record_tsp_demo.py --floor 4 --mode coverage
    ~/isaacsim/python.sh navigation/record_tsp_demo.py --floor 4 --mode doors --max-waypoints 20
    ~/isaacsim/python.sh navigation/record_tsp_demo.py --stream --stream-ip 100.107.199.57
"""

import argparse
import importlib.util
import json
import math
import os
import sys
import time
from pathlib import Path

import numpy as np
import yaml


# ---------------------------------------------------------------------------
# CLI parsed before Isaac Sim starts
# ---------------------------------------------------------------------------

def parse_args():
    p = argparse.ArgumentParser(
        description="Nav2-driven TSP tour with 3-camera recording."
    )
    p.add_argument("--building", default="InnovationCenter",
                   help="Building directory name under assets/environments/")
    p.add_argument("--floor", type=int, default=4)
    p.add_argument("--mode", choices=("doors", "coverage"), default="doors",
                   help="doors = TSP over door nodes only (default). "
                        "coverage = TSP over every reachable node.")
    p.add_argument("--door-map", default=None,
                   help="Path to floor_N_door_node_map.json. Defaults to "
                        "{nav-graph dir}/floor_{N}_door_node_map.json. "
                        "Only used when --mode doors.")
    p.add_argument("--max-waypoints", type=int, default=-1,
                   help="Cap TSP-expanded route to N waypoints. "
                        "-1 (default) = full route.")
    p.add_argument("--capture-every", type=int, default=2,
                   help="Record every Nth sim step (2 ≈ 15fps at 30Hz physics)")
    p.add_argument("--fps", type=int, default=15)
    p.add_argument("--tour-timeout", type=float, default=1800.0,
                   help="Seconds to wait for the entire NavigateThroughPoses "
                        "goal before giving up. Default 1800 (30 min).")
    p.add_argument("--stream", action="store_true",
                   help="Enable Isaac Sim WebRTC stream (visual only; cams still record).")
    p.add_argument("--stream-ip", default="100.107.199.57")
    p.add_argument("--headless", action="store_true",
                   help="Run Isaac Sim fully headless (no viewport, no stream).")
    p.add_argument("--no-map-video", action="store_true",
                   help="Skip the matplotlib top-down video (saves CPU).")
    p.add_argument("--obstacle", action="append", default=[],
                   metavar="X,Y[,SIZE]",
                   help="Spawn a kinematic red cube at world (X, Y) with "
                        "optional side length in meters (default 0.6). "
                        "Repeatable. Cubes are invisible to the static map, "
                        "so Nav2 only sees them via live lidar.")
    p.add_argument("--start-node", type=int, default=None,
                   help="Override spawn + TSP start: use this nav-graph node "
                        "ID instead of the building's default spawn. Spawn "
                        "XY is set to that node's world_xy_m; yaw is set to "
                        "face the second route node.")
    p.add_argument("--open-doors", type=int, default=0,
                   help="Pick K revolute door-joints near the route (on the "
                        "current floor) and set their DriveAPI target angle "
                        "to ~80° so physics opens them before the robot "
                        "arrives. 0 disables.")
    p.add_argument("--crack-glass", type=int, default=0,
                   help="Bind a cracked-glass OmniGlass material to K random "
                        "glass panels near the route. 0 disables.")
    p.add_argument("--seed", type=int, default=0,
                   help="RNG seed for door/glass selection (for reproducible demos).")
    return p.parse_args()


args = parse_args()


# ---------------------------------------------------------------------------
# Repo root on sys.path (needed for navigation.coverage + sim_utils)
# ---------------------------------------------------------------------------

REPO_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
sys.path.insert(0, REPO_ROOT)

# Load building_info directly to avoid sim_utils/__init__.py pulling in pxr
_bi_path = os.path.join(REPO_ROOT, "sim_utils", "building_info.py")
_spec = importlib.util.spec_from_file_location("_bi", _bi_path)
building_info = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(building_info)

floor = building_info.get_floor(args.building, args.floor)
spawn_x   = floor["default_spawn"]["x"]
spawn_y   = floor["default_spawn"]["y"]
spawn_z   = floor["floor_deck_z"]
spawn_yaw = floor["default_spawn"]["yaw"]

print(f"[demo] Building={args.building}  Floor={args.floor}  Mode={args.mode}")
print(f"[demo] Spawn XYZ=({spawn_x:.2f}, {spawn_y:.2f}, {spawn_z:.2f}) yaw={spawn_yaw:.3f}")
print(f"[demo] Graph: {floor['nav_graph_json']}")


# ---------------------------------------------------------------------------
# TSP route generation (pre-Isaac Sim; pure Python + numpy)
# ---------------------------------------------------------------------------

from navigation.coverage import (
    find_nearest_node, generate_route, generate_door_route,
)

with open(floor["nav_graph_json"]) as f:
    graph = json.load(f)

# coverage.py reads world_xy_m / distance_m natively now (post-refactor),
# but other downstream code in this file still uses world_xy_m via node_xy_m.
node_xy_m = {n["id"]: np.array(n["world_xy_m"], dtype=float)
             for n in graph["nodes"]}

if args.start_node is not None:
    if args.start_node not in node_xy_m:
        print(f"[demo] ERROR: --start-node {args.start_node} not in nav graph",
              file=sys.stderr)
        sys.exit(1)
    start_node = args.start_node
    xy = node_xy_m[start_node]
    spawn_x, spawn_y = float(xy[0]), float(xy[1])
    print(f"[demo] --start-node {start_node} overrides spawn to "
          f"({spawn_x:.2f}, {spawn_y:.2f})")
else:
    start_node = find_nearest_node(graph, [spawn_x, spawn_y])
    print(f"[demo] Spawn-nearest node: {start_node} @ {node_xy_m[start_node]}")

if args.mode == "doors":
    door_map_path = args.door_map or str(
        Path(floor["nav_graph_json"]).parent
        / f"floor_{args.floor}_door_node_map.json"
    )
    if not os.path.isfile(door_map_path):
        print(f"[demo] ERROR: --mode doors requires a door map; not found at "
              f"{door_map_path}", file=sys.stderr)
        sys.exit(1)
    print(f"[demo] Door map: {door_map_path}")
    with open(door_map_path) as f:
        door_map = json.load(f)
    door_ids = sorted({m["closest_node_id"]
                       for m in door_map["mappings"]
                       if m["closest_node_id"] is not None})
    print(f"[demo] Door nodes: {len(door_ids)} unique")
    print("[demo] Generating door TSP route (with nav-graph corridor expansion)...")
    # expand=True → TSP orders the doors, but the pose list is Dijkstra-
    # expanded with every intermediate corridor node along the shortest
    # nav-graph path between consecutive doors. Nav2 then just follows
    # our topology through narrow junctions instead of having its own
    # planner invent a route that may hug walls.
    route = generate_door_route(graph, start_node, door_ids,
                                num_restarts=10, seed=0, expand=True)
else:
    print("[demo] Generating full-coverage TSP route...")
    route = generate_route(graph, start_node, num_restarts=10, seed=0)

if args.max_waypoints > 0:
    route = route[: args.max_waypoints]
print(f"[demo] Route: {len(route)} waypoints "
      f"({len(set(route))} unique nodes, mode={args.mode})")

# If --start-node was used, set spawn yaw to face the second route node so
# the robot is already pointing the right way at t=0.
if args.start_node is not None and len(route) >= 2:
    a = node_xy_m[route[0]]
    b = node_xy_m[route[1]]
    spawn_yaw = float(math.atan2(b[1] - a[1], b[0] - a[0]))
    print(f"[demo] spawn yaw → face route[1] ({route[1]}): "
          f"{math.degrees(spawn_yaw):.1f}°")


# ---------------------------------------------------------------------------
# Map metadata (ROS map_server format)
# ---------------------------------------------------------------------------

with open(floor["map_metadata_yaml"]) as f:
    meta = yaml.safe_load(f)

MAP_ORIGIN_X, MAP_ORIGIN_Y, _ = meta["origin"]
MAP_RES    = float(meta["resolution"])    # m/px
MAP_WIDTH  = int(meta["grid_width"])
MAP_HEIGHT = int(meta["grid_height"])


def world_to_pixel(xy):
    """World (x, y) in metres → (col, row) in occupancy pixel frame.
    ROS map_server convention: row 0 is y_max (top of the PNG)."""
    col = (xy[0] - MAP_ORIGIN_X) / MAP_RES
    row = MAP_HEIGHT - (xy[1] - MAP_ORIGIN_Y) / MAP_RES
    return col, row


# ---------------------------------------------------------------------------
# Matplotlib top-down figure (pre-rendered once)
# ---------------------------------------------------------------------------

MAP_VID_W, MAP_VID_H = 1280, 720

if not args.no_map_video:
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    import matplotlib.collections as mc
    import matplotlib.image as mpimg

    occ_img = mpimg.imread(floor["occupancy_png"])
    IMG_H, IMG_W = occ_img.shape[:2]

    # Zoom to bbox of route + padding (route often covers a subset of the floor)
    route_px = np.array([world_to_pixel(node_xy_m[nid]) for nid in route])
    PAD = 150
    x_min = max(0,     route_px[:, 0].min() - PAD)
    x_max = min(IMG_W, route_px[:, 0].max() + PAD)
    y_min = max(0,     route_px[:, 1].min() - PAD)
    y_max = min(IMG_H, route_px[:, 1].max() + PAD)

    _fig, _ax = plt.subplots(figsize=(MAP_VID_W / 100, MAP_VID_H / 100), dpi=100)
    _fig.subplots_adjust(left=0, right=1, top=1, bottom=0)
    _ax.set_xlim(x_min, x_max)
    _ax.set_ylim(y_max, y_min)   # image coords: y grows downward
    _ax.axis("off")
    _ax.imshow(occ_img, extent=[0, IMG_W, IMG_H, 0], aspect="auto",
               cmap="gray", zorder=0)

    # All nodes (blue) + edges (green) for context
    node_idx = {n["id"]: i for i, n in enumerate(graph["nodes"])}
    all_px = np.array([world_to_pixel(np.array(n["world_xy_m"]))
                       for n in graph["nodes"]])
    edge_segs = [[all_px[node_idx[e["from"]]], all_px[node_idx[e["to"]]]]
                 for e in graph["edges"]
                 if e["from"] in node_idx and e["to"] in node_idx]
    _ax.add_collection(mc.LineCollection(
        edge_segs, colors=(0, 0.63, 0, 0.4), linewidths=1.5, zorder=1))
    _ax.scatter(all_px[:, 0], all_px[:, 1],
                s=12, c="#0064ff", zorder=2, linewidths=0)

    # TSP route (orange)
    _ax.plot(route_px[:, 0], route_px[:, 1],
             color="#ff8800", lw=2.5, zorder=3)
    _ax.scatter(route_px[:, 0], route_px[:, 1],
                s=25, c="#ff8800", zorder=4, linewidths=0)

    # Start marker (green) — spawn node
    _ax.scatter([route_px[0, 0]], [route_px[0, 1]],
                s=130, c="#00dd00", zorder=5, linewidths=0)

    # Dynamic artists updated each frame
    _tail_line, = _ax.plot([], [], color="#ff4444", lw=2.5, zorder=6)
    _robot_dot, = _ax.plot([], [], "o", color="#ff0000", ms=12, zorder=7,
                           markeredgecolor="white", markeredgewidth=1.5)
    _goal_x,    = _ax.plot([], [], "x", color="#ffff00", ms=14, mew=3.0,
                           zorder=8)
    _tail_px = []

    print(f"[demo] Map figure ready ({MAP_VID_W}x{MAP_VID_H}) "
          f"window px=[{x_min:.0f},{y_min:.0f}]-[{x_max:.0f},{y_max:.0f}]")


# ---------------------------------------------------------------------------
# Isaac Sim boot (must happen before any omni/usd/isaacsim imports)
# ---------------------------------------------------------------------------

os.environ["OMNI_KIT_ACCEPT_EULA"] = "YES"
from isaacsim import SimulationApp  # noqa: E402

if args.stream and args.headless:
    print("[demo] --headless ignored when --stream is set")

if args.stream:
    launch_config = {
        "width": 1280, "height": 720,
        "window_width": 1920, "window_height": 1080,
        "headless": True, "hide_ui": False,
        "renderer": "RaytracedLighting",
        "display_options": 3286,
    }
    print(f"[demo] Streaming mode; connect WebRTC client to {args.stream_ip}")
else:
    # Always headless for batch recording — cams sense internally
    launch_config = {"headless": True}

simulation_app = SimulationApp(launch_config)

if args.stream:
    simulation_app.set_setting("/app/window/drawMouse", True)
    simulation_app.set_setting("/app/livestream/allowResize", True)
    simulation_app.set_setting("/app/livestream/nvcf/quitOnSessionEnded", False)
    simulation_app.set_setting("/app/livestream/publicEndpointAddress", args.stream_ip)
    simulation_app.set_setting("/app/livestream/port", 49100)
    simulation_app.set_setting("/app/renderer/skipWhileMinimized", False)
    simulation_app.set_setting("/app/renderer/sleepMsOutOfFocus", 0)
    from isaacsim.core.utils.extensions import enable_extension
    enable_extension("omni.services.livestream.nvcf")

# ROS2 bridge BEFORE robot load — Nova_Carter_ROS action graphs reference it
from isaacsim.core.utils.extensions import enable_extension  # noqa: E402
enable_extension("isaacsim.ros2.bridge")
simulation_app.update()

import carb  # noqa: E402
carb.settings.get_settings().set_bool(
    "/exts/isaacsim.ros2.bridge/publish_without_verification", True
)
print("[demo] ros2 bridge enabled")

# ---------------------------------------------------------------------------
# Isaac Sim API imports (valid only after SimulationApp)
# ---------------------------------------------------------------------------

import omni.usd                                              # noqa: E402
import omni.graph.core as og                                 # noqa: E402
import random as _random                                     # noqa: E402
from pxr import Sdf, Usd, UsdGeom, UsdLux, UsdPhysics        # noqa: E402
from scipy.spatial.transform import Rotation as ScipyR       # noqa: E402

import isaacsim.core.utils.numpy.rotations as rot_utils      # noqa: E402
from isaacsim.core.api import World                          # noqa: E402
from isaacsim.core.api.objects import DynamicCuboid          # noqa: E402
from isaacsim.robot.wheeled_robots.robots import WheeledRobot  # noqa: E402
from isaacsim.sensors.camera import Camera                   # noqa: E402
from isaacsim.storage.native import get_assets_root_path     # noqa: E402

from sim_utils.building_physics import setup_physics_scene   # noqa: E402
from sim_utils.follow_camera import FollowCamera             # noqa: E402
from sim_utils.video_capture import VideoRecorder            # noqa: E402

# ---------------------------------------------------------------------------
# ROS2 imports are deferred to main() — the isaacsim.ros2.bridge extension
# needs several sim cycles after enable to patch sys.path for its bundled
# Python-3.11 rclpy. Importing at module top-level races the extension and
# fails with "No module named 'rclpy._rclpy_pybind11'".
# ---------------------------------------------------------------------------


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def yaw_to_quat_wxyz(yaw_rad):
    q = rot_utils.euler_angles_to_quats(
        np.array([0.0, 0.0, math.degrees(yaw_rad)]), degrees=True
    )
    return q  # [w, x, y, z]


def heading_to_next(wp_idx, route, node_xy):
    """Yaw (radians) facing from current waypoint toward the next.
    For the final waypoint, reuse the previous heading."""
    if wp_idx + 1 < len(route):
        a = node_xy[route[wp_idx]]
        b = node_xy[route[wp_idx + 1]]
    elif wp_idx > 0:
        a = node_xy[route[wp_idx - 1]]
        b = node_xy[route[wp_idx]]
    else:
        return 0.0
    return math.atan2(b[1] - a[1], b[0] - a[0])


def publish_initialpose(node, pos, quat, timeout_s=60.0):
    """Seed AMCL with the ground-truth spawn pose (same approach as
    nav2_orchestrator.py). Runs inline here rather than in a thread — the
    demo script has time to block briefly."""
    pub = node.create_publisher(PoseWithCovarianceStamped, "/initialpose", 10)
    t0 = time.time()
    while pub.get_subscription_count() == 0:
        if time.time() - t0 > timeout_s:
            print(f"[demo] /initialpose subscriber timeout after {timeout_s}s")
            return False
        rclpy.spin_once(node, timeout_sec=0.5)

    msg = PoseWithCovarianceStamped()
    msg.header.frame_id = "map"
    msg.pose.pose.position.x = float(pos[0])
    msg.pose.pose.position.y = float(pos[1])
    msg.pose.pose.position.z = 0.0
    msg.pose.pose.orientation.w = float(quat[0])
    msg.pose.pose.orientation.x = float(quat[1])
    msg.pose.pose.orientation.y = float(quat[2])
    msg.pose.pose.orientation.z = float(quat[3])
    msg.pose.covariance[0]  = 0.01
    msg.pose.covariance[7]  = 0.01
    msg.pose.covariance[35] = 0.05
    for _ in range(3):
        pub.publish(msg)
        rclpy.spin_once(node, timeout_sec=0.2)
    print(f"[demo] /initialpose seeded (xyz={[round(float(v),3) for v in pos]})")
    return True


def _open_doors_near_route(stage, route_node_ids, node_xy, floor_deck_z,
                           k, rng, search_radius=3.5, floor_z_tol=2.5,
                           open_angle_single=80.0, open_angle_double=60.0):
    """Find revolute joints (door hinges) within ``search_radius`` m of any
    route node, pick ``k`` of them, and set their angular DriveAPI target so
    physics swings them open during sim startup.

    Floor filter uses joint-parent bbox centroid Z vs ``floor_deck_z`` to
    avoid opening doors on other floors that happen to sit above/below a
    route XY.

    Returns the number of doors actually opened.
    """
    bbox_cache = UsdGeom.BBoxCache(Usd.TimeCode.Default(),
                                   [UsdGeom.Tokens.default_])
    joints = []
    for prim in stage.Traverse():
        if not prim.IsA(UsdPhysics.RevoluteJoint):
            continue
        parent = prim.GetParent()
        try:
            bbox = bbox_cache.ComputeWorldBound(parent)
        except Exception:
            continue
        if bbox.GetRange().IsEmpty():
            continue
        c = bbox.ComputeCentroid()
        xy = np.array([c[0], c[1]], dtype=float)
        joints.append((prim, xy, float(c[2]), parent.GetName()))

    if not joints:
        print("[demo] open-doors: no RevoluteJoints found under stage")
        return 0

    route_xy = np.array([node_xy[n] for n in route_node_ids])
    near = []
    for (joint_prim, xy, z, name) in joints:
        if abs(z - floor_deck_z) > floor_z_tol:
            continue
        d = float(np.linalg.norm(route_xy - xy, axis=1).min())
        if d < search_radius:
            near.append((d, joint_prim, xy, name))

    near.sort(key=lambda t: t[0])  # deterministic order before shuffle
    rng.shuffle(near)
    picked = near[:k]
    opened = 0
    for (d, joint_prim, xy, name) in picked:
        drive = UsdPhysics.DriveAPI(joint_prim, "angular")
        target_attr = drive.GetTargetPositionAttr()
        if not target_attr or not target_attr.IsValid():
            # Joint exists but no angular DriveAPI authored — apply it.
            drive = UsdPhysics.DriveAPI.Apply(joint_prim, "angular")
            target_attr = drive.CreateTargetPositionAttr()
        parent_name = joint_prim.GetParent().GetName() or ""
        is_double = any(tok in parent_name for tok in
                        ("2F", "2G", "2FG", "2FU", "Dual"))
        angle = open_angle_double if is_double else open_angle_single
        target_attr.Set(float(angle))
        print(f"[demo]   open door: {joint_prim.GetPath()}  "
              f"xy=({xy[0]:.2f},{xy[1]:.2f})  dist={d:.2f}m  target={angle:.0f}°")
        opened += 1
    return opened


def _crack_glass_near_route(stage, route_node_ids, node_xy, floor_deck_z,
                            k, rng, search_radius=6.0, floor_z_tol=3.0):
    """Bind a cracked-glass OmniGlass material to ``k`` glass panels within
    ``search_radius`` m of any route node on the current floor.

    Returns the number of panels actually cracked.
    """
    from sim_utils.glass_anomaly import (
        enumerate_glass_prims, create_cracked_glass_materials,
        bind_visual_material,
    )

    glass = enumerate_glass_prims(stage, "/World/Building")
    if not glass:
        print("[demo] crack-glass: no glass prims found under /World/Building")
        return 0

    tex_dir = os.path.join(REPO_ROOT, "assets", "materials", "cracked_glass")
    mats = create_cracked_glass_materials(stage, tex_dir)
    if not mats:
        print(f"[demo] crack-glass: no materials created from {tex_dir}")
        return 0

    bbox_cache = UsdGeom.BBoxCache(Usd.TimeCode.Default(),
                                   [UsdGeom.Tokens.default_])
    route_xy = np.array([node_xy[n] for n in route_node_ids])
    near = []
    for prim in glass:
        try:
            bbox = bbox_cache.ComputeWorldBound(prim)
        except Exception:
            continue
        if bbox.GetRange().IsEmpty():
            continue
        c = bbox.ComputeCentroid()
        if abs(float(c[2]) - floor_deck_z) > floor_z_tol:
            continue
        xy = np.array([c[0], c[1]], dtype=float)
        d = float(np.linalg.norm(route_xy - xy, axis=1).min())
        if d < search_radius:
            near.append((d, prim, xy))

    near.sort(key=lambda t: t[0])
    rng.shuffle(near)
    picked = near[:k]
    for (d, prim, xy) in picked:
        mat = rng.choice(mats)
        bind_visual_material(stage, prim, mat)
        print(f"[demo]   crack glass: {prim.GetPath()}  "
              f"xy=({xy[0]:.2f},{xy[1]:.2f})  dist={d:.2f}m  mat={mat}")
    return len(picked)


def _spawn_test_obstacles(stage, world, positions, floor_deck_z):
    """Spawn kinematic red cubes under /World/Obstacles so Nav2's obstacle
    layer (live lidar) sees them — static_layer won't, since they're not
    in the occupancy PNG. Must be called BEFORE world.reset(); freeze
    them kinematic AFTER world.reset() via _freeze_test_obstacles.
    Same pattern as nav2_orchestrator.py."""
    if not positions:
        return
    stage.DefinePrim("/World/Obstacles", "Xform")
    for i, (x, y, size) in enumerate(positions):
        # Place the cube's BOTTOM on the floor slab, not its center at
        # robot spawn z. With size=S, center must be at floor_deck_z + S/2
        # so the cube rests on the ground. That keeps the full cube
        # visible above the floor for both 2D (low) and 3D (high) lidars.
        z = floor_deck_z + size / 2.0
        cube = DynamicCuboid(
            prim_path=f"/World/Obstacles/obstacle_{i}",
            name=f"test_obstacle_{i}",
            position=np.array([x, y, z]),
            size=size,
            color=np.array([0.9, 0.15, 0.15]),  # bright red
            mass=10.0,
        )
        world.scene.add(cube)
        print(f"[demo] spawned obstacle {i}: ({x:.2f}, {y:.2f}, {z:.2f}) "
              f"size={size:.2f} m (resting on floor)", flush=True)


def _freeze_test_obstacles(stage, count):
    """Flip each spawned cube to kinematic AFTER world.reset(). Doing this
    before reset races PhysX's tensor view and breaks /cmd_vel silently
    (see nav2_orchestrator.py for the history)."""
    for i in range(count):
        prim = stage.GetPrimAtPath(f"/World/Obstacles/obstacle_{i}")
        if prim and prim.IsValid():
            UsdPhysics.RigidBodyAPI(prim).CreateKinematicEnabledAttr(True)
    if count > 0:
        print(f"[demo] froze {count} obstacles kinematic", flush=True)


def build_pose_stamped(node, xy, yaw):
    """Build a PoseStamped in the map frame for one waypoint."""
    ps = PoseStamped()
    ps.header.frame_id = "map"
    ps.header.stamp = node.get_clock().now().to_msg()
    ps.pose.position.x = float(xy[0])
    ps.pose.position.y = float(xy[1])
    ps.pose.position.z = 0.0
    q = yaw_to_quat_wxyz(yaw)
    ps.pose.orientation.w = float(q[0])
    ps.pose.orientation.x = float(q[1])
    ps.pose.orientation.y = float(q[2])
    ps.pose.orientation.z = float(q[3])
    return ps


class ThroughPosesDriver:
    """Wrap Nav2's NavigateThroughPoses action client. One goal contains
    the entire ordered pose list — Nav2 plans a continuous path and DWB
    drives it without per-waypoint stop-and-go.

    Tick the driver every sim loop iteration. State machine identical in
    shape to the previous per-waypoint driver, but for ONE goal."""

    PHASE_IDLE         = "idle"
    PHASE_AWAIT_HANDLE = "await_handle"
    PHASE_AWAIT_RESULT = "await_result"

    def __init__(self, node, timeout_s):
        self.node = node
        self.timeout_s = timeout_s
        self.client = ActionClient(node, NavigateThroughPoses,
                                   "navigate_through_poses")
        self.phase = self.PHASE_IDLE
        self.send_future = None
        self.result_future = None
        self.goal_handle = None
        self.last_status = None
        self.goal_start_t = 0.0
        self.poses_remaining = None   # filled by feedback callback
        self.total_poses = 0

    def wait_for_server(self, timeout_s=60.0):
        return self.client.wait_for_server(timeout_sec=timeout_s)

    def _on_feedback(self, msg):
        # NavigateThroughPoses.Feedback.number_of_poses_remaining
        self.poses_remaining = int(msg.feedback.number_of_poses_remaining)

    def send(self, pose_list):
        goal = NavigateThroughPoses.Goal()
        goal.poses = pose_list
        self.total_poses = len(pose_list)
        self.poses_remaining = len(pose_list)
        self.send_future = self.client.send_goal_async(
            goal, feedback_callback=self._on_feedback
        )
        self.phase = self.PHASE_AWAIT_HANDLE
        self.goal_start_t = time.time()
        self.last_status = None

    def tick(self):
        """Return True once the tour has concluded
        (success/fail/reject/timeout)."""
        if self.phase == self.PHASE_IDLE:
            return True

        if time.time() - self.goal_start_t > self.timeout_s:
            if self.goal_handle is not None:
                self.goal_handle.cancel_goal_async()
            self.last_status = "TIMEOUT"
            self.phase = self.PHASE_IDLE
            return True

        if self.phase == self.PHASE_AWAIT_HANDLE and self.send_future.done():
            self.goal_handle = self.send_future.result()
            if not self.goal_handle.accepted:
                self.last_status = "REJECTED"
                self.phase = self.PHASE_IDLE
                return True
            self.result_future = self.goal_handle.get_result_async()
            self.phase = self.PHASE_AWAIT_RESULT

        if self.phase == self.PHASE_AWAIT_RESULT and self.result_future.done():
            self.last_status = f"status={self.result_future.result().status}"
            self.phase = self.PHASE_IDLE
            return True

        return False


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    # Defer rclpy / nav2_msgs imports until we're inside main() — the
    # ROS2 bridge extension needs extra sim cycles after enable to make
    # its bundled Python-3.11 rclpy discoverable. Pushing these onto
    # module globals lets the helpers (publish_initialpose,
    # ThroughPosesDriver) resolve them at call time without surfacing
    # them in their signatures.
    global rclpy, ActionClient, NavigateThroughPoses
    global PoseWithCovarianceStamped, PoseStamped
    import rclpy
    from rclpy.action import ActionClient
    from nav2_msgs.action import NavigateThroughPoses
    from geometry_msgs.msg import PoseWithCovarianceStamped, PoseStamped

    t0 = time.time()
    def stamp(msg):
        print(f"[demo] [+{time.time()-t0:6.1f}s] {msg}", flush=True)

    # ----- Scene -----
    world = World(stage_units_in_meters=1.0)
    stage = omni.usd.get_context().get_stage()
    mpu = UsdGeom.GetStageMetersPerUnit(stage)
    stamp(f"stage metersPerUnit = {mpu}")

    building_prim = stage.DefinePrim("/World/Building", "Xform")
    building_prim.GetReferences().AddReference(
        floor["_usd_physics"], primPath=Sdf.Path("/World/Building")
    )
    setup_physics_scene(stage)
    stamp("building + physics ready")

    # --- Scene decoration: open some doors + crack some glass along the route ---
    if args.open_doors > 0 or args.crack_glass > 0:
        _rng = _random.Random(args.seed)
        if args.open_doors > 0:
            _opened = _open_doors_near_route(
                stage, route, node_xy_m, spawn_z, args.open_doors, _rng,
            )
            stamp(f"opened {_opened} doors near route (requested {args.open_doors})")
        if args.crack_glass > 0:
            _cracked = _crack_glass_near_route(
                stage, route, node_xy_m, spawn_z, args.crack_glass, _rng,
            )
            stamp(f"cracked {_cracked} glass panes near route (requested {args.crack_glass})")

    # Dome + travelling fills (interior corridors are dim otherwise)
    dome = UsdLux.DomeLight.Define(stage, Sdf.Path("/World/DomeLight"))
    dome.CreateIntensityAttr(30000)

    fill = UsdLux.SphereLight.Define(stage, Sdf.Path("/World/RobotFill"))
    fill.CreateIntensityAttr(300000)
    fill.CreateRadiusAttr(0.1)
    fill_xform = UsdGeom.Xformable(fill.GetPrim())
    fill_translate = fill_xform.AddTranslateOp()

    follow_fill = UsdLux.SphereLight.Define(stage, Sdf.Path("/World/FollowFill"))
    follow_fill.CreateIntensityAttr(600000)
    follow_fill.CreateRadiusAttr(0.1)
    ff_xform = UsdGeom.Xformable(follow_fill.GetPrim())
    ff_translate = ff_xform.AddTranslateOp()

    # ----- Robot -----
    assets_root = get_assets_root_path()
    if assets_root is None:
        raise RuntimeError("get_assets_root_path() returned None")

    orient = rot_utils.euler_angles_to_quats(
        np.array([0.0, 0.0, math.degrees(spawn_yaw)]), degrees=True
    )
    spawn_pos = np.array([spawn_x, spawn_y, spawn_z + 0.1])

    carter = world.scene.add(
        WheeledRobot(
            prim_path="/World/NovaCarter",
            name="nova_carter",
            wheel_dof_names=["joint_wheel_left", "joint_wheel_right"],
            create_robot=True,
            usd_path=assets_root + "/Isaac/Samples/ROS2/Robots/Nova_Carter_ROS.usd",
            position=spawn_pos,
            orientation=orient,
        )
    )
    stamp("robot added")

    # ----- Cameras -----
    ONBOARD_OFFSET = np.array([0.20, 0.0, 0.65])  # front, 65 cm up
    CAM_W, CAM_H = 640, 480

    onboard_cam = Camera(
        prim_path="/World/OnboardCam",
        position=spawn_pos + ONBOARD_OFFSET,
        frequency=args.fps,
        resolution=(CAM_W, CAM_H),
        orientation=orient,
        annotator_device="cpu",
    )
    follow_cam = FollowCamera(
        prim_path="/World/FollowCam",
        local_offset=[-5.5, 0.0, 2.5],   # 5.5 m behind, 2.5 m up — closer chase shot, wide lens keeps context
        resolution=(CAM_W, CAM_H),
        fps=args.fps,
        z_clamp=(spawn_z - 0.5, spawn_z + 6.0),
        look_height=0.3,
    )

    # Parse --obstacle flags into (x, y, size) tuples and spawn cubes
    # BEFORE world.reset() so physics handles initialize on first reset.
    obstacle_positions = []
    for spec in args.obstacle:
        parts = [float(p) for p in spec.split(",")]
        if len(parts) == 2:
            x, y = parts; s = 0.6
        elif len(parts) == 3:
            x, y, s = parts
        else:
            raise ValueError(f"--obstacle must be 'X,Y' or 'X,Y,SIZE'; got {spec!r}")
        obstacle_positions.append((x, y, s))
    _spawn_test_obstacles(stage, world, obstacle_positions, spawn_z)

    world.reset()
    carter.set_world_pose(position=spawn_pos, orientation=orient)
    # Freeze obstacles kinematic NOW (after reset) — doing it earlier
    # invalidates the tensor view used by Nova Carter's wheel controller.
    _freeze_test_obstacles(stage, len(obstacle_positions))
    simulation_app.update()
    onboard_cam.initialize()
    follow_cam.initialize()

    # Wide-FOV onboard lens
    usd_onboard = UsdGeom.Camera(stage.GetPrimAtPath("/World/OnboardCam"))
    usd_onboard.GetFocalLengthAttr().Set(5.0)
    usd_onboard.GetClippingRangeAttr().Set((0.01, 10000.0))

    # Wider follow-cam lens (~75° HFOV) for a roomier cinematic shot
    usd_follow = UsdGeom.Camera(stage.GetPrimAtPath("/World/FollowCam"))
    usd_follow.GetFocalLengthAttr().Set(14.0)
    usd_follow.GetClippingRangeAttr().Set((0.05, 10000.0))

    # Enable lidars, disable hawks (identical to nav2_orchestrator.py)
    robot_prim = "/World/NovaCarter"
    for lidar in ("front_2d_lidar_render_product",
                  "back_2d_lidar_render_product",
                  "front_3d_lidar_render_product"):
        og.Controller.attribute(
            f"{robot_prim}/ros_lidars/{lidar}.inputs:enabled"
        ).set(True)
    for hawk in ("front_hawk", "left_hawk", "right_hawk", "back_hawk"):
        for cam in ("left_camera_render_product", "right_camera_render_product"):
            try:
                og.Controller.attribute(
                    f"{robot_prim}/{hawk}/{cam}.inputs:enabled"
                ).set(False)
            except Exception:
                pass

    # Warm-up render so cameras have data before the first capture
    for _ in range(20):
        world.step(render=True)
    stamp("warmup complete")

    # ----- ROS 2 node + initialpose + NavigateThroughPoses client -----
    try:
        rclpy.init()
    except Exception:
        pass  # bridge extension may have already initialised rclpy

    ros_node = rclpy.create_node("record_tsp_demo")
    true_pos, true_orn = carter.get_world_pose()
    stamp(f"ground-truth pose: xyz={[round(float(v),3) for v in true_pos]}")

    publish_initialpose(ros_node, true_pos, true_orn, timeout_s=180.0)

    driver = ThroughPosesDriver(ros_node, timeout_s=args.tour_timeout)
    print("[demo] Waiting for /navigate_through_poses action server...")
    if not driver.wait_for_server(timeout_s=120.0):
        print("[demo] Nav2 action server did not come up — aborting.")
        ros_node.destroy_node()
        return
    stamp("nav2 action server exists")

    # Nav2's bt_navigator action server registers early but stays INACTIVE
    # until lifecycle_manager_navigation finishes activating (which itself
    # waits for AMCL localization, triggered by our /initialpose publish).
    # Without this warmup the goal lands on the inactive server and comes
    # back as REJECTED. Tick the sim for ~15s to advance /clock, fill TF,
    # and let the nav lifecycle come up.
    warmup_s = 15.0
    stamp(f"warming up Nav2 for {warmup_s:.0f}s "
          f"(ticking sim + spinning rclpy)")
    warmup_t0 = time.time()
    while time.time() - warmup_t0 < warmup_s:
        world.step(render=True)
        rclpy.spin_once(ros_node, timeout_sec=0.0)
    stamp("nav2 warmup complete")

    # ----- Build pose_list ONCE (skip route[0] — robot starts there) -----
    pose_list = []
    for i in range(1, len(route)):
        yaw = heading_to_next(i, route, node_xy_m)
        pose_list.append(build_pose_stamped(ros_node, node_xy_m[route[i]], yaw))
    stamp(f"built pose_list with {len(pose_list)} poses (route len {len(route)})")

    # ----- Dispatch the single NavigateThroughPoses goal -----
    driver.send(pose_list)
    stamp(f"NavigateThroughPoses goal dispatched (timeout={args.tour_timeout:.0f}s)")

    # ----- Open recorders + main tick loop -----
    run_stamp = time.strftime("%Y%m%d_%H%M%S")
    videos_dir = os.path.join(REPO_ROOT, "videos")
    os.makedirs(videos_dir, exist_ok=True)

    robot_xy_log = []
    video_frame_idx = 0
    step = 0
    last_logged_remaining = None

    recorders = [
        VideoRecorder(f"tsp_{args.mode}_follow_f{args.floor}_{run_stamp}",
                      fps=args.fps, output_dir=videos_dir),
        VideoRecorder(f"tsp_{args.mode}_onboard_f{args.floor}_{run_stamp}",
                      fps=args.fps, output_dir=videos_dir),
    ]
    if not args.no_map_video:
        recorders.append(VideoRecorder(
            f"tsp_{args.mode}_map_f{args.floor}_{run_stamp}",
            fps=args.fps, output_dir=videos_dir))

    active = [r.__enter__() for r in recorders]
    follow_rec  = active[0]
    onboard_rec = active[1]
    map_rec     = active[2] if not args.no_map_video else None

    try:
        while not driver.tick():
            pos, quat = carter.get_world_pose()

            # Tick cameras BEFORE world.step(render=True)
            r_obj = ScipyR.from_quat([quat[1], quat[2], quat[3], quat[0]])
            onboard_world_pos = pos + r_obj.apply(ONBOARD_OFFSET)
            onboard_cam.set_world_pose(
                position=onboard_world_pos, orientation=quat
            )
            follow_cam.update(pos, quat)

            # Fill lights travel with the robot
            lp = pos + np.array([0.0, 0.0, 1.0])
            fill_translate.Set((float(lp[0]), float(lp[1]), float(lp[2])))
            flp = pos + np.array([-2.5, 0.0, 2.0])
            ff_translate.Set((float(flp[0]), float(flp[1]), float(flp[2])))

            world.step(render=True)
            rclpy.spin_once(ros_node, timeout_sec=0.0)

            # Coarse progress log when poses_remaining ticks down
            if (driver.poses_remaining is not None
                    and driver.poses_remaining != last_logged_remaining):
                visited = driver.total_poses - driver.poses_remaining
                print(f"[demo] progress: {visited}/{driver.total_poses} "
                      f"poses passed ({driver.poses_remaining} remaining)")
                last_logged_remaining = driver.poses_remaining

            if step % args.capture_every == 0:
                ob = onboard_cam.get_rgba()
                fl = follow_cam.get_rgba()
                if ob is not None and fl is not None:
                    ob = np.clip(ob, 0, 255).astype(np.uint8)
                    fl = np.clip(fl, 0, 255).astype(np.uint8)
                    onboard_rec.capture_rgba(ob)
                    follow_rec.capture_rgba(fl)
                    robot_xy_log.append(pos[:2].copy())

                    if map_rec is not None:
                        rx, ry = world_to_pixel(pos[:2])
                        _tail_px.append((rx, ry))
                        if len(_tail_px) > 1:
                            _tail_line.set_data(
                                [p[0] for p in _tail_px],
                                [p[1] for p in _tail_px],
                            )
                        _robot_dot.set_data([rx], [ry])

                        # Yellow X = next pose Nav2 has yet to pass.
                        # poses_remaining=N → next target is pose_list
                        # index (total - N), which is route index
                        # (total - N) + 1 (because route[0] = spawn).
                        if driver.poses_remaining is not None:
                            target_pose_idx = max(0,
                                driver.total_poses - driver.poses_remaining)
                            target_route_idx = min(
                                len(route) - 1, target_pose_idx + 1)
                        else:
                            target_route_idx = 1
                        gx, gy = world_to_pixel(node_xy_m[route[target_route_idx]])
                        _goal_x.set_data([gx], [gy])

                        _fig.canvas.draw()
                        buf = np.frombuffer(
                            _fig.canvas.tostring_argb(), dtype=np.uint8
                        )
                        frame = buf.reshape(MAP_VID_H, MAP_VID_W, 4).copy()
                        # ARGB → RGBA (roll alpha to the end)
                        frame = np.concatenate(
                            [frame[:, :, 1:], frame[:, :, :1]], axis=2
                        )
                        map_rec.capture_rgba(frame)

                    video_frame_idx += 1

            step += 1

        elapsed = time.time() - driver.goal_start_t
        print(f"\n[demo] Tour finished: {driver.last_status} "
              f"after {elapsed:.1f}s, {video_frame_idx} video frames captured")

    finally:
        # Close recorders in reverse order
        for r in reversed(active):
            try:
                r.__exit__(None, None, None)
            except Exception as e:
                print(f"[demo] recorder close error: {e}")

        # Detach cameras to avoid syntheticdata segfault on shutdown
        for cam in (onboard_cam, follow_cam._camera):
            try:
                cam.detach()
            except Exception:
                pass

        # Cache trajectory for downstream VLM work
        cache_path = os.path.join(
            REPO_ROOT,
            f"tsp_{args.mode}_demo_cache_f{args.floor}_{run_stamp}.npz",
        )
        np.savez_compressed(
            cache_path,
            robot_xy_m=np.array(robot_xy_log, dtype=np.float32),
            route_node_ids=np.array(route),
            mode=np.array([args.mode]),
            config=np.array([args.fps, CAM_W, CAM_H, video_frame_idx],
                            dtype=np.int32),
            run_stamp=np.array([run_stamp]),
        )
        print(f"[demo] Cache saved → {cache_path}")

        ros_node.destroy_node()
        # Intentionally skip simulation_app.close() — OmniGraph cleanup
        # segfaults on exit (same reason as nav2_orchestrator.py).


if __name__ == "__main__":
    main()
