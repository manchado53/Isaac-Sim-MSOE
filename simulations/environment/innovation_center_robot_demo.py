#!/usr/bin/env python3
"""
Robot Deployment Demo — Innovation Center (finalV1)

Loads the Innovation Center finalV1 physics USD, discovers floor level 1,
spawns a JetBot, drives it in a rectangular patrol pattern, and records
from two camera angles (follow camera + static overview).

Validates that the robot stays on the floor (doesn't fall through),
proving the static colliders work correctly.

Output:
  - videos/environment/ic_robot_demo_follow.mp4
  - videos/environment/ic_robot_demo_overview.mp4
  - logs/ic_robot_demo.txt
"""

import os
import sys
import math
import signal
import threading

os.environ["OMNI_KIT_ACCEPT_EULA"] = "YES"

from isaacsim import SimulationApp

simulation_app = SimulationApp({"headless": True})

import numpy as np
import isaacsim.core.utils.numpy.rotations as rot_utils

import carb
import omni.usd
from pxr import Sdf, UsdLux, UsdGeom, Usd, UsdPhysics, Gf
from isaacsim.core.api import World
from isaacsim.robot.wheeled_robots.controllers.differential_controller import DifferentialController
from isaacsim.robot.wheeled_robots.robots import WheeledRobot
from isaacsim.storage.native import get_assets_root_path

_script_dir = os.path.dirname(os.path.abspath(__file__))
_project_dir = os.path.abspath(os.path.join(_script_dir, '..', '..'))
sys.path.insert(0, _project_dir)

from sim_utils.video_capture import VideoRecorder
from sim_utils.follow_camera import FollowCamera
from sim_utils.building_physics import setup_physics_scene, iter_building_elements

_report_path = os.path.join(_project_dir, "logs", "ic_robot_demo.txt")
os.makedirs(os.path.dirname(_report_path), exist_ok=True)
_report_file = None


def log(msg=""):
    print(msg, flush=True)
    if _report_file:
        _report_file.write(msg + "\n")
        _report_file.flush()


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def discover_floor_levels(stage, building_path="/World/Building"):
    """Discover distinct floor surfaces by clustering floor element bboxes."""
    bbox_cache = UsdGeom.BBoxCache(
        Usd.TimeCode.Default(), [UsdGeom.Tokens.default_]
    )
    building_prim = stage.GetPrimAtPath(building_path)

    floor_data = []
    for _level, category_name, element in iter_building_elements(stage, building_prim):
        if category_name != "Floors":
            continue
        bbox = bbox_cache.ComputeWorldBound(element)
        box_range = bbox.ComputeAlignedRange()
        bb_min = box_range.GetMin()
        bb_max = box_range.GetMax()
        if bb_min == bb_max:
            continue
        dx = bb_max[0] - bb_min[0]
        dy = bb_max[1] - bb_min[1]
        if dx < 0.5 and dy < 0.5:
            continue
        floor_data.append({
            "z_top": bb_max[2],
            "x_center": (bb_min[0] + bb_max[0]) / 2.0,
            "y_center": (bb_min[1] + bb_max[1]) / 2.0,
            "area": dx * dy,
        })

    if not floor_data:
        return []

    floor_data.sort(key=lambda f: f["z_top"])
    clusters = []
    current = [floor_data[0]]
    for fd in floor_data[1:]:
        if fd["z_top"] - current[0]["z_top"] > 1.5:
            clusters.append(current)
            current = [fd]
        else:
            current.append(fd)
    clusters.append(current)

    levels = []
    for i, cluster in enumerate(clusters):
        total_area = sum(f["area"] for f in cluster)
        if total_area < 0.01:
            continue
        avg_x = sum(f["x_center"] * f["area"] for f in cluster) / total_area
        avg_y = sum(f["y_center"] * f["area"] for f in cluster) / total_area
        avg_z = sum(f["z_top"] * f["area"] for f in cluster) / total_area
        levels.append({
            "name": f"floor_{i + 1}",
            "z": avg_z,
            "center_x": avg_x,
            "center_y": avg_y,
            "count": len(cluster),
        })

    return levels


def add_ground_plane(stage, z):
    """Add a large collision plane at the given Z height."""
    UsdGeom.Xform.Define(stage, "/World/GroundPlane")
    mesh = UsdGeom.Mesh.Define(stage, "/World/GroundPlane/Mesh")
    half = 300.0
    mesh.CreatePointsAttr([
        Gf.Vec3f(-half, -half, z), Gf.Vec3f(half, -half, z),
        Gf.Vec3f(half, half, z), Gf.Vec3f(-half, half, z),
    ])
    mesh.CreateFaceVertexCountsAttr([4])
    mesh.CreateFaceVertexIndicesAttr([0, 1, 2, 3])
    mesh.CreateNormalsAttr([Gf.Vec3f(0, 0, 1)] * 4)
    UsdPhysics.CollisionAPI.Apply(mesh.GetPrim())
    coll = UsdPhysics.MeshCollisionAPI.Apply(mesh.GetPrim())
    coll.CreateApproximationAttr("none")


def freeze_doors(stage):
    """Make all RigidBodyAPI prims kinematic so doors don't fly."""
    frozen = 0
    for prim in stage.Traverse():
        if prim.HasAPI(UsdPhysics.RigidBodyAPI):
            rb = UsdPhysics.RigidBodyAPI(prim)
            rb.CreateKinematicEnabledAttr(True)
            frozen += 1
    return frozen


def look_at_orientation(cam_pos, target):
    dx = target[0] - cam_pos[0]
    dy = target[1] - cam_pos[1]
    dz = target[2] - cam_pos[2]
    yaw = math.degrees(math.atan2(dy, dx))
    pitch = math.degrees(math.atan2(-dz, math.sqrt(dx**2 + dy**2)))
    return rot_utils.euler_angles_to_quats(
        np.array([0, pitch, yaw]), degrees=True
    )


# ---------------------------------------------------------------------------
# Navigation
# ---------------------------------------------------------------------------

def build_nav_sequence(patrol_distance=3.0):
    """Build a rectangular patrol: forward, turn 90, repeat 4 times."""
    fwd_speed = 0.2     # m/s
    turn_rate = math.pi / 6  # rad/s

    fwd_frames = int(patrol_distance / fwd_speed * 60)
    turn_frames = int((math.pi / 2) / turn_rate * 60)

    seq = []
    frame = 0
    for _ in range(4):
        seq.append((frame, frame + fwd_frames, [fwd_speed, 0.0]))
        frame += fwd_frames
        seq.append((frame, frame + turn_frames, [0.0, turn_rate]))
        frame += turn_frames

    seq.append((frame, frame + fwd_frames, [fwd_speed, 0.0]))
    frame += fwd_frames

    return seq, frame


def get_command(step, nav_sequence):
    for start, end, cmd in nav_sequence:
        if start <= step < end:
            return cmd
    return [0, 0]


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    global _report_file
    _report_file = open(_report_path, "w")

    log("=" * 60)
    log("Robot Deployment Demo — Innovation Center (finalV1)")
    log("=" * 60)

    # --- Phase 1: Scene Setup ---
    log("\n--- Phase 1: Scene Setup ---")

    world = World(stage_units_in_meters=1.0)
    stage = omni.usd.get_context().get_stage()

    physics_usd = os.path.join(
        _project_dir, "assets", "environments", "msoe_building",
        "innovation_center_finalV1_physics.usd"
    )
    if not os.path.exists(physics_usd):
        log(f"ERROR: {physics_usd} not found.")
        log("Run setup_innovation_center_physics.py first.")
        _report_file.close()
        simulation_app.close()
        sys.exit(1)

    log(f"Loading: {physics_usd}")
    building_prim = stage.DefinePrim("/World/Building", "Xform")
    building_prim.GetReferences().AddReference(
        physics_usd, primPath=Sdf.Path("/World/Building")
    )
    log("Building loaded.")

    setup_physics_scene(stage)

    # Building bbox for ground plane
    building_prim = stage.GetPrimAtPath("/World/Building")
    bbox_cache = UsdGeom.BBoxCache(
        Usd.TimeCode.Default(), [UsdGeom.Tokens.default_]
    )
    bbox = bbox_cache.ComputeWorldBound(building_prim)
    box_range = bbox.ComputeAlignedRange()
    bb_min = box_range.GetMin()
    bb_max = box_range.GetMax()
    ground_z = bb_min[2] - 0.5

    log(f"Building bbox: ({bb_min[0]:.1f}, {bb_min[1]:.1f}, {bb_min[2]:.1f}) to "
        f"({bb_max[0]:.1f}, {bb_max[1]:.1f}, {bb_max[2]:.1f})")
    log(f"Ground plane at Z={ground_z:.1f}")

    add_ground_plane(stage, ground_z)

    frozen = freeze_doors(stage)
    log(f"Froze {frozen} dynamic doors")

    # Lighting
    distant = UsdLux.DistantLight.Define(stage, Sdf.Path("/World/Sun"))
    distant.CreateIntensityAttr(3000)
    distant.CreateAngleAttr(1.0)
    xform = UsdGeom.Xformable(distant.GetPrim())
    xform.AddRotateXYZOp().Set(Gf.Vec3f(-45, 30, 0))

    UsdLux.DomeLight.Define(stage, Sdf.Path("/World/DomeLight")).CreateIntensityAttr(500)

    # --- Phase 2: Floor Discovery & Robot Spawn ---
    log("\n--- Phase 2: Floor Discovery & Robot Spawn ---")

    floors = discover_floor_levels(stage)
    log(f"Found {len(floors)} floor levels:")
    for fl in floors:
        log(f"  {fl['name']:12s}  Z={fl['z']:.2f}m  "
            f"center=({fl['center_x']:.1f}, {fl['center_y']:.1f})  "
            f"({fl['count']} elements)")

    if not floors:
        log("ERROR: No floor levels found!")
        _report_file.close()
        simulation_app.close()
        sys.exit(1)

    # Select floor 1 (lowest)
    floor1 = floors[0]
    floor_cx = float(floor1["center_x"])
    floor_cy = float(floor1["center_y"])
    floor_z = float(floor1["z"])
    log(f"\nSelected: {floor1['name']} at Z={floor_z:.2f}")
    log(f"  Center: ({floor_cx:.1f}, {floor_cy:.1f})")

    # Get assets root for robot USD
    assets_root_path = get_assets_root_path()
    if assets_root_path is None:
        carb.log_error("Could not find Isaac Sim assets folder")
        log("ERROR: Could not find Isaac Sim assets folder")
        _report_file.close()
        simulation_app.close()
        sys.exit(1)

    # Spawn JetBot on floor 1
    jetbot_asset = assets_root_path + "/Isaac/Robots/NVIDIA/Jetbot/jetbot.usd"
    spawn_pos = np.array([floor_cx, floor_cy, floor_z + 0.05])
    log(f"Spawning JetBot at ({spawn_pos[0]:.1f}, {spawn_pos[1]:.1f}, {spawn_pos[2]:.2f})")

    jetbot = world.scene.add(WheeledRobot(
        prim_path="/World/Jetbot",
        name="jetbot",
        wheel_dof_names=["left_wheel_joint", "right_wheel_joint"],
        create_robot=True,
        usd_path=jetbot_asset,
        position=spawn_pos,
    ))
    controller = DifferentialController(
        name="ctrl", wheel_radius=0.03, wheel_base=0.1125
    )

    # --- Phase 3: Camera Setup ---
    log("\n--- Phase 3: Camera Setup ---")

    ceiling_z = floor_z + 3.5
    follow_cam = FollowCamera(
        prim_path="/World/FollowCamera",
        local_offset=[-1.5, 0.0, 0.8],
        resolution=(640, 480),
        fps=15,
        z_clamp=(floor_z + 0.3, ceiling_z),
    )
    log(f"  Follow camera: offset=[-1.5, 0, 0.8], z_clamp=({floor_z + 0.3:.1f}, {ceiling_z:.1f})")

    overview_cam = FollowCamera(
        prim_path="/World/OverviewCamera",
        local_offset=[8.0, 8.0, 12.0],
        resolution=(640, 480),
        fps=15,
        z_clamp=(floor_z + 5.0, floor_z + 20.0),
        look_height=0.0,
    )
    log(f"  Overview camera: FollowCamera offset=[8, 8, 12]")

    # --- Phase 4: Simulation ---
    log("\n--- Phase 4: Navigation Sequence ---")

    nav_sequence, total_nav_frames = build_nav_sequence(patrol_distance=2.0)
    settle_frames = 90
    idle_frames = 30
    max_frames = settle_frames + total_nav_frames + idle_frames
    log(f"Total frames: {max_frames} (settle={settle_frames}, nav={total_nav_frames}, idle={idle_frames})")

    world.reset()
    simulation_app.update()
    follow_cam.initialize()
    overview_cam.initialize()

    log("Warming up renderer (30 frames)...")
    for _ in range(30):
        world.step(render=True)

    z_samples = []
    initial_z = None

    log(f"\nRunning simulation ({max_frames} frames)...")

    stop_requested = False

    def _sigterm_handler(signum, frame_):
        nonlocal stop_requested
        stop_requested = True
        log("\n[SIGTERM received — finishing video recording...]")

    signal.signal(signal.SIGTERM, _sigterm_handler)

    with VideoRecorder("environment/ic_robot_demo_follow", fps=15) as follow_rec, \
         VideoRecorder("environment/ic_robot_demo_overview", fps=15) as overview_rec:

        for frame in range(max_frames):
            if stop_requested:
                log(f"  Stopped early at frame {frame}/{max_frames}")
                break

            if world.is_playing():
                robot_pos, robot_quat = jetbot.get_world_pose()
                follow_cam.update(robot_pos, robot_quat)
                overview_cam.update(robot_pos, robot_quat)

            world.step(render=True)

            if world.is_playing():
                nav_frame = frame - settle_frames
                if nav_frame >= 0:
                    cmd = get_command(nav_frame, nav_sequence)
                else:
                    cmd = [0, 0]
                jetbot.apply_wheel_actions(controller.forward(command=cmd))

                if frame % 2 == 0:
                    follow_rgba = follow_cam.get_rgba()
                    follow_rec.capture_rgba(follow_rgba)
                    overview_rgba = overview_cam.get_rgba()
                    overview_rec.capture_rgba(overview_rgba)

                robot_pos, _ = jetbot.get_world_pose()
                current_z = float(robot_pos[2])

                if initial_z is None:
                    initial_z = current_z

                if frame % 100 == 0:
                    z_samples.append(current_z)
                    log(f"  Frame {frame:5d}/{max_frames}  "
                        f"robot_z={current_z:.3f}  floor_z={floor_z:.3f}  "
                        f"delta={current_z - floor_z:.3f}")

    log("Video recording finalized.")

    # --- Phase 5: Validation ---
    log(f"\n{'=' * 60}")
    log("VALIDATION RESULTS")
    log("=" * 60)

    final_pos, _ = jetbot.get_world_pose()
    final_z = float(final_pos[2])
    z_delta = abs(final_z - floor_z)

    log(f"  Floor Z:        {floor_z:.3f}")
    log(f"  Initial Z:      {initial_z:.3f}")
    log(f"  Final Z:        {final_z:.3f}")
    log(f"  |Final - Floor|: {z_delta:.3f}")

    passed = z_delta < 0.5
    log(f"\n  RESULT: {'PASS' if passed else 'FAIL'}")
    if not passed:
        log(f"  Robot fell through floor or drifted too far (delta={z_delta:.3f} > 0.5m)")

    log(f"\n  Z-position samples (every 100 frames):")
    for i, z in enumerate(z_samples):
        log(f"    frame {i * 100:5d}: z={z:.3f}")

    log(f"\n{'=' * 60}")
    log("Output files:")
    log("  videos/environment/ic_robot_demo_follow.mp4")
    log("  videos/environment/ic_robot_demo_overview.mp4")
    log(f"  {_report_path}")
    log("=" * 60)

    _report_file.close()

    log("Closing simulation app (30s timeout)...")
    close_thread = threading.Thread(target=simulation_app.close, daemon=True)
    close_thread.start()
    close_thread.join(timeout=30)
    if close_thread.is_alive():
        log("simulation_app.close() timed out — forcing exit")

    sys.exit(0 if passed else 1)


if __name__ == "__main__":
    main()
