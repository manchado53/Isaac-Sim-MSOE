#!/usr/bin/env python3
"""
Validation: Robot Movement Test

Spawns a JetBot on each floor near the test camera, drives it in a short
patrol in front of the camera, records video, and verifies the robot
stays on the floor surface.

Usage:
    sbatch submit_isaac_job.sh simulations/environment/validate_robot_movement.py <usd_path>

Output:
    logs/validation/robot_movement_results.json
    videos/validate_robot_cam_*.mp4
"""

import os
import sys
import json
import math
import time

os.environ["OMNI_KIT_ACCEPT_EULA"] = "YES"

from isaacsim import SimulationApp

simulation_app = SimulationApp({"headless": True})

import numpy as np
import carb
import omni.usd
import isaacsim.core.utils.numpy.rotations as rot_utils
from pxr import Usd, UsdGeom, UsdPhysics, Sdf, Gf

_script_dir = os.path.dirname(os.path.abspath(__file__))
_project_dir = os.path.abspath(os.path.join(_script_dir, '..', '..'))
sys.path.insert(0, _project_dir)

from isaacsim.core.api import World
from isaacsim.robot.wheeled_robots.controllers.differential_controller import DifferentialController
from isaacsim.robot.wheeled_robots.robots import WheeledRobot
from isaacsim.storage.native import get_assets_root_path
from sim_utils.test_cameras import (
    TEST_CAMERAS, create_test_camera, add_interior_lighting,
    set_focal_length_on_prim,
)
from sim_utils.video_capture import VideoRecorder
from sim_utils.building_physics import setup_physics_scene

# Navigation parameters
FWD_SPEED = 0.2          # m/s
TURN_RATE = math.pi / 6  # rad/s
PATROL_DIST = 2.0        # meters per leg
SETTLE_FRAMES = 90       # 1.5s settle before driving
RECORD_FRAMES = 150      # 5s at 30fps
Z_DROP_THRESHOLD = 0.5   # meters — max allowed Z drop


def freeze_doors(stage):
    """Make all RigidBodyAPI prims kinematic."""
    for prim in stage.Traverse():
        if prim.HasAPI(UsdPhysics.RigidBodyAPI):
            UsdPhysics.RigidBodyAPI(prim).CreateKinematicEnabledAttr(True)


def add_ground_plane(stage, z):
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


def build_nav_sequence():
    """Build a rectangular patrol: forward, turn 90, repeat 4 times.

    Returns (sequence, total_frames) where sequence is list of
    (start_frame, end_frame, [linear_vel, angular_vel]).
    """
    fwd_frames = int(PATROL_DIST / FWD_SPEED * 60)
    turn_frames = int((math.pi / 2) / TURN_RATE * 60)

    seq = []
    frame = 0
    for _ in range(4):
        seq.append((frame, frame + fwd_frames, [FWD_SPEED, 0.0]))
        frame += fwd_frames
        seq.append((frame, frame + turn_frames, [0.0, TURN_RATE]))
        frame += turn_frames
    return seq, frame


def get_command(step, nav_sequence):
    for start, end, cmd in nav_sequence:
        if start <= step < end:
            return cmd
    return [0, 0]


def main():
    t_start = time.time()
    usd_path = sys.argv[1] if len(sys.argv) > 1 else os.path.join(
        _project_dir, "assets", "environments", "msoe_building",
        "innovation_center_physics.usd"
    )

    if not os.path.exists(usd_path):
        print(f"ERROR: USD not found: {usd_path}", flush=True)
        simulation_app.close()
        sys.exit(1)

    print(f"[robot_movement] Loading: {usd_path}", flush=True)

    # --- Scene setup ---
    world = World(stage_units_in_meters=1.0)
    stage = omni.usd.get_context().get_stage()

    building_prim = stage.DefinePrim("/World/Building", "Xform")
    building_prim.GetReferences().AddReference(
        usd_path, primPath=Sdf.Path("/World/Building")
    )

    setup_physics_scene(stage)
    add_interior_lighting(stage)

    # Ground plane
    building_prim = stage.GetPrimAtPath("/World/Building")
    bbox_cache = UsdGeom.BBoxCache(Usd.TimeCode.Default(), [UsdGeom.Tokens.default_])
    bbox = bbox_cache.ComputeWorldBound(building_prim)
    bb_min = bbox.ComputeAlignedRange().GetMin()
    add_ground_plane(stage, bb_min[2] - 0.5)

    freeze_doors(stage)

    # Robot asset — Nova Carter AMR (0.72m x 0.5m x 0.56m, 280mm wheels)
    assets_root_path = get_assets_root_path()
    if assets_root_path is None:
        carb.log_error("Could not find Isaac Sim assets folder")
        print("ERROR: Could not find Isaac Sim assets folder", flush=True)
        simulation_app.close()
        sys.exit(1)
    carter_asset = assets_root_path + "/Isaac/Robots/NVIDIA/NovaCarter/nova_carter.usd"

    # Nav sequence
    nav_sequence, total_nav_frames = build_nav_sequence()
    total_sim_frames = SETTLE_FRAMES + total_nav_frames

    # --- Per-camera robot test ---
    # We test each camera sequentially. For each camera:
    # 1. Spawn robot near camera target (on the floor)
    # 2. Run patrol, record video
    # 3. Check final Z
    # 4. Remove robot, reset for next camera

    camera_results = {}
    overall_pass = True

    # Create all test cameras up front
    cameras = {}
    for key, cam_dict in TEST_CAMERAS.items():
        cameras[key] = create_test_camera(key, cam_dict, "/World/RobotCam", stage=stage)

    # We need to test one floor per camera. The target Z gives us the floor level.
    for cam_key, cam_dict in TEST_CAMERAS.items():
        tx, ty, tz = cam_dict["target"]
        room = cam_dict["room"]
        print(f"\n[robot_movement] Testing camera {cam_key}: {room}", flush=True)

        # Spawn Nova Carter near camera target, slightly above floor
        spawn_pos = np.array([tx, ty, tz + 0.05])
        robot_name = f"carter_{cam_key}"

        carter = world.scene.add(WheeledRobot(
            prim_path=f"/World/NovaCarter_{cam_key}",
            name=robot_name,
            wheel_dof_names=["joint_wheel_left", "joint_wheel_right"],
            create_robot=True,
            usd_path=carter_asset,
            position=spawn_pos,
        ))
        controller = DifferentialController(
            name=f"ctrl_{cam_key}", wheel_radius=0.14, wheel_base=0.4
        )

        print(f"  Spawned at ({tx:.1f}, {ty:.1f}, {tz + 0.05:.2f})", flush=True)

    # Single world.reset() after all robots are added
    world.reset()
    for cam in cameras.values():
        cam.initialize()
        set_focal_length_on_prim(stage, cam)

    # Warmup
    print("[robot_movement] Warming up (30 frames)...", flush=True)
    for _ in range(30):
        world.step(render=True)

    # Now run the patrol and record per-camera
    # All robots run simultaneously but we record per-camera video
    print(f"[robot_movement] Settling ({SETTLE_FRAMES} frames)...", flush=True)
    for i in range(SETTLE_FRAMES):
        world.step(render=True)

    # Capture initial positions
    initial_positions = {}
    for cam_key in TEST_CAMERAS:
        robot_name = f"carter_{cam_key}"
        robot = world.scene.get_object(robot_name)
        pos, _ = robot.get_world_pose()
        initial_positions[cam_key] = float(pos[2])

    # Run navigation + record video for each camera
    print(f"[robot_movement] Running patrol ({total_nav_frames} frames)...", flush=True)

    # We record all cameras' videos simultaneously during the patrol
    recorders = {}
    video_dir = os.path.join(_project_dir, "videos")
    os.makedirs(video_dir, exist_ok=True)

    # Open all recorders
    for cam_key in TEST_CAMERAS:
        vname = f"validate_robot_cam_{cam_key}"
        recorders[cam_key] = VideoRecorder(vname, fps=30)
        recorders[cam_key].__enter__()

    position_tracks = {k: [] for k in TEST_CAMERAS}

    for frame in range(total_nav_frames):
        # Apply navigation commands to all robots
        for cam_key in TEST_CAMERAS:
            robot_name = f"carter_{cam_key}"
            robot = world.scene.get_object(robot_name)
            cmd = get_command(frame, nav_sequence)
            robot.apply_wheel_actions(
                controller.forward(command=cmd)
            )
            # Track position
            pos, _ = robot.get_world_pose()
            position_tracks[cam_key].append(
                [float(pos[0]), float(pos[1]), float(pos[2])]
            )

        world.step(render=True)

        # Capture from all cameras
        for cam_key, cam in cameras.items():
            rgba = cam.get_rgba()
            if rgba is not None and frame > 0:
                recorders[cam_key].capture_rgba(rgba)

        if frame % 120 == 0:
            print(f"  Frame {frame}/{total_nav_frames}", flush=True)

    # Close all recorders
    for rec in recorders.values():
        rec.__exit__(None, None, None)

    # --- Validate final positions ---
    print(f"\n[robot_movement] Validating positions...", flush=True)

    for cam_key, cam_dict in TEST_CAMERAS.items():
        tz = cam_dict["target"][2]
        robot_name = f"carter_{cam_key}"
        robot = world.scene.get_object(robot_name)
        pos, _ = robot.get_world_pose()
        final_z = float(pos[2])
        initial_z = initial_positions[cam_key]
        z_drop = initial_z - final_z

        passed = z_drop < Z_DROP_THRESHOLD

        # Subsample position track for JSON (every 30th frame)
        track = position_tracks[cam_key]
        sampled_track = track[::30] if len(track) > 30 else track

        camera_results[cam_key] = {
            "room": cam_dict["room"],
            "passed": passed,
            "initial_z": round(initial_z, 3),
            "final_z": round(final_z, 3),
            "z_drop": round(z_drop, 3),
            "final_position": [float(pos[0]), float(pos[1]), float(pos[2])],
            "position_samples": sampled_track,
            "video": f"validate_robot_cam_{cam_key}.mp4",
        }

        if not passed:
            overall_pass = False

        status = "PASS" if passed else "FAIL"
        print(f"  Camera {cam_key} ({cam_dict['room']}): "
              f"z_drop={z_drop:.3f} [{status}]", flush=True)

    # --- Save results ---
    results = {
        "test": "robot_movement",
        "usd_file": os.path.basename(usd_path),
        "cameras": camera_results,
        "overall_passed": overall_pass,
        "runtime_seconds": round(time.time() - t_start, 1),
    }

    out_dir = os.path.join(_project_dir, "logs", "validation")
    os.makedirs(out_dir, exist_ok=True)
    out_path = os.path.join(out_dir, "robot_movement_results.json")
    with open(out_path, "w") as f:
        json.dump(results, f, indent=2)

    print(f"\n[robot_movement] Results -> {out_path}", flush=True)
    print(f"[robot_movement] OVERALL: {'PASS' if overall_pass else 'FAIL'}", flush=True)

    simulation_app.close()


if __name__ == "__main__":
    main()
