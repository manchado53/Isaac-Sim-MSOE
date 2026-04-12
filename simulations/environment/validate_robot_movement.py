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
    videos/validation/validate_robot_cam_*.mp4
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
from isaacsim.sensors.camera import Camera
from sim_utils.test_cameras import (
    TEST_CAMERAS, create_test_camera, add_interior_lighting,
    set_focal_length_on_prim, look_at_orientation,
    CAMERA_FREQUENCY, CAMERA_RESOLUTION,
)
from sim_utils.video_capture import VideoRecorder
from sim_utils.building_physics import setup_physics_scene, raycast_down

# Navigation parameters
# Robot is at native meter scale in a feet-scale building.
# Scaling articulated robots breaks PhysX (mass/inertia/joints don't rescale).
# Instead we keep native scale — the robot appears ~3x smaller than realistic
# but physics is correct. Camera positions compensate for visibility.
FWD_SPEED = 7.0           # stage_units/s — robot forward speed
APPROACH_OFFSET = 2.0     # feet — spawn this far beyond camera target (keep on floor)
SETTLE_FRAMES = 90        # 1.5s settle before driving
DRIVE_FRAMES = 600        # 10s at 60Hz
RECORD_FRAMES = 300       # 10s at 30fps video
Z_DROP_THRESHOLD = 2.0    # stage_units — max allowed Z drop (generous for feet-scale)


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

    camera_results = {}
    overall_pass = True

    # Create overhead test cameras (for screenshots / metrics)
    cameras = {}
    for key, cam_dict in TEST_CAMERAS.items():
        cameras[key] = create_test_camera(key, cam_dict, "/World/RobotCam", stage=stage)

    # Spawn robots beyond each camera's target, oriented to approach the camera
    controllers = {}
    robot_view_cams = {}  # low side-view cameras that actually show the robot
    for cam_key, cam_dict in TEST_CAMERAS.items():
        cx, cy, cz = cam_dict["pos"]
        tx, ty, tz = cam_dict["target"]
        room = cam_dict["room"]
        print(f"\n[robot_movement] Testing camera {cam_key}: {room}", flush=True)

        # Direction from camera to target (horizontal)
        dx = tx - cx
        dy = ty - cy
        length = math.sqrt(dx**2 + dy**2)
        fwd_x, fwd_y = dx / length, dy / length

        # Spawn beyond target so robot approaches toward camera
        spawn_x = tx + fwd_x * APPROACH_OFFSET
        spawn_y = ty + fwd_y * APPROACH_OFFSET
        spawn_pos = np.array([spawn_x, spawn_y, tz + 0.2])

        # Orient robot to face toward camera
        heading_yaw = math.degrees(math.atan2(-fwd_y, -fwd_x))
        orientation = rot_utils.euler_angles_to_quats(
            np.array([0, 0, heading_yaw]), degrees=True
        )

        robot_name = f"carter_{cam_key}"
        carter = world.scene.add(WheeledRobot(
            prim_path=f"/World/NovaCarter_{cam_key}",
            name=robot_name,
            wheel_dof_names=["joint_wheel_left", "joint_wheel_right"],
            create_robot=True,
            usd_path=carter_asset,
            position=spawn_pos,
            orientation=orientation,
        ))

        # Native Nova Carter wheel parameters (meters) — no scaling.
        # Scaling articulated robots breaks PhysX joint/collision physics.
        controllers[cam_key] = DifferentialController(
            name=f"ctrl_{cam_key}",
            wheel_radius=0.14,
            wheel_base=0.4,
        )

        # --- Robot-view camera: positioned at test camera target (verified
        # indoor point), pulled back toward camera so we're inside the room,
        # at low height looking toward the robot spawn point.
        rv_cam_pos = np.array([tx - fwd_x * 3.0, ty - fwd_y * 3.0, tz + 2.0])
        rv_cam_target = np.array([spawn_x, spawn_y, tz + 0.3])
        rv_orient = look_at_orientation(rv_cam_pos, rv_cam_target)

        rv_cam = Camera(
            prim_path=f"/World/RobotViewCam_{cam_key}",
            position=rv_cam_pos,
            frequency=CAMERA_FREQUENCY,
            resolution=CAMERA_RESOLUTION,
            orientation=rv_orient,
        )
        # Wider lens to show more of the room around the robot
        cam_prim = stage.GetPrimAtPath(f"/World/RobotViewCam_{cam_key}")
        UsdGeom.Camera(cam_prim).CreateFocalLengthAttr(10.0)
        robot_view_cams[cam_key] = rv_cam

        print(f"  Spawned at ({spawn_x:.1f}, {spawn_y:.1f}, {tz + 0.05:.2f}) "
              f"facing camera", flush=True)

    # Single world.reset() after all robots are added
    world.reset()
    for cam in cameras.values():
        cam.initialize()
        set_focal_length_on_prim(stage, cam)
    for rv_cam in robot_view_cams.values():
        rv_cam.initialize()

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

    # Drive robots toward cameras and record video
    print(f"[robot_movement] Driving toward cameras ({DRIVE_FRAMES} frames)...", flush=True)

    recorders = {}
    recording = False
    try:
        for cam_key in TEST_CAMERAS:
            vname = f"validation/validate_robot_cam_{cam_key}"
            recorders[cam_key] = VideoRecorder(vname, fps=30)
            recorders[cam_key].__enter__()
        recording = True
    except Exception as e:
        print(f"[robot_movement] WARNING: Video recording setup failed ({e}), "
              f"running physics only", flush=True)
        recorders = {}

    position_tracks = {k: [] for k in TEST_CAMERAS}
    velocity_tracks = {k: [] for k in TEST_CAMERAS}
    clearance_tracks = {k: [] for k in TEST_CAMERAS}

    for frame in range(DRIVE_FRAMES):
        # All robots drive forward (each oriented toward its camera)
        for cam_key in TEST_CAMERAS:
            robot_name = f"carter_{cam_key}"
            robot = world.scene.get_object(robot_name)
            robot.apply_wheel_actions(
                controllers[cam_key].forward(command=[FWD_SPEED, 0.0])
            )
            pos, _ = robot.get_world_pose()
            position_tracks[cam_key].append(
                [float(pos[0]), float(pos[1]), float(pos[2])]
            )

            # Track velocity (XY speed magnitude)
            try:
                vel = robot.get_linear_velocity()
                xy_speed = float(np.sqrt(vel[0]**2 + vel[1]**2))
                velocity_tracks[cam_key].append(xy_speed)
            except Exception:
                velocity_tracks[cam_key].append(0.0)

            # Track ground clearance (every 10th frame)
            if frame % 10 == 0:
                hit, dist, _ = raycast_down(
                    (float(pos[0]), float(pos[1]), float(pos[2])),
                    max_distance=20.0,
                )
                if hit and dist is not None:
                    clearance_tracks[cam_key].append(float(dist))

        world.step(render=True)

        if recording:
            for cam_key, rv_cam in robot_view_cams.items():
                try:
                    rgba = rv_cam.get_rgba()
                    if rgba is not None and frame > 0:
                        recorders[cam_key].capture_rgba(rgba)
                except Exception:
                    pass

        if frame % 120 == 0:
            print(f"  Frame {frame}/{DRIVE_FRAMES}", flush=True)

    if recording:
        for rec in recorders.values():
            try:
                rec.__exit__(None, None, None)
            except Exception:
                pass

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

        # Velocity stats
        vel_data = velocity_tracks[cam_key]
        mean_speed = round(float(np.mean(vel_data)), 4) if vel_data else 0.0
        max_speed = round(float(np.max(vel_data)), 4) if vel_data else 0.0
        sampled_vel = vel_data[::30] if len(vel_data) > 30 else vel_data
        sampled_vel = [round(v, 4) for v in sampled_vel]

        # Ground clearance stats
        clr_data = clearance_tracks[cam_key]
        min_clearance = round(float(min(clr_data)), 4) if clr_data else None
        sampled_clr = [round(c, 4) for c in clr_data]

        camera_results[cam_key] = {
            "room": cam_dict["room"],
            "passed": passed,
            "initial_z": round(initial_z, 3),
            "final_z": round(final_z, 3),
            "z_drop": round(z_drop, 3),
            "final_position": [float(pos[0]), float(pos[1]), float(pos[2])],
            "position_samples": sampled_track,
            "velocity_samples": sampled_vel,
            "mean_speed": mean_speed,
            "max_speed": max_speed,
            "ground_clearance_samples": sampled_clr,
            "min_ground_clearance": min_clearance,
            "video": f"validate_robot_cam_{cam_key}.mp4",
        }

        if not passed:
            overall_pass = False

        status = "PASS" if passed else "FAIL"
        print(f"  Camera {cam_key} ({cam_dict['room']}): "
              f"z_drop={z_drop:.3f}, speed={mean_speed:.3f}/{max_speed:.3f} m/s, "
              f"min_clearance={min_clearance} [{status}]", flush=True)

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

    # Skip simulation_app.close() — Isaac Sim's OmniGraph cleanup can segfault
    os._exit(0)


if __name__ == "__main__":
    main()
