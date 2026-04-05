#!/usr/bin/env python3
"""
Validation: Cube Drop Test

Drops 9 randomly scattered cubes near each test camera, lets them settle,
records video from each camera, and checks that cubes rest on surfaces.

Usage:
    sbatch submit_isaac_job.sh simulations/environment/validate_cube_drop.py <usd_path>

Output:
    logs/validation/cube_drop_results.json
    videos/validation/validate_cubes_cam_*.mp4
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
import omni.usd
from pxr import Usd, UsdGeom, UsdPhysics, Sdf, UsdLux

_script_dir = os.path.dirname(os.path.abspath(__file__))
_project_dir = os.path.abspath(os.path.join(_script_dir, '..', '..'))
sys.path.insert(0, _project_dir)

from isaacsim.core.api import World
from isaacsim.core.api.objects import DynamicCuboid
from sim_utils.test_cameras import (
    TEST_CAMERAS, create_test_camera, add_interior_lighting, look_at_orientation,
    set_focal_length_on_prim,
)
from sim_utils.video_capture import VideoRecorder
from sim_utils.building_physics import setup_physics_scene, check_body_sleeping, raycast_down

CUBE_SIZE = 1.0         # feet (~0.3m) — visible from ceiling cameras
NUM_CUBES = 9           # per camera
DROP_HEIGHT = 4.0       # feet above target z (higher for bigger cubes)
SETTLE_FRAMES = 300     # ~5s at 60Hz physics (after recording, for final position check)
RECORD_FRAMES = 300     # ~10s at 30fps video (captures drop + bounce + settle)
PASS_THRESHOLD = -1.0   # cube final_z must be > (target_z + threshold)

CUBE_COLORS = [
    [1.0, 0.15, 0.15],   # red
    [0.2, 0.3, 1.0],     # blue
    [0.1, 0.8, 0.2],     # green
    [1.0, 0.9, 0.1],     # yellow
    [1.0, 0.4, 0.7],     # pink
    [0.0, 0.9, 0.9],     # cyan
    [1.0, 0.5, 0.0],     # orange
    [0.5, 0.0, 1.0],     # purple
    [0.8, 0.8, 0.8],     # white
]


def freeze_doors(stage):
    """Make all RigidBodyAPI prims kinematic so doors don't interfere."""
    for prim in stage.Traverse():
        if prim.HasAPI(UsdPhysics.RigidBodyAPI):
            rb = UsdPhysics.RigidBodyAPI(prim)
            rb.CreateKinematicEnabledAttr(True)


def add_ground_plane(stage, z):
    """Large collision plane as safety net."""
    from pxr import Gf
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

    print(f"[cube_drop] Loading: {usd_path}", flush=True)

    # --- Scene setup ---
    world = World(stage_units_in_meters=1.0)
    stage = omni.usd.get_context().get_stage()

    building_prim = stage.DefinePrim("/World/Building", "Xform")
    building_prim.GetReferences().AddReference(
        usd_path, primPath=Sdf.Path("/World/Building")
    )

    setup_physics_scene(stage)
    add_interior_lighting(stage)

    # Ground plane at building bottom
    building_prim = stage.GetPrimAtPath("/World/Building")
    bbox_cache = UsdGeom.BBoxCache(Usd.TimeCode.Default(), [UsdGeom.Tokens.default_])
    bbox = bbox_cache.ComputeWorldBound(building_prim)
    bb_min = bbox.ComputeAlignedRange().GetMin()
    add_ground_plane(stage, bb_min[2] - 0.5)

    # Freeze doors so they don't fly
    freeze_doors(stage)

    # --- Spawn cubes in camera's field of view ---
    # Camera is at ceiling corner looking down diagonally toward target.
    # Place cubes 65% of the way from camera to target on XY (center of
    # the visible floor area), at target Z (floor level) + drop height.
    # Tight 1.5ft grid so they stay grouped in frame.
    print("[cube_drop] Spawning cubes in camera FOV...", flush=True)

    all_cubes = {}  # cam_key -> list of (name, cube_obj, floor_z)
    for cam_key, cam_dict in TEST_CAMERAS.items():
        cx, cy, cz = cam_dict["pos"]
        tx, ty, tz = cam_dict["target"]

        # Center point: 65% from camera toward target on XY plane
        center_x = cx + (tx - cx) * 0.65
        center_y = cy + (ty - cy) * 0.65

        # Compute perpendicular direction for grid spread
        dx = tx - cx
        dy = ty - cy
        length = math.sqrt(dx**2 + dy**2)
        if length < 0.01:
            perp_x, perp_y = 1.0, 0.0
            fwd_x, fwd_y = 0.0, 1.0
        else:
            fwd_x, fwd_y = dx / length, dy / length
            perp_x, perp_y = -fwd_y, fwd_x

        # 3x3 grid with 2.5-foot spacing (room for 1ft cubes)
        grid_spacing = 2.5
        cubes_for_cam = []
        idx = 0
        for row in [-1, 0, 1]:
            for col in [-1, 0, 1]:
                drop_x = center_x + perp_x * col * grid_spacing + fwd_x * row * grid_spacing
                drop_y = center_y + perp_y * col * grid_spacing + fwd_y * row * grid_spacing
                drop_z = tz + DROP_HEIGHT

                name = f"cube_{cam_key}_{idx}"
                color = CUBE_COLORS[idx % len(CUBE_COLORS)]
                cube = world.scene.add(DynamicCuboid(
                    prim_path=f"/World/{name}",
                    name=name,
                    position=np.array([drop_x, drop_y, drop_z]),
                    scale=np.array([CUBE_SIZE] * 3),
                    color=np.array(color),
                ))
                cubes_for_cam.append((name, cube, tz))
                print(f"  {name}  drop=({drop_x:.1f}, {drop_y:.1f}, {drop_z:.1f})", flush=True)
                idx += 1

        all_cubes[cam_key] = cubes_for_cam

    # --- Create cameras ---
    cameras = {}
    for key, cam_dict in TEST_CAMERAS.items():
        cameras[key] = create_test_camera(key, cam_dict, "/World/CubeCam", stage=stage)

    # --- Simulate: warmup then record the drop live ---
    world.reset()
    for cam in cameras.values():
        cam.initialize()
        set_focal_length_on_prim(stage, cam)

    print(f"[cube_drop] Warming up (30 frames)...", flush=True)
    for _ in range(30):
        world.step(render=True)

    # --- Pre-drop floor raycast verification ---
    print("[cube_drop] Verifying floor colliders via raycast...", flush=True)
    floor_raycast_results = {}  # "cube_name" -> {hit, distance}
    for cam_key, cubes_for_cam in all_cubes.items():
        for name, cube, floor_z in cubes_for_cam:
            pos, _ = cube.get_world_pose()
            hit, dist, _ = raycast_down((float(pos[0]), float(pos[1]), float(pos[2])))
            floor_raycast_results[name] = {"hit": hit, "distance": round(dist, 3) if dist else None}
            if not hit:
                print(f"  WARNING: {name} has no floor collider below spawn point!", flush=True)

    # --- Velocity tracking setup ---
    cube_max_speed = {name: 0.0 for cam_key, cubes in all_cubes.items() for name, _, _ in cubes}
    cube_final_vel = {}

    # Record ALL cameras simultaneously while cubes drop and settle.
    # This captures the cubes falling, bouncing, and coming to rest.
    print(f"[cube_drop] Recording drop + settle ({RECORD_FRAMES} frames)...", flush=True)
    recorders = {}
    for cam_key in cameras:
        vname = f"validation/validate_cubes_cam_{cam_key}"
        recorders[cam_key] = VideoRecorder(vname, fps=30)
        recorders[cam_key].__enter__()

    for i in range(RECORD_FRAMES):
        world.step(render=True)
        for cam_key, cam in cameras.items():
            rgba = cam.get_rgba()
            if rgba is not None and i > 0:
                recorders[cam_key].capture_rgba(rgba)
        # Track velocity every 5th frame during drop
        if i % 5 == 0:
            for cam_key, cubes_for_cam in all_cubes.items():
                for name, cube, _ in cubes_for_cam:
                    try:
                        vel = cube.get_linear_velocity()
                        speed = float(np.linalg.norm(vel))
                        if speed > cube_max_speed[name]:
                            cube_max_speed[name] = speed
                        cube_final_vel[name] = [float(vel[0]), float(vel[1]), float(vel[2])]
                    except Exception:
                        pass
        if i % 60 == 0:
            print(f"  Frame {i}/{RECORD_FRAMES}", flush=True)

    for rec in recorders.values():
        rec.__exit__(None, None, None)

    # Additional settle frames (no recording) for final position check
    # Also track sleep state to detect when cubes have truly settled
    print(f"[cube_drop] Extra settling ({SETTLE_FRAMES} frames) with sleep detection...", flush=True)
    settle_frames = {cam_key: None for cam_key in all_cubes}
    for i in range(SETTLE_FRAMES):
        world.step(render=False)
        # Check sleep state every 10th frame
        if i % 10 == 0:
            for cam_key, cubes_for_cam in all_cubes.items():
                if settle_frames[cam_key] is not None:
                    continue  # already settled
                all_sleeping = True
                for name, cube, _ in cubes_for_cam:
                    sleeping = check_body_sleeping(stage, f"/World/{name}")
                    if sleeping is not True:
                        all_sleeping = False
                        break
                if all_sleeping:
                    settle_frames[cam_key] = RECORD_FRAMES + i
                    print(f"  Camera {cam_key}: all cubes settled at frame {RECORD_FRAMES + i}", flush=True)
            # Also capture final velocity on last frame
            if i == SETTLE_FRAMES - 1:
                for cam_key, cubes_for_cam in all_cubes.items():
                    for name, cube, _ in cubes_for_cam:
                        try:
                            vel = cube.get_linear_velocity()
                            cube_final_vel[name] = [float(vel[0]), float(vel[1]), float(vel[2])]
                        except Exception:
                            pass
        if i % 60 == 0:
            print(f"  Step {i}/{SETTLE_FRAMES}", flush=True)

    # --- Validate cube positions ---
    print(f"\n[cube_drop] Validating cube positions...", flush=True)
    camera_results = {}
    overall_pass = True

    for cam_key, cubes_for_cam in all_cubes.items():
        cubes_landed = 0
        cube_details = []
        for name, cube, floor_z in cubes_for_cam:
            pos, _ = cube.get_world_pose()
            final_z = float(pos[2])
            passed = final_z > (floor_z + PASS_THRESHOLD)
            if passed:
                cubes_landed += 1
            raycast_info = floor_raycast_results.get(name, {})
            final_vel = cube_final_vel.get(name)
            cube_details.append({
                "name": name,
                "final_position": [float(pos[0]), float(pos[1]), float(pos[2])],
                "floor_z": floor_z,
                "passed": passed,
                "final_velocity": final_vel,
                "max_impact_speed": round(cube_max_speed.get(name, 0.0), 3),
                "floor_raycast_hit": raycast_info.get("hit"),
                "floor_raycast_distance": raycast_info.get("distance"),
            })

        cam_passed = cubes_landed == len(cubes_for_cam)
        if not cam_passed:
            overall_pass = False

        camera_results[cam_key] = {
            "room": TEST_CAMERAS[cam_key]["room"],
            "passed": cam_passed,
            "cubes_landed": cubes_landed,
            "cubes_total": len(cubes_for_cam),
            "cube_details": cube_details,
            "frames_to_settle": settle_frames.get(cam_key),
            "video": f"validate_cubes_cam_{cam_key}.mp4",
        }

        status = "PASS" if cam_passed else "FAIL"
        print(f"  Camera {cam_key} ({TEST_CAMERAS[cam_key]['room']}): "
              f"{cubes_landed}/{len(cubes_for_cam)} landed [{status}]", flush=True)

    # --- Settle detection summary ---
    settled_values = [v for v in settle_frames.values() if v is not None]
    settle_detection = {
        "method": "sleep_state",
        "all_settled": len(settled_values) == len(settle_frames),
        "max_frames_to_settle": max(settled_values) if settled_values else None,
    }

    # --- Save results ---
    results = {
        "test": "cube_drop",
        "usd_file": os.path.basename(usd_path),
        "cameras": camera_results,
        "settle_detection": settle_detection,
        "overall_passed": overall_pass,
        "runtime_seconds": round(time.time() - t_start, 1),
    }

    out_dir = os.path.join(_project_dir, "logs", "validation")
    os.makedirs(out_dir, exist_ok=True)
    out_path = os.path.join(out_dir, "cube_drop_results.json")
    with open(out_path, "w") as f:
        json.dump(results, f, indent=2)

    print(f"\n[cube_drop] Results -> {out_path}", flush=True)
    print(f"[cube_drop] OVERALL: {'PASS' if overall_pass else 'FAIL'}", flush=True)

    simulation_app.close()


if __name__ == "__main__":
    main()
