#!/usr/bin/env python3
"""
Validation: Door Physics Test

For each test camera, finds nearby doors (within DOOR_SEARCH_RADIUS feet),
drives them to 80 degrees, measures final angle, and records video.

Usage:
    sbatch submit_isaac_job.sh simulations/environment/validate_door_physics.py <usd_path>

Output:
    logs/validation/door_physics_results.json
    videos/validate_doors_cam_*.mp4
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
from pxr import Usd, UsdGeom, UsdPhysics, Sdf, Gf

_script_dir = os.path.dirname(os.path.abspath(__file__))
_project_dir = os.path.abspath(os.path.join(_script_dir, '..', '..'))
sys.path.insert(0, _project_dir)

from isaacsim.core.api import World
from sim_utils.test_cameras import (
    TEST_CAMERAS, create_test_camera, add_interior_lighting,
    set_focal_length_on_prim,
)
from sim_utils.video_capture import VideoRecorder
from sim_utils.building_physics import setup_physics_scene, set_drive_target

DOOR_SEARCH_RADIUS = 20.0  # feet — max distance from camera target to door
DRIVE_TARGET_ANGLE = 80.0  # degrees
RECORD_FRAMES = 300        # ~10s at 30fps — records while doors swing open
PASS_THRESHOLD = 5.0       # degrees — minimum delta to count as "swung"


def get_door_angle_from_orient(prim):
    """Read the door's Z-axis rotation from its local xformOp:orient.

    IMPORTANT: Do NOT use ComputeLocalToWorldTransform — parent scale
    from Revit corrupts quaternion extraction.
    """
    xformable = UsdGeom.Xformable(prim)
    for op in xformable.GetOrderedXformOps():
        if op.GetOpType() == UsdGeom.XformOp.TypeOrient:
            quat = op.Get()
            if quat is None:
                return 0.0
            w = float(quat.GetReal())
            img = quat.GetImaginary()
            x, y, z = float(img[0]), float(img[1]), float(img[2])
            siny_cosp = 2.0 * (w * z + x * y)
            cosy_cosp = 1.0 - 2.0 * (y * y + z * z)
            return math.degrees(math.atan2(siny_cosp, cosy_cosp))
    return 0.0


def find_driven_doors(stage):
    """Find all revolute joints with angular DriveAPI on Door prims.

    Returns list of (path_str, joint_prim, drive, door_prim).
    """
    doors = []
    for prim in stage.Traverse():
        if not prim.IsA(UsdPhysics.RevoluteJoint):
            continue
        path_str = str(prim.GetPath())
        if "Door" not in path_str:
            continue
        drive = UsdPhysics.DriveAPI.Get(prim, "angular")
        if not drive or not drive.GetStiffnessAttr().HasValue():
            continue
        joint = UsdPhysics.RevoluteJoint(prim)
        targets = joint.GetBody1Rel().GetTargets()
        if not targets:
            continue
        door_prim = stage.GetPrimAtPath(targets[0])
        if not door_prim.IsValid():
            continue
        doors.append((path_str, prim, drive, door_prim))
    return doors


def door_short_name(path_str):
    """Extract short door identifier from path (e.g. 'Doors_2520150')."""
    for part in path_str.split("/"):
        if part.startswith("Doors_") or part.startswith("Door_"):
            return part
    return path_str.split("/")[-2] if "/" in path_str else path_str


def door_world_position(door_prim):
    """Get the world-space center of a door's bounding box."""
    bbox_cache = UsdGeom.BBoxCache(
        Usd.TimeCode.Default(), [UsdGeom.Tokens.default_]
    )
    bbox = bbox_cache.ComputeWorldBound(door_prim)
    box_range = bbox.ComputeAlignedRange()
    bb_min = box_range.GetMin()
    bb_max = box_range.GetMax()
    return np.array([
        (bb_min[0] + bb_max[0]) / 2.0,
        (bb_min[1] + bb_max[1]) / 2.0,
        (bb_min[2] + bb_max[2]) / 2.0,
    ])


def doors_near_camera(all_doors, cam_target, radius):
    """Filter doors within radius (feet) of camera target position."""
    nearby = []
    target = np.array(cam_target)
    for path_str, joint_prim, drive, door_prim in all_doors:
        door_pos = door_world_position(door_prim)
        dist = np.linalg.norm(door_pos - target)
        if dist <= radius:
            nearby.append((path_str, joint_prim, drive, door_prim, float(dist)))
    return nearby


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

    print(f"[door_physics] Loading: {usd_path}", flush=True)

    # --- Scene setup ---
    world = World(stage_units_in_meters=1.0)
    stage = omni.usd.get_context().get_stage()

    building_prim = stage.DefinePrim("/World/Building", "Xform")
    building_prim.GetReferences().AddReference(
        usd_path, primPath=Sdf.Path("/World/Building")
    )

    setup_physics_scene(stage)
    add_interior_lighting(stage)

    # --- Discover all driven doors ---
    all_doors = find_driven_doors(stage)
    print(f"[door_physics] Found {len(all_doors)} total driven doors", flush=True)

    # --- Create cameras ---
    cameras = {}
    for key, cam_dict in TEST_CAMERAS.items():
        cameras[key] = create_test_camera(key, cam_dict, "/World/DoorCam", stage=stage)

    # --- Assign doors to cameras ---
    cam_doors = {}
    for cam_key, cam_dict in TEST_CAMERAS.items():
        nearby = doors_near_camera(
            all_doors, cam_dict["target"], DOOR_SEARCH_RADIUS
        )
        cam_doors[cam_key] = nearby
        names = [door_short_name(p) for p, _, _, _, _ in nearby]
        print(f"  Camera {cam_key} ({cam_dict['room']}): "
              f"{len(nearby)} doors nearby", flush=True)
        for n in names:
            print(f"    - {n}", flush=True)

    # --- Simulation ---
    world.reset()
    for cam in cameras.values():
        cam.initialize()
        set_focal_length_on_prim(stage, cam)

    # Warmup
    print("[door_physics] Warming up (30 frames)...", flush=True)
    for _ in range(30):
        world.step(render=True)

    # Record initial angles for ALL doors
    initial_angles = {}
    for path_str, joint_prim, drive, door_prim in all_doors:
        initial_angles[path_str] = get_door_angle_from_orient(door_prim)

    # --- Record all cameras while doors swing open ---
    # Drive doors AND record simultaneously so the video shows the swing.
    print(f"[door_physics] Driving doors to {DRIVE_TARGET_ANGLE} deg + recording "
          f"({RECORD_FRAMES} frames)...", flush=True)

    # Set drive target on ALL doors at the start of recording
    for path_str, joint_prim, drive, door_prim in all_doors:
        set_drive_target(drive, DRIVE_TARGET_ANGLE)

    # Open all recorders
    recorders = {}
    for cam_key in cameras:
        vname = f"validate_doors_cam_{cam_key}"
        recorders[cam_key] = VideoRecorder(vname, fps=30)
        recorders[cam_key].__enter__()

    for i in range(RECORD_FRAMES):
        world.step(render=True)
        for cam_key, cam in cameras.items():
            rgba = cam.get_rgba()
            if rgba is not None and i > 0:
                recorders[cam_key].capture_rgba(rgba)
        if i % 60 == 0:
            print(f"  Frame {i}/{RECORD_FRAMES}", flush=True)

    # Close all recorders
    for rec in recorders.values():
        rec.__exit__(None, None, None)

    # Measure final angles for ALL doors
    final_angles = {}
    for path_str, joint_prim, drive, door_prim in all_doors:
        final_angles[path_str] = get_door_angle_from_orient(door_prim)

    # --- Build results ---
    print(f"\n[door_physics] Analyzing results...", flush=True)
    camera_results = {}
    overall_pass = True

    for cam_key, cam_dict in TEST_CAMERAS.items():
        nearby = cam_doors[cam_key]

        if not nearby:
            camera_results[cam_key] = {
                "room": cam_dict["room"],
                "status": "SKIP",
                "doors_found": 0,
                "doors_passed": 0,
                "doors_failed": 0,
                "door_details": [],
                "video": f"validate_doors_cam_{cam_key}.mp4",
            }
            print(f"  Camera {cam_key}: no doors nearby [SKIP]", flush=True)
            continue

        doors_passed = 0
        doors_failed = 0
        door_details = []

        for path_str, joint_prim, drive, door_prim, dist in nearby:
            name = door_short_name(path_str)
            initial = initial_angles.get(path_str, 0.0)
            final = final_angles.get(path_str, 0.0)
            delta = abs(final - initial)
            passed = delta > PASS_THRESHOLD

            if passed:
                doors_passed += 1
            else:
                doors_failed += 1

            door_details.append({
                "name": name,
                "path": path_str,
                "distance_from_camera": round(dist, 1),
                "initial_angle": round(initial, 1),
                "final_angle": round(final, 1),
                "delta": round(delta, 1),
                "passed": passed,
            })

            status = "PASS" if passed else "FAIL"
            print(f"  {name:<25s} {initial:7.1f} -> {final:7.1f} "
                  f"(delta={delta:.1f}) [{status}]", flush=True)

        cam_passed = doors_failed == 0
        if not cam_passed:
            overall_pass = False

        camera_results[cam_key] = {
            "room": cam_dict["room"],
            "status": "PASS" if cam_passed else "FAIL",
            "doors_found": len(nearby),
            "doors_passed": doors_passed,
            "doors_failed": doors_failed,
            "door_details": door_details,
            "video": f"validate_doors_cam_{cam_key}.mp4",
        }

        print(f"  Camera {cam_key}: {doors_passed}/{len(nearby)} passed", flush=True)

    # --- Global door summary ---
    total_doors = len(all_doors)
    total_passed = sum(
        1 for p in all_doors
        if abs(final_angles.get(p[0], 0) - initial_angles.get(p[0], 0)) > PASS_THRESHOLD
    )
    total_failed = total_doors - total_passed

    # --- Save results ---
    results = {
        "test": "door_physics",
        "usd_file": os.path.basename(usd_path),
        "global_summary": {
            "total_doors": total_doors,
            "total_passed": total_passed,
            "total_failed": total_failed,
        },
        "cameras": camera_results,
        "overall_passed": overall_pass,
        "runtime_seconds": round(time.time() - t_start, 1),
    }

    out_dir = os.path.join(_project_dir, "logs", "validation")
    os.makedirs(out_dir, exist_ok=True)
    out_path = os.path.join(out_dir, "door_physics_results.json")
    with open(out_path, "w") as f:
        json.dump(results, f, indent=2)

    print(f"\n[door_physics] Global: {total_passed}/{total_doors} doors passed", flush=True)
    print(f"[door_physics] Results -> {out_path}", flush=True)
    print(f"[door_physics] OVERALL: {'PASS' if overall_pass else 'FAIL'}", flush=True)

    simulation_app.close()


if __name__ == "__main__":
    main()
