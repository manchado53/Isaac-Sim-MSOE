#!/usr/bin/env python3
"""
Validation: Building Stats

Traverses the USD and collects quantitative statistics:
  - Total prims, colliders, rigid bodies, joints, driven joints
  - Mesh count, floor elements, door elements
  - Building bounding box dimensions
  - Per-camera screenshot (non-black check)

Usage:
    sbatch submit_isaac_job.sh simulations/environment/validate_building_stats.py <usd_path>

Output:
    logs/validation/building_stats_results.json
    videos/validation/validate_stats_cam_*.png
"""

import os
import sys
import json
import time

os.environ["OMNI_KIT_ACCEPT_EULA"] = "YES"

from isaacsim import SimulationApp

simulation_app = SimulationApp({"headless": True})

import numpy as np
import omni.usd
from pxr import Usd, UsdGeom, UsdPhysics, Sdf

_script_dir = os.path.dirname(os.path.abspath(__file__))
_project_dir = os.path.abspath(os.path.join(_script_dir, '..', '..'))
sys.path.insert(0, _project_dir)

from isaacsim.core.api import World
from isaacsim.sensors.camera import Camera
from sim_utils.test_cameras import (
    TEST_CAMERAS, create_test_camera, add_interior_lighting, look_at_orientation,
    set_focal_length_on_prim,
)
from sim_utils.building_physics import (
    setup_physics_scene, iter_building_elements,
    get_physics_scene_stats, get_solver_residuals, get_compute_cost,
    enable_residual_reporting,
)


def save_screenshot(camera, output_path):
    """Save a single RGBA frame as PNG. Returns True if non-black."""
    rgba = camera.get_rgba()
    if rgba is None:
        return False
    try:
        from PIL import Image
        img = Image.fromarray(rgba[:, :, :3])
        img.save(output_path)
    except ImportError:
        # Fallback: save raw numpy
        np.save(output_path.replace(".png", ".npy"), rgba)
    # Non-black check: mean brightness > 5
    return float(np.mean(rgba[:, :, :3])) > 5.0


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

    print(f"[building_stats] Loading: {usd_path}", flush=True)

    # --- Scene setup ---
    world = World(stage_units_in_meters=1.0)
    stage = omni.usd.get_context().get_stage()

    building_prim = stage.DefinePrim("/World/Building", "Xform")
    building_prim.GetReferences().AddReference(
        usd_path, primPath=Sdf.Path("/World/Building")
    )

    setup_physics_scene(stage)
    add_interior_lighting(stage)

    # --- Count stats ---
    print("[building_stats] Counting prims...", flush=True)
    total_prims = 0
    collider_count = 0
    rigid_body_count = 0
    revolute_joint_count = 0
    driven_joint_count = 0
    mesh_count = 0

    for prim in stage.Traverse():
        total_prims += 1
        if prim.HasAPI(UsdPhysics.CollisionAPI):
            collider_count += 1
        if prim.HasAPI(UsdPhysics.RigidBodyAPI):
            rigid_body_count += 1
        if prim.IsA(UsdPhysics.RevoluteJoint):
            revolute_joint_count += 1
            drive = UsdPhysics.DriveAPI.Get(prim, "angular")
            if drive and drive.GetStiffnessAttr().HasValue():
                driven_joint_count += 1
        if prim.IsA(UsdGeom.Mesh):
            mesh_count += 1

    # Count building elements by category
    building_prim = stage.GetPrimAtPath("/World/Building")
    floor_element_count = 0
    door_element_count = 0
    wall_element_count = 0
    for _level, category_name, _element in iter_building_elements(stage, building_prim):
        if category_name == "Floors":
            floor_element_count += 1
        elif category_name and "Door" in category_name:
            door_element_count += 1
        elif category_name and "Wall" in category_name:
            wall_element_count += 1

    # Building bounding box
    bbox_cache = UsdGeom.BBoxCache(
        Usd.TimeCode.Default(), [UsdGeom.Tokens.default_]
    )
    bbox = bbox_cache.ComputeWorldBound(building_prim)
    box_range = bbox.ComputeAlignedRange()
    bb_min = box_range.GetMin()
    bb_max = box_range.GetMax()
    dims = [bb_max[i] - bb_min[i] for i in range(3)]

    stats = {
        "total_prims": total_prims,
        "collider_count": collider_count,
        "rigid_body_count": rigid_body_count,
        "revolute_joint_count": revolute_joint_count,
        "driven_joint_count": driven_joint_count,
        "mesh_count": mesh_count,
        "floor_element_count": floor_element_count,
        "door_element_count": door_element_count,
        "wall_element_count": wall_element_count,
        "building_bbox": {
            "min": [float(bb_min[i]) for i in range(3)],
            "max": [float(bb_max[i]) for i in range(3)],
        },
        "building_dimensions_ft": [round(d, 1) for d in dims],
    }

    print(f"[building_stats] Prims: {total_prims}", flush=True)
    print(f"[building_stats] Colliders: {collider_count}", flush=True)
    print(f"[building_stats] Rigid bodies: {rigid_body_count}", flush=True)
    print(f"[building_stats] Revolute joints: {revolute_joint_count}", flush=True)
    print(f"[building_stats] Driven joints: {driven_joint_count}", flush=True)
    print(f"[building_stats] Meshes: {mesh_count}", flush=True)
    print(f"[building_stats] Floor elements: {floor_element_count}", flush=True)
    print(f"[building_stats] Door elements: {door_element_count}", flush=True)
    print(f"[building_stats] Dimensions: {dims[0]:.1f} x {dims[1]:.1f} x {dims[2]:.1f} ft", flush=True)

    # --- Camera screenshots ---
    print("[building_stats] Creating cameras...", flush=True)
    cameras = {}
    for key, cam_dict in TEST_CAMERAS.items():
        cameras[key] = create_test_camera(key, cam_dict, "/World/StatsCam", stage=stage)

    # Enable solver residual reporting (must be before world.reset)
    enable_residual_reporting(stage)

    world.reset()
    for cam in cameras.values():
        cam.initialize()
        set_focal_length_on_prim(stage, cam)

    # Warmup
    for _ in range(30):
        world.step(render=True)

    # --- Collect native PhysX metrics (requires running simulation) ---
    print("[building_stats] Collecting PhysX scene stats...", flush=True)
    stats["physics_scene_stats"] = get_physics_scene_stats(stage)
    stats["solver_residuals"] = get_solver_residuals(stage)
    stats["compute_cost"] = get_compute_cost(stage)

    if stats["physics_scene_stats"]:
        pss = stats["physics_scene_stats"]
        print(f"[building_stats] PhysX: {pss.get('nb_static_rigids', '?')} static, "
              f"{pss.get('nb_dynamic_rigids', '?')} dynamic, "
              f"{pss.get('nb_active_constraints', '?')} active constraints", flush=True)
        gpu_mb = stats["compute_cost"].get("gpu_mem_total_mb", "?")
        print(f"[building_stats] GPU memory: {gpu_mb} MB", flush=True)
    if stats["solver_residuals"]:
        sr = stats["solver_residuals"]
        print(f"[building_stats] Solver residuals: RMS pos={sr.get('rms_position', '?')}, "
              f"max pos={sr.get('max_position', '?')}", flush=True)

    # Capture per-camera screenshots
    screenshot_dir = os.path.join(_project_dir, "videos", "validation")
    os.makedirs(screenshot_dir, exist_ok=True)

    camera_results = {}
    for key, cam in cameras.items():
        # Render a few frames to let camera settle
        for _ in range(10):
            world.step(render=True)

        png_path = os.path.join(screenshot_dir, f"validate_stats_cam_{key}.png")
        non_black = save_screenshot(cam, png_path)
        camera_results[key] = {
            "screenshot": f"validate_stats_cam_{key}.png",
            "non_black": non_black,
            "room": TEST_CAMERAS[key]["room"],
        }
        print(f"[building_stats] Camera {key} ({TEST_CAMERAS[key]['room']}): "
              f"{'OK' if non_black else 'BLACK'}", flush=True)

    # --- Assemble results ---
    all_screenshots_ok = all(c["non_black"] for c in camera_results.values())
    has_physics = collider_count > 0 and rigid_body_count > 0

    results = {
        "test": "building_stats",
        "usd_file": os.path.basename(usd_path),
        "stats": stats,
        "cameras": camera_results,
        "overall_passed": has_physics and all_screenshots_ok,
        "runtime_seconds": round(time.time() - t_start, 1),
    }

    out_dir = os.path.join(_project_dir, "logs", "validation")
    os.makedirs(out_dir, exist_ok=True)
    out_path = os.path.join(out_dir, "building_stats_results.json")
    with open(out_path, "w") as f:
        json.dump(results, f, indent=2)

    print(f"\n[building_stats] Results -> {out_path}", flush=True)
    print(f"[building_stats] OVERALL: {'PASS' if results['overall_passed'] else 'FAIL'}", flush=True)

    # Skip simulation_app.close() — Isaac Sim's OmniGraph cleanup can segfault
    os._exit(0)


if __name__ == "__main__":
    main()
