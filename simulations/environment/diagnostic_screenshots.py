#!/usr/bin/env python3
"""
Diagnostic Screenshots -- Before/After Camera Comparison

Loads a building USD and captures PNG screenshots from all 4 test camera
positions (G, 1, 2, 3).  Run once on the raw USD, once on the physics USD,
then use compare_screenshots.py to create side-by-side diffs.

Usage:
    sbatch scripts/slurm/submit_isaac_job.sh simulations/environment/diagnostic_screenshots.py \
        --input assets/environments/msoe_building/MSOEBuildingApril.usd --prefix before

    sbatch scripts/slurm/submit_isaac_job.sh simulations/environment/diagnostic_screenshots.py \
        --input assets/environments/msoe_building/MSOEBuildingApril_physics.usd --prefix after

Output:
    images/diagnostics/{prefix}_cam_G.png
    images/diagnostics/{prefix}_cam_1.png
    images/diagnostics/{prefix}_cam_2.png
    images/diagnostics/{prefix}_cam_3.png
"""

import argparse
import json
import os
import sys
import time

# Parse CLI args BEFORE SimulationApp (which takes ~40s and needs GPU).
parser = argparse.ArgumentParser(description="Capture diagnostic screenshots from test camera positions.")
parser.add_argument("--input", required=True, help="Path to building USD file")
parser.add_argument("--prefix", required=True, help="Output filename prefix (e.g. 'before' or 'after')")
parser.add_argument("--output-dir", default=None, help="Output directory (default: images/diagnostics/)")
args, _unknown = parser.parse_known_args()

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
from isaacsim.core.utils.stage import add_reference_to_stage
from sim_utils.test_cameras import (
    TEST_CAMERAS, create_test_camera, add_interior_lighting,
    set_focal_length_on_prim,
)
from sim_utils.building_physics import setup_physics_scene


def save_screenshot(camera, output_path):
    """Save a single RGBA frame as PNG. Returns mean brightness."""
    rgba = camera.get_rgba()
    if rgba is None:
        return 0.0
    try:
        from PIL import Image
        img = Image.fromarray(rgba[:, :, :3])
        img.save(output_path)
    except ImportError:
        np.save(output_path.replace(".png", ".npy"), rgba)
    return float(np.mean(rgba[:, :, :3]))


def count_door_prims(stage):
    """Quick traversal to count prims with 'Door' in the path."""
    door_prims = []
    for prim in stage.Traverse():
        path_str = str(prim.GetPath())
        if "Door" not in path_str and "door" not in path_str:
            continue
        if prim.IsA(UsdGeom.Mesh):
            door_prims.append(path_str)
    return door_prims


def count_physics_prims(stage):
    """Count colliders, rigid bodies, and joints."""
    colliders = 0
    rigid_bodies = 0
    joints = 0
    for prim in stage.Traverse():
        if prim.HasAPI(UsdPhysics.CollisionAPI):
            colliders += 1
        if prim.HasAPI(UsdPhysics.RigidBodyAPI):
            rigid_bodies += 1
        if prim.IsA(UsdPhysics.RevoluteJoint):
            joints += 1
    return colliders, rigid_bodies, joints


def main():
    t_start = time.time()

    input_usd = os.path.abspath(args.input)
    prefix = args.prefix
    output_dir = os.path.abspath(args.output_dir) if args.output_dir else os.path.join(
        _project_dir, "images", "diagnostics"
    )
    os.makedirs(output_dir, exist_ok=True)

    if not os.path.exists(input_usd):
        print(f"ERROR: USD not found: {input_usd}", flush=True)
        simulation_app.close()
        sys.exit(1)

    print(f"[diagnostic] Input:  {input_usd}", flush=True)
    print(f"[diagnostic] Prefix: {prefix}", flush=True)
    print(f"[diagnostic] Output: {output_dir}", flush=True)

    # --- Scene setup ---
    world = World(stage_units_in_meters=1.0)
    stage = omni.usd.get_context().get_stage()

    # Detect USD type: physics USDs (from stage.Export) have /World/Building
    # internally and need primPath extraction.  Raw Revit USDs don't.
    print(f"[diagnostic] Loading building...", flush=True)
    root_layer = Sdf.Layer.FindOrOpen(input_usd)
    has_world_building = root_layer and root_layer.GetPrimAtPath("/World/Building")
    if has_world_building:
        # Physics USD — extract the /World/Building subtree
        print(f"[diagnostic]   Detected physics USD (has /World/Building)", flush=True)
        building_prim = stage.DefinePrim("/World/Building", "Xform")
        building_prim.GetReferences().AddReference(
            input_usd, primPath=Sdf.Path("/World/Building")
        )
    else:
        # Raw Revit USD — reference the whole file
        print(f"[diagnostic]   Detected raw Revit USD", flush=True)
        add_reference_to_stage(input_usd, prim_path="/World/Building")
    print(f"[diagnostic] Building loaded.", flush=True)

    # Add physics scene (needed for renderer even if no physics)
    setup_physics_scene(stage)
    add_interior_lighting(stage)

    # --- Count door and physics prims ---
    print(f"[diagnostic] Counting prims...", flush=True)
    door_meshes = count_door_prims(stage)
    colliders, rigid_bodies, joints = count_physics_prims(stage)
    print(f"[diagnostic] Door meshes:   {len(door_meshes)}", flush=True)
    print(f"[diagnostic] Colliders:     {colliders}", flush=True)
    print(f"[diagnostic] Rigid bodies:  {rigid_bodies}", flush=True)
    print(f"[diagnostic] Joints:        {joints}", flush=True)

    # --- Create cameras ---
    print(f"[diagnostic] Creating cameras...", flush=True)
    cameras = {}
    for key, cam_dict in TEST_CAMERAS.items():
        cameras[key] = create_test_camera(key, cam_dict, "/World/DiagCam", stage=stage)

    world.reset()
    for cam in cameras.values():
        cam.initialize()
        set_focal_length_on_prim(stage, cam)

    # Warmup (30 frames for renderer stabilization)
    print(f"[diagnostic] Warming up (30 frames)...", flush=True)
    for _ in range(30):
        world.step(render=True)

    # --- Capture screenshots ---
    print(f"[diagnostic] Capturing screenshots...", flush=True)
    camera_results = {}
    for key, cam in cameras.items():
        # Settle frames per camera
        for _ in range(10):
            world.step(render=True)

        png_path = os.path.join(output_dir, f"{prefix}_cam_{key}.png")
        brightness = save_screenshot(cam, png_path)
        non_black = brightness > 5.0

        camera_results[key] = {
            "file": f"{prefix}_cam_{key}.png",
            "room": TEST_CAMERAS[key]["room"],
            "brightness": round(brightness, 1),
            "non_black": non_black,
        }
        status = "OK" if non_black else "BLACK"
        print(f"  Camera {key} ({TEST_CAMERAS[key]['room']}): "
              f"brightness={brightness:.1f} [{status}] -> {png_path}", flush=True)

    # --- Save summary JSON ---
    summary = {
        "prefix": prefix,
        "usd_file": os.path.basename(input_usd),
        "usd_path": input_usd,
        "door_mesh_count": len(door_meshes),
        "colliders": colliders,
        "rigid_bodies": rigid_bodies,
        "joints": joints,
        "cameras": camera_results,
        "runtime_seconds": round(time.time() - t_start, 1),
    }

    summary_path = os.path.join(output_dir, f"{prefix}_summary.json")
    with open(summary_path, "w") as f:
        json.dump(summary, f, indent=2)

    # Also write door mesh list for detailed comparison
    door_list_path = os.path.join(output_dir, f"{prefix}_door_prims.txt")
    with open(door_list_path, "w") as f:
        for p in sorted(door_meshes):
            f.write(p + "\n")

    print(f"\n[diagnostic] Summary -> {summary_path}", flush=True)
    print(f"[diagnostic] Door prim list -> {door_list_path}", flush=True)
    print(f"[diagnostic] Runtime: {time.time() - t_start:.1f}s", flush=True)
    print(f"[diagnostic] Done.", flush=True)

    os._exit(0)


if __name__ == "__main__":
    main()
