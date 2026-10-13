#!/usr/bin/env python3
"""
Building Door Drive Demo — Innovation Center

Loads the physics-ready Innovation Center USD (with DriveAPI already applied),
finds a door near the building exterior, and records it opening/closing
from an exterior camera viewpoint.

Prerequisites:
  - Run setup_innovation_center_physics.py first to generate the physics USD

Output:
  - videos/environment/building_door_demo.mp4  (10s @ 30fps)

Run on Rosie:
    sbatch scripts/slurm/submit_isaac_job.sh simulations/environment/building_door_demo.py
"""

import os
import sys
import math

os.environ["OMNI_KIT_ACCEPT_EULA"] = "YES"

from isaacsim import SimulationApp

simulation_app = SimulationApp({"headless": True})

import numpy as np
import omni.usd
from pxr import Usd, UsdGeom, UsdPhysics, UsdLux, Gf, Sdf
import isaacsim.core.utils.numpy.rotations as rot_utils
from isaacsim.core.api import World
from isaacsim.sensors.camera import Camera

_script_dir = os.path.dirname(os.path.abspath(__file__))
_project_dir = os.path.abspath(os.path.join(_script_dir, '..', '..'))
sys.path.insert(0, _project_dir)

from sim_utils.video_capture import VideoRecorder
from sim_utils.building_physics import set_drive_target


def look_at_orientation(cam_pos, target):
    """Compute camera orientation quaternion to look from cam_pos at target."""
    dx = target[0] - cam_pos[0]
    dy = target[1] - cam_pos[1]
    dz = target[2] - cam_pos[2]
    yaw = math.degrees(math.atan2(dy, dx))
    pitch = math.degrees(math.atan2(-dz, math.sqrt(dx**2 + dy**2)))
    return rot_utils.euler_angles_to_quats(
        np.array([0, pitch, yaw]), degrees=True
    )


def find_driven_doors(stage):
    """Find all revolute joints with angular DriveAPI.

    Returns list of (path_str, joint_prim) tuples.
    """
    candidates = []
    for prim in stage.Traverse():
        if not prim.IsA(UsdPhysics.RevoluteJoint):
            continue
        path_str = str(prim.GetPath())
        if "Door" not in path_str:
            continue
        drive = UsdPhysics.DriveAPI.Get(prim, "angular")
        if not drive or not drive.GetStiffnessAttr().HasValue():
            continue
        candidates.append((path_str, prim))

    print(f"  Found {len(candidates)} joints with DriveAPI", flush=True)
    return candidates


def get_door_prim(stage, joint_prim):
    """Get the door body prim (body1) from a joint."""
    joint = UsdPhysics.RevoluteJoint(joint_prim)
    targets = joint.GetBody1Rel().GetTargets()
    if targets:
        return stage.GetPrimAtPath(targets[0])
    return None


def get_bbox_center(stage, prim):
    """Get world-space bounding box center."""
    bbox_cache = UsdGeom.BBoxCache(Usd.TimeCode.Default(), ["default", "render"])
    bbox = bbox_cache.ComputeWorldBound(prim)
    rng = bbox.ComputeAlignedRange()
    if rng.IsEmpty():
        return None
    return (rng.GetMin() + rng.GetMax()) / 2.0


def find_best_exterior_door(stage, candidates, bldg_bbox_min, bldg_bbox_max):
    """Pick the door closest to any building edge — most likely to be visible
    from outside."""
    best = None
    best_dist = float('inf')
    for path_str, joint_prim in candidates:
        door_prim = get_door_prim(stage, joint_prim)
        if door_prim is None:
            continue
        center = get_bbox_center(stage, door_prim)
        if center is None:
            continue
        cx, cy = float(center[0]), float(center[1])
        edge_dist = min(
            abs(cx - float(bldg_bbox_min[0])),
            abs(cx - float(bldg_bbox_max[0])),
            abs(cy - float(bldg_bbox_min[1])),
            abs(cy - float(bldg_bbox_max[1])),
        )
        if edge_dist < best_dist:
            best_dist = edge_dist
            best = (path_str, joint_prim, door_prim, center)

    return best, best_dist


def main():
    print("=" * 60, flush=True)
    print("Building Door Drive Demo", flush=True)
    print("=" * 60, flush=True)

    world = World(stage_units_in_meters=1.0)
    stage = omni.usd.get_context().get_stage()

    # Load the physics-ready building USD
    physics_usd = os.path.join(
        _project_dir, "assets", "environments", "msoe_building",
        "innovation_center_physics.usd"
    )
    if not os.path.exists(physics_usd):
        print(f"ERROR: Physics USD not found: {physics_usd}", flush=True)
        simulation_app.close()
        return

    print(f"Loading: {physics_usd}", flush=True)
    building_prim = stage.DefinePrim("/World/Building", "Xform")
    building_prim.GetReferences().AddReference(
        physics_usd, primPath=Sdf.Path("/World/Building")
    )
    print("Building loaded.", flush=True)

    # Get building bounding box
    building_prim = stage.GetPrimAtPath("/World/Building")
    bbox_cache = UsdGeom.BBoxCache(Usd.TimeCode.Default(), [UsdGeom.Tokens.default_])
    bbox = bbox_cache.ComputeWorldBound(building_prim)
    rng = bbox.ComputeAlignedRange()
    bldg_min = rng.GetMin()
    bldg_max = rng.GetMax()
    bldg_center = (bldg_min + bldg_max) / 2.0
    bldg_extent = bldg_max - bldg_min
    print(f"Building bbox: min=({bldg_min[0]:.0f},{bldg_min[1]:.0f},{bldg_min[2]:.0f}) "
          f"max=({bldg_max[0]:.0f},{bldg_max[1]:.0f},{bldg_max[2]:.0f})", flush=True)

    # Find all driven doors and pick the one closest to building edge
    candidates = find_driven_doors(stage)
    if not candidates:
        print("ERROR: No driven doors found.", flush=True)
        simulation_app.close()
        return

    result, edge_dist = find_best_exterior_door(stage, candidates, bldg_min, bldg_max)
    if result is None:
        print("ERROR: No door with valid bbox found.", flush=True)
        simulation_app.close()
        return

    path_str, joint_prim, door_prim, door_center = result
    drive = UsdPhysics.DriveAPI.Get(joint_prim, "angular")
    cx, cy, cz = float(door_center[0]), float(door_center[1]), float(door_center[2])
    print(f"Selected door: {path_str}", flush=True)
    print(f"Door center: ({cx:.1f}, {cy:.1f}, {cz:.1f})", flush=True)
    print(f"Distance from nearest edge: {edge_dist:.1f} ft", flush=True)

    # --- Camera: exterior view looking at the door ---
    # Determine which edge the door is nearest to, place camera outside that edge
    edge_dists = {
        "x_min": abs(cx - float(bldg_min[0])),
        "x_max": abs(cx - float(bldg_max[0])),
        "y_min": abs(cy - float(bldg_min[1])),
        "y_max": abs(cy - float(bldg_max[1])),
    }
    nearest_edge = min(edge_dists, key=edge_dists.get)

    # Place camera well outside the building, 30 feet beyond the edge
    margin = 30.0
    if nearest_edge == "x_min":
        cam_pos = np.array([float(bldg_min[0]) - margin, cy, cz + 5.0])
    elif nearest_edge == "x_max":
        cam_pos = np.array([float(bldg_max[0]) + margin, cy, cz + 5.0])
    elif nearest_edge == "y_min":
        cam_pos = np.array([cx, float(bldg_min[1]) - margin, cz + 5.0])
    else:
        cam_pos = np.array([cx, float(bldg_max[1]) + margin, cz + 5.0])

    target = np.array([cx, cy, cz])
    orient = look_at_orientation(cam_pos, target)

    print(f"Nearest edge: {nearest_edge}", flush=True)
    print(f"Camera pos: ({cam_pos[0]:.1f}, {cam_pos[1]:.1f}, {cam_pos[2]:.1f})", flush=True)
    print(f"Looking at door: ({cx:.1f}, {cy:.1f}, {cz:.1f})", flush=True)

    camera = Camera(
        prim_path="/World/DoorDemoCamera",
        position=cam_pos,
        frequency=30,
        resolution=(1280, 720),
        orientation=orient,
    )

    # Exterior lighting (same as engineering_physics_demo)
    distant = UsdLux.DistantLight.Define(stage, Sdf.Path("/World/Sun"))
    distant.CreateIntensityAttr(3000)
    xf = UsdGeom.Xformable(distant.GetPrim())
    xf.AddRotateXYZOp().Set(Gf.Vec3f(-45, 30, 0))
    UsdLux.DomeLight.Define(stage, Sdf.Path("/World/DomeLight")).CreateIntensityAttr(500)

    world.reset()
    camera.initialize()

    # Warm up rendering (extra frames for shader compilation)
    for _ in range(30):
        world.step(render=True)

    # Diagnostic: check first frame pixel values
    test_rgba = camera.get_rgba()
    if test_rgba is not None:
        print(f"Frame diagnostic: shape={test_rgba.shape}, "
              f"min={test_rgba.min()}, max={test_rgba.max()}, "
              f"mean={test_rgba.mean():.1f}", flush=True)
    else:
        print("WARNING: camera returned None for first frame", flush=True)

    # --- Simulation: 300 frames (10s @ 30fps) ---
    total_frames = 300
    open_start = 60
    open_end = 180
    close_start = 240

    print(f"Recording {total_frames} frames...", flush=True)

    with VideoRecorder("environment/building_door_demo", fps=30) as recorder:
        for i in range(total_frames):
            if i < open_start:
                set_drive_target(drive, 0.0)
            elif i < open_end:
                t = (i - open_start) / (open_end - open_start)
                set_drive_target(drive, t * 80.0)
            elif i < close_start:
                set_drive_target(drive, 80.0)
            else:
                t = (i - close_start) / (total_frames - close_start)
                set_drive_target(drive, 80.0 * (1.0 - t))

            world.step(render=True)

            rgba = camera.get_rgba()
            if rgba is not None and i > 0:
                recorder.capture_rgba(rgba)

            if i % 60 == 0:
                print(f"  Frame {i}/{total_frames}", flush=True)

    print("Done.", flush=True)
    simulation_app.close()


if __name__ == "__main__":
    main()
