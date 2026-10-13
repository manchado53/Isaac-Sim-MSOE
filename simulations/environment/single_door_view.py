#!/usr/bin/env python3
"""
Single Door View — Clear close-up of one hinged door from the Revit USD.

Loads the raw Revit-exported building USD, finds the first hinged single
door with visible mesh geometry, and records a slow-orbit video around it
with strong interior lighting.

Usage:
    sbatch scripts/slurm/submit_isaac_job.sh simulations/environment/single_door_view.py
"""
import math
import os
import sys

os.environ["OMNI_KIT_ACCEPT_EULA"] = "YES"

from isaacsim import SimulationApp
simulation_app = SimulationApp({"headless": True})

import numpy as np
from pxr import Gf, Sdf, Usd, UsdGeom, UsdLux
import isaacsim.core.utils.numpy.rotations as rot_utils
from isaacsim.core.api import World
from isaacsim.core.utils.stage import add_reference_to_stage
from isaacsim.sensors.camera import Camera

_script_dir = os.path.dirname(os.path.abspath(__file__))
_project_dir = os.path.abspath(os.path.join(_script_dir, '..', '..'))
sys.path.insert(0, _project_dir)

from sim_utils.video_capture import VideoRecorder

FPS = 30
VIDEO_SECONDS = 8
TOTAL_FRAMES = FPS * VIDEO_SECONDS
ORBIT_DEGREES = 90


def look_at_quat(cam_pos, target):
    """Compute quaternion orientation looking from cam_pos toward target."""
    dx = target[0] - cam_pos[0]
    dy = target[1] - cam_pos[1]
    dz = target[2] - cam_pos[2]
    yaw = math.degrees(math.atan2(dy, dx))
    pitch = math.degrees(math.atan2(-dz, math.sqrt(dx**2 + dy**2)))
    return rot_utils.euler_angles_to_quats(
        np.array([0, pitch, yaw]), degrees=True
    )


def find_hinged_door(stage):
    """Find the first hinged single door instance with mesh geometry.

    Searches under Geometry/Doors for leaf Xform prims named 'Doors_*'
    that have Mesh children (actual renderable geometry).
    """
    bbox_cache = UsdGeom.BBoxCache(
        Usd.TimeCode.Default(), [UsdGeom.Tokens.default_]
    )

    for prim in stage.Traverse():
        path = str(prim.GetPath())
        if "FamilyData" in path:
            continue
        if "Doors" not in path:
            continue
        if prim.GetTypeName() != "Xform":
            continue
        if not prim.GetName().startswith("Doors_"):
            continue
        # Must have Mesh children (actual geometry, not just a grouping node)
        has_mesh = any(c.GetTypeName() == "Mesh" for c in prim.GetChildren())
        if not has_mesh:
            continue

        bbox = bbox_cache.ComputeWorldBound(prim)
        rng = bbox.ComputeAlignedRange()
        if rng.IsEmpty():
            continue

        bmin, bmax = rng.GetMin(), rng.GetMax()
        center = np.array([(bmin[i] + bmax[i]) / 2.0 for i in range(3)])
        size = np.array([bmax[i] - bmin[i] for i in range(3)])

        # Classify from ancestor folder name
        door_type = "HINGED"
        for ancestor in [prim.GetParent(), prim.GetParent().GetParent()]:
            if ancestor:
                aname = ancestor.GetName()
                if "Type_F" in aname:
                    door_type = "HINGED_SINGLE"
                    break

        return {
            "name": prim.GetName(),
            "path": path,
            "center": center,
            "size": size,
            "type": door_type,
        }
    return None


def main():
    world = World()
    stage = world.stage

    usd_path = os.path.join(
        _project_dir, "assets", "environments", "MSOEBuildingTake2 1.usd"
    )
    print(f"Loading: {usd_path}", flush=True)

    if not os.path.exists(usd_path):
        print(f"ERROR: Not found: {usd_path}")
        simulation_app.close()
        sys.exit(1)

    add_reference_to_stage(usd_path, prim_path="/World/Building")

    door = find_hinged_door(stage)
    if not door:
        print("ERROR: No hinged door found in the USD!")
        simulation_app.close()
        sys.exit(1)

    print(f"Door: {door['name']} ({door['type']})", flush=True)
    print(f"  Path:   {door['path']}", flush=True)
    print(f"  Center: ({door['center'][0]:.2f}, {door['center'][1]:.2f}, {door['center'][2]:.2f})",
          flush=True)
    print(f"  Size:   ({door['size'][0]:.2f}, {door['size'][1]:.2f}, {door['size'][2]:.2f})",
          flush=True)

    cx, cy, cz = door['center']
    door_height = max(door['size'])
    cam_dist = door_height * 2.5
    cam_up = door_height * 0.3

    print(f"  Camera distance: {cam_dist:.2f}, height offset: {cam_up:.2f}", flush=True)

    # --- Lighting ---
    sun = UsdLux.DistantLight.Define(stage, "/World/Sun")
    sun.CreateIntensityAttr(15000)
    sun.CreateAngleAttr(1.0)
    UsdGeom.Xformable(sun.GetPrim()).AddRotateXYZOp().Set(Gf.Vec3f(-45, 30, 0))

    dome = UsdLux.DomeLight.Define(stage, "/World/Sky")
    dome.CreateIntensityAttr(5000)

    for i, (dx, dy) in enumerate([(1, 1), (-1, -1)]):
        sl = UsdLux.SphereLight.Define(stage, f"/World/Fill_{i}")
        sl.CreateIntensityAttr(80000)
        sl.CreateRadiusAttr(door_height * 0.05)
        UsdGeom.Xformable(sl.GetPrim()).AddTranslateOp().Set(
            Gf.Vec3d(
                cx + dx * cam_dist * 0.5,
                cy + dy * cam_dist * 0.5,
                cz + cam_up + door_height * 0.3,
            )
        )

    # --- Camera ---
    start_angle = math.radians(-60)
    init_pos = np.array([
        cx + cam_dist * math.cos(start_angle),
        cy + cam_dist * math.sin(start_angle),
        cz + cam_up,
    ])
    target = np.array([cx, cy, cz])
    orient = look_at_quat(init_pos, target)

    camera = Camera(
        prim_path="/World/DoorCam",
        position=init_pos,
        frequency=FPS,
        resolution=(1920, 1080),
        orientation=orient,
    )
    UsdGeom.Camera(
        stage.GetPrimAtPath("/World/DoorCam")
    ).CreateFocalLengthAttr(24.0)

    world.reset()
    camera.initialize()

    # Warm up renderer
    print("Warming up (30 frames)...", flush=True)
    for _ in range(30):
        world.step(render=True)

    # --- Record slow orbit around the door ---
    print(f"Recording {VIDEO_SECONDS}s orbit ({TOTAL_FRAMES} frames)...", flush=True)

    with VideoRecorder("single_door_view", fps=FPS) as rec:
        for f in range(TOTAL_FRAMES):
            t = f / TOTAL_FRAMES
            angle = start_angle + math.radians(ORBIT_DEGREES) * t
            pos = np.array([
                cx + cam_dist * math.cos(angle),
                cy + cam_dist * math.sin(angle),
                cz + cam_up,
            ])
            camera.set_world_pose(pos, look_at_quat(pos, target))
            world.step(render=True)
            rgba = camera.get_rgba()
            if rgba is not None:
                rec.capture_rgba(rgba)

    print("Done!", flush=True)
    simulation_app.close()


if __name__ == "__main__":
    main()
