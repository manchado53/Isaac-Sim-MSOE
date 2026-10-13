#!/usr/bin/env python3
"""
Door Close-Up Demo — Records a single door opening and closing.

Picks a known-good door (by name or highest delta), places cameras on
all 4 sides of the door, and records from each angle. All coordinates
in feet (Revit stage units, metersPerUnit=0.3048).

Usage:
    sbatch scripts/slurm/submit_isaac_job.sh simulations/environment/door_closeup_demo.py
"""
import math
import os
import sys

os.environ["OMNI_KIT_ACCEPT_EULA"] = "YES"

from isaacsim import SimulationApp
simulation_app = SimulationApp({"headless": True})

import numpy as np
from pxr import Gf, Sdf, Usd, UsdGeom, UsdLux, UsdPhysics
import isaacsim.core.utils.numpy.rotations as rot_utils

from isaacsim.core.api import World
from isaacsim.sensors.camera import Camera

_script_dir = os.path.dirname(os.path.abspath(__file__))
_project_dir = os.path.abspath(os.path.join(_script_dir, '..', '..'))
sys.path.insert(0, _project_dir)

from sim_utils.video_capture import VideoRecorder
from sim_utils.building_physics import set_drive_target

# Door to showcase — set to None to auto-pick the best one
PREFERRED_DOOR = "Doors_9817744"

CAM_DISTANCE = 8.0    # feet from door center
CAM_HEIGHT_OFFSET = 1.0  # feet above door center


def look_at_orientation(cam_pos, target):
    dx = target[0] - cam_pos[0]
    dy = target[1] - cam_pos[1]
    dz = target[2] - cam_pos[2]
    yaw = math.degrees(math.atan2(dy, dx))
    pitch = math.degrees(math.atan2(-dz, math.sqrt(dx**2 + dy**2)))
    return rot_utils.euler_angles_to_quats(
        np.array([0, pitch, yaw]), degrees=True
    )


def find_doors_with_drive(stage):
    """Find all doors that have a HingeJoint with DriveAPI."""
    doors = []
    for prim in stage.Traverse():
        if prim.GetName() == "HingeJoint":
            door_prim = prim.GetParent()
            if not door_prim or not door_prim.HasAPI(UsdPhysics.RigidBodyAPI):
                continue
            drive = UsdPhysics.DriveAPI.Get(prim, "angular")
            if not drive:
                continue

            bbox_cache = UsdGeom.BBoxCache(
                Usd.TimeCode.Default(), [UsdGeom.Tokens.default_]
            )
            bbox = bbox_cache.ComputeWorldBound(door_prim)
            rng = bbox.ComputeAlignedRange()
            if rng.IsEmpty():
                continue

            bmin, bmax = rng.GetMin(), rng.GetMax()
            center = np.array([
                (bmin[0] + bmax[0]) / 2.0,
                (bmin[1] + bmax[1]) / 2.0,
                (bmin[2] + bmax[2]) / 2.0,
            ])

            doors.append({
                "door_prim": door_prim,
                "joint_prim": prim,
                "drive": drive,
                "center": center,
                "bmin": np.array([bmin[0], bmin[1], bmin[2]]),
                "bmax": np.array([bmax[0], bmax[1], bmax[2]]),
                "name": door_prim.GetName(),
            })
    return doors


def main():
    world = World()
    stage = world.stage

    usd_path = sys.argv[1] if len(sys.argv) > 1 else os.path.join(
        _project_dir, 'assets', 'environments', 'msoe_building',
        'MSOEBuildingV4_physics.usd'
    )
    print(f"Loading: {usd_path}", flush=True)

    building_prim = stage.DefinePrim("/World/Building", "Xform")
    building_prim.GetReferences().AddReference(
        usd_path, primPath=Sdf.Path("/World/Building")
    )

    # Freeze ALL doors
    for prim in stage.Traverse():
        if prim.HasAPI(UsdPhysics.RigidBodyAPI):
            UsdPhysics.RigidBodyAPI(prim).CreateKinematicEnabledAttr(True)

    doors = find_doors_with_drive(stage)
    print(f"Found {len(doors)} doors", flush=True)

    # Pick door
    door = None
    if PREFERRED_DOOR:
        for d in doors:
            if d['name'] == PREFERRED_DOOR:
                door = d
                break
    if door is None:
        # Fallback: pick largest
        doors.sort(key=lambda d: (d['bmax'][2] - d['bmin'][2]) *
                   max(d['bmax'][0] - d['bmin'][0], d['bmax'][1] - d['bmin'][1]),
                   reverse=True)
        door = doors[0]

    print(f"\nShowcase door: {door['name']}", flush=True)
    print(f"  Center: ({door['center'][0]:.1f}, {door['center'][1]:.1f}, {door['center'][2]:.1f})",
          flush=True)

    # Unfreeze this door
    UsdPhysics.RigidBodyAPI(door['door_prim']).CreateKinematicEnabledAttr(False)

    # --- Lighting ---
    distant = UsdLux.DistantLight.Define(stage, "/World/SunLight")
    distant.CreateIntensityAttr(15000)
    distant.CreateAngleAttr(1.0)
    xf = UsdGeom.Xformable(distant.GetPrim())
    xf.AddRotateXYZOp().Set(Gf.Vec3f(-45, 30, 0))

    dome = UsdLux.DomeLight.Define(stage, "/World/SkyDome")
    dome.CreateIntensityAttr(5000)

    # Two sphere lights on opposite sides of the door
    for i, offset in enumerate([(8, 8), (-8, -8)]):
        sl = UsdLux.SphereLight.Define(stage, f"/World/DoorLight_{i}")
        sl.CreateIntensityAttr(60000)
        sl.CreateRadiusAttr(0.3)
        UsdGeom.Xformable(sl.GetPrim()).AddTranslateOp().Set(Gf.Vec3f(
            float(door['center'][0] + offset[0]),
            float(door['center'][1] + offset[1]),
            float(door['center'][2] + 2.0),
        ))

    # --- Single camera on the far side of the door (through the doorway) ---
    # Small +Y to clear the wall, offset in X for a diagonal view, raised slightly
    door_cx, door_cy, door_cz = door['center']
    cam_pos = np.array([door_cx + 4.0, door_cy + 5.0, door_cz + 2.0])
    cam_target = door['center']
    orient = look_at_orientation(cam_pos, cam_target)

    cam = Camera(
        prim_path="/World/DoorCam",
        position=cam_pos,
        frequency=30,
        resolution=(1920, 1080),
        orientation=orient,
    )
    cam_prim = stage.GetPrimAtPath("/World/DoorCam")
    UsdGeom.Camera(cam_prim).CreateFocalLengthAttr(24.0)
    print(f"  Camera: ({cam_pos[0]:.1f}, {cam_pos[1]:.1f}, {cam_pos[2]:.1f})", flush=True)

    world.reset()
    cam.initialize()

    # Warm up
    print("Warming up (60 frames)...", flush=True)
    for _ in range(60):
        world.step(render=True)

    # --- Record door opening and closing ---
    fps = 30
    phases = [
        ("closed",  1 * fps,   0.0),
        ("opening", 4 * fps,  90.0),
        ("open",    2 * fps,  90.0),
        ("closing", 4 * fps,   0.0),
        ("closed",  2 * fps,   0.0),
    ]

    total_frames = sum(f for _, f, _ in phases)
    print(f"Recording {total_frames} frames...", flush=True)

    with VideoRecorder("door_swing_demo", fps=fps) as recorder:
        frame = 0
        for phase_name, phase_frames, target_angle in phases:
            set_drive_target(door['drive'], target_angle)
            print(f"  Phase '{phase_name}': target={target_angle} deg", flush=True)

            for i in range(phase_frames):
                world.step(render=True)
                rgba = cam.get_rgba()
                if rgba is not None and frame > 0:
                    recorder.capture_rgba(rgba)
                frame += 1

    print("Done! Video saved to videos/door_swing_demo.mp4", flush=True)
    simulation_app.close()


if __name__ == "__main__":
    main()
