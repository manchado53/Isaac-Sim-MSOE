#!/usr/bin/env python3
"""Render exterior orbit of the MSOE engineering-innovation building.

All coordinates are in feet (the building's native stage units from Revit,
metersPerUnit=0.3048). Uses Isaac Sim Camera sensor with get_rgba() for
capture — the same proven pattern as building_physics_demo.py.
"""
import math
import os
import sys

os.environ["OMNI_KIT_ACCEPT_EULA"] = "YES"

from isaacsim import SimulationApp
simulation_app = SimulationApp({"headless": True})

from isaacsim.core.api import World
from isaacsim.sensors.camera import Camera
import isaacsim.core.utils.numpy.rotations as rot_utils
from pxr import Usd, UsdGeom, UsdPhysics, UsdLux, Gf, Sdf
import numpy as np

_script_dir = os.path.dirname(os.path.abspath(__file__))
_project_dir = os.path.abspath(os.path.join(_script_dir, '..', '..'))
sys.path.insert(0, _project_dir)

from sim_utils.video_capture import VideoRecorder


def look_at_orientation(cam_pos, target):
    """Compute quaternion orientation for a camera looking at target."""
    dx = target[0] - cam_pos[0]
    dy = target[1] - cam_pos[1]
    dz = target[2] - cam_pos[2]
    yaw = math.degrees(math.atan2(dy, dx))
    pitch = math.degrees(math.atan2(-dz, math.sqrt(dx**2 + dy**2)))
    return rot_utils.euler_angles_to_quats(
        np.array([0, pitch, yaw]), degrees=True
    )


def main():
    world = World()
    stage = world.stage

    # Load the physics USD
    usd_path = os.path.join(
        _project_dir, 'assets', 'environments',
        'msoe-engineering-innovation-USD_physics.usd'
    )
    print(f"Loading: {usd_path}", flush=True)

    building_prim = stage.DefinePrim("/World/Building", "Xform")
    building_prim.GetReferences().AddReference(
        usd_path, primPath=Sdf.Path("/World/Building")
    )

    # Freeze doors so they don't fly during the orbit
    frozen = 0
    for prim in stage.Traverse():
        if prim.HasAPI(UsdPhysics.RigidBodyAPI):
            UsdPhysics.RigidBodyAPI(prim).CreateKinematicEnabledAttr(True)
            frozen += 1
    print(f"Froze {frozen} dynamic bodies", flush=True)

    # Exterior lighting — bright sun + sky dome
    distant = UsdLux.DistantLight.Define(stage, "/World/SunLight")
    distant.CreateIntensityAttr(15000)
    distant.CreateAngleAttr(1.0)
    xf = UsdGeom.Xformable(distant.GetPrim())
    xf.AddRotateXYZOp().Set(Gf.Vec3f(-45, 30, 0))

    dome = UsdLux.DomeLight.Define(stage, "/World/SkyDome")
    dome.CreateIntensityAttr(5000)

    # Compute building bounding box (in feet)
    world.reset()
    world.step(render=True)

    bbox_cache = UsdGeom.BBoxCache(Usd.TimeCode.Default(), [UsdGeom.Tokens.default_])
    building = stage.GetPrimAtPath("/World/Building")
    bbox = bbox_cache.ComputeWorldBound(building)
    bbox_range = bbox.ComputeAlignedRange()
    bmin = bbox_range.GetMin()
    bmax = bbox_range.GetMax()

    bldg_center = np.array([
        (bmin[0] + bmax[0]) / 2.0,
        (bmin[1] + bmax[1]) / 2.0,
        (bmin[2] + bmax[2]) / 2.0,
    ])
    bldg_extent = np.array([
        bmax[0] - bmin[0],
        bmax[1] - bmin[1],
        bmax[2] - bmin[2],
    ])

    print(f"Building center: ({bldg_center[0]:.1f}, {bldg_center[1]:.1f}, {bldg_center[2]:.1f}) ft",
          flush=True)
    print(f"Building extent: ({bldg_extent[0]:.1f} x {bldg_extent[1]:.1f} x {bldg_extent[2]:.1f}) ft",
          flush=True)

    # Camera orbit parameters (all in feet)
    radius = max(bldg_extent[0], bldg_extent[1]) * 0.85
    cam_height = bldg_center[2] + bldg_extent[2] * 0.3

    # Create Isaac Sim Camera sensor (proven capture method)
    initial_pos = np.array([
        bldg_center[0] + radius,
        bldg_center[1],
        cam_height,
    ])
    initial_orient = look_at_orientation(initial_pos, bldg_center)

    camera = Camera(
        prim_path="/World/ExteriorCam",
        position=initial_pos,
        frequency=30,
        resolution=(1920, 1080),
        orientation=initial_orient,
    )
    # Wide-angle lens for exterior view
    cam_prim = stage.GetPrimAtPath("/World/ExteriorCam")
    UsdGeom.Camera(cam_prim).CreateFocalLengthAttr(14.0)

    world.reset()
    camera.initialize()

    # Warm up renderer
    print("Warming up (30 frames)...", flush=True)
    for _ in range(30):
        world.step(render=True)

    # Orbital recording — 360 degree orbit
    orbit_frames = 600  # 20s at 30fps
    print(f"Recording {orbit_frames} frames, radius={radius:.0f}ft, height={cam_height:.0f}ft",
          flush=True)

    with VideoRecorder("building_exterior_orbit", fps=30) as recorder:
        for i in range(orbit_frames):
            angle = 2 * math.pi * i / orbit_frames
            # Gentle height oscillation for cinematic effect
            height_offset = bldg_extent[2] * 0.1 * math.sin(2 * angle)

            cam_x = bldg_center[0] + radius * math.cos(angle)
            cam_y = bldg_center[1] + radius * math.sin(angle)
            cam_z = cam_height + height_offset

            cam_pos = np.array([cam_x, cam_y, cam_z])
            orient = look_at_orientation(cam_pos, bldg_center)
            camera.set_world_pose(position=cam_pos, orientation=orient)

            world.step(render=True)
            rgba = camera.get_rgba()
            if rgba is not None and i > 0:
                recorder.capture_rgba(rgba)

            if i % 120 == 0:
                print(f"  Frame {i}/{orbit_frames}  "
                      f"pos=({cam_x:.0f}, {cam_y:.0f}, {cam_z:.0f})", flush=True)

    print("Done!", flush=True)
    simulation_app.close()


if __name__ == "__main__":
    main()
