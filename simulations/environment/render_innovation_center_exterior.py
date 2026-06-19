#!/usr/bin/env python3
"""Render a single exterior hero shot of the Innovation Center.

Loads InnovationCenter.usd (metersPerUnit=1.0) and saves a 1920x1080 PNG
to images/innovation_center_exterior.png.
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
from pxr import Usd, UsdGeom, UsdLux, Gf, Sdf
import numpy as np
import imageio.v2 as imageio

_script_dir = os.path.dirname(os.path.abspath(__file__))
_project_dir = os.path.abspath(os.path.join(_script_dir, '..', '..'))
sys.path.insert(0, _project_dir)


def look_at_orientation(cam_pos, target):
    dx = target[0] - cam_pos[0]
    dy = target[1] - cam_pos[1]
    dz = target[2] - cam_pos[2]
    yaw = math.degrees(math.atan2(dy, dx))
    pitch = math.degrees(math.atan2(-dz, math.sqrt(dx ** 2 + dy ** 2)))
    return rot_utils.euler_angles_to_quats(
        np.array([0, pitch, yaw]), degrees=True
    )


def main():
    world = World()
    stage = world.stage

    usd_path = os.path.join(
        _project_dir, 'assets', 'environments',
        'InnovationCenter', 'InnovationCenter.usd'
    )
    print(f"Loading: {usd_path}", flush=True)

    building_prim = stage.DefinePrim("/World/Building", "Xform")
    building_prim.GetReferences().AddReference(
        usd_path, primPath=Sdf.Path("/World/Building")
    )

    # Bright daylight: angled sun + sky dome
    sun = UsdLux.DistantLight.Define(stage, "/World/SunLight")
    sun.CreateIntensityAttr(15000)
    sun.CreateAngleAttr(1.0)
    sun.CreateColorTemperatureAttr(6500.0)
    UsdGeom.Xformable(sun.GetPrim()).AddRotateXYZOp().Set(Gf.Vec3f(-50, 35, 0))

    dome = UsdLux.DomeLight.Define(stage, "/World/SkyDome")
    dome.CreateIntensityAttr(4000)
    dome.CreateColorTemperatureAttr(8000.0)

    # Let USD compose, then measure the building
    world.reset()
    world.step(render=True)

    bbox_cache = UsdGeom.BBoxCache(Usd.TimeCode.Default(), [UsdGeom.Tokens.default_])
    building = stage.GetPrimAtPath("/World/Building")
    bbox_range = bbox_cache.ComputeWorldBound(building).ComputeAlignedRange()
    bmin, bmax = bbox_range.GetMin(), bbox_range.GetMax()

    center = np.array([
        (bmin[0] + bmax[0]) / 2.0,
        (bmin[1] + bmax[1]) / 2.0,
        (bmin[2] + bmax[2]) / 2.0,
    ])
    extent = np.array([
        bmax[0] - bmin[0],
        bmax[1] - bmin[1],
        bmax[2] - bmin[2],
    ])
    print(f"Building center: ({center[0]:.1f}, {center[1]:.1f}, {center[2]:.1f}) m", flush=True)
    print(f"Building extent: ({extent[0]:.1f} x {extent[1]:.1f} x {extent[2]:.1f}) m", flush=True)

    # NW-corner hero angle: -x (west), +y (north), elevated above roof
    radius = max(extent[0], extent[1]) * 1.55
    cam_pos = np.array([
        center[0] + radius * math.cos(math.radians(113)),
        center[1] + radius * math.sin(math.radians(113)),
        bmax[2] + extent[2] * 0.25,
    ])
    orient = look_at_orientation(cam_pos, center)

    camera = Camera(
        prim_path="/World/HeroCam",
        position=cam_pos,
        frequency=30,
        resolution=(1920, 1080),
        orientation=orient,
    )
    UsdGeom.Camera(stage.GetPrimAtPath("/World/HeroCam")).CreateFocalLengthAttr(17.0)

    world.reset()
    camera.initialize()

    print(f"Camera pos: ({cam_pos[0]:.1f}, {cam_pos[1]:.1f}, {cam_pos[2]:.1f}) m", flush=True)
    print("Warming up renderer (60 frames for shader compilation)...", flush=True)
    for i in range(60):
        world.step(render=True)
        if i % 20 == 0:
            print(f"  warmup {i}/60", flush=True)

    # Capture
    rgba = None
    for attempt in range(10):
        world.step(render=True)
        rgba = camera.get_rgba()
        if rgba is not None and rgba.size > 0:
            break
        print(f"  capture retry {attempt}", flush=True)

    if rgba is None or rgba.size == 0:
        print("ERROR: camera returned no data", flush=True)
        simulation_app.close()
        sys.exit(1)

    rgb = rgba[..., :3].astype(np.uint8)
    out_dir = os.path.join(_project_dir, 'images')
    os.makedirs(out_dir, exist_ok=True)
    out_path = os.path.join(out_dir, 'innovation_center_exterior.png')
    imageio.imwrite(out_path, rgb)
    print(f"Saved: {out_path}  ({rgb.shape[1]}x{rgb.shape[0]})", flush=True)

    simulation_app.close()


if __name__ == "__main__":
    main()
