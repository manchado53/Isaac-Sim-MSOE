#!/usr/bin/env python3
"""
Quick door screenshot — places camera 5ft from the largest door and saves a PNG.
Used to verify camera position before recording video.

Usage:
    sbatch scripts/slurm/submit_isaac_job.sh simulations/environment/door_screenshot_verify.py \
        assets/environments/msoe_building/MSOEBuildingV4_physics.usd
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


def look_at_orientation(cam_pos, target):
    dx = target[0] - cam_pos[0]
    dy = target[1] - cam_pos[1]
    dz = target[2] - cam_pos[2]
    yaw = math.degrees(math.atan2(dy, dx))
    pitch = math.degrees(math.atan2(-dz, math.sqrt(dx**2 + dy**2)))
    return rot_utils.euler_angles_to_quats(
        np.array([0, pitch, yaw]), degrees=True
    )


def main():
    usd_path = sys.argv[1] if len(sys.argv) > 1 else os.path.join(
        _project_dir, 'assets', 'environments', 'msoe_building',
        'MSOEBuildingV4_physics.usd'
    )
    usd_path = os.path.abspath(usd_path)

    world = World()
    stage = world.stage

    print(f"Loading: {usd_path}", flush=True)
    building_prim = stage.DefinePrim("/World/Building", "Xform")
    building_prim.GetReferences().AddReference(
        usd_path, primPath=Sdf.Path("/World/Building")
    )

    # Freeze all doors so they stay in place for the screenshot
    for prim in stage.Traverse():
        if prim.HasAPI(UsdPhysics.RigidBodyAPI):
            UsdPhysics.RigidBodyAPI(prim).CreateKinematicEnabledAttr(True)

    # Find all doors with HingeJoint + DriveAPI
    doors = []
    bbox_cache = UsdGeom.BBoxCache(Usd.TimeCode.Default(), [UsdGeom.Tokens.default_])
    for prim in stage.Traverse():
        if prim.GetName() == "HingeJoint":
            door_prim = prim.GetParent()
            if not door_prim or not door_prim.HasAPI(UsdPhysics.RigidBodyAPI):
                continue
            bbox = bbox_cache.ComputeWorldBound(door_prim)
            rng = bbox.ComputeAlignedRange()
            if rng.IsEmpty():
                continue
            bmin, bmax = rng.GetMin(), rng.GetMax()
            dx = bmax[0] - bmin[0]
            dy = bmax[1] - bmin[1]
            dz = bmax[2] - bmin[2]
            area = max(dx, dy) * dz
            center = np.array([
                (bmin[0] + bmax[0]) / 2.0,
                (bmin[1] + bmax[1]) / 2.0,
                (bmin[2] + bmax[2]) / 2.0,
            ])
            doors.append({
                "name": door_prim.GetName(),
                "center": center,
                "size": (dx, dy, dz),
                "area": area,
                "bmin": np.array([bmin[0], bmin[1], bmin[2]]),
                "bmax": np.array([bmax[0], bmax[1], bmax[2]]),
            })

    # Sort by area — pick the largest door
    doors.sort(key=lambda d: d["area"], reverse=True)

    print(f"Found {len(doors)} doors. Top 5 by size:", flush=True)
    for d in doors[:5]:
        print(f"  {d['name']}: size=({d['size'][0]:.2f}x{d['size'][1]:.2f}x{d['size'][2]:.2f}) "
              f"area={d['area']:.1f} center=({d['center'][0]:.1f},{d['center'][1]:.1f},{d['center'][2]:.1f})",
              flush=True)

    door = doors[0]
    print(f"\nUsing: {door['name']}", flush=True)

    # Camera 5ft away, looking at door center
    # Figure out which side is wider (X or Y) to position camera perpendicular
    dx, dy, dz = door['size']
    cx, cy, cz = door['center']

    if dx >= dy:
        # Door extends along X — camera offset along Y
        cam_pos = np.array([cx, cy - 8.0, cz])
    else:
        # Door extends along Y — camera offset along X
        cam_pos = np.array([cx - 8.0, cy, cz])

    cam_target = door['center']
    orient = look_at_orientation(cam_pos, cam_target)

    print(f"Camera pos: ({cam_pos[0]:.1f}, {cam_pos[1]:.1f}, {cam_pos[2]:.1f})", flush=True)
    print(f"Looking at: ({cam_target[0]:.1f}, {cam_target[1]:.1f}, {cam_target[2]:.1f})", flush=True)

    cam = Camera(
        prim_path="/World/VerifyCam",
        position=cam_pos,
        frequency=30,
        resolution=(1920, 1080),
        orientation=orient,
    )
    # Tighter lens for close-up
    cam_prim = stage.GetPrimAtPath("/World/VerifyCam")
    UsdGeom.Camera(cam_prim).CreateFocalLengthAttr(24.0)

    # Lighting
    distant = UsdLux.DistantLight.Define(stage, "/World/Sun")
    distant.CreateIntensityAttr(15000)
    distant.CreateAngleAttr(1.0)
    xf = UsdGeom.Xformable(distant.GetPrim())
    xf.AddRotateXYZOp().Set(Gf.Vec3f(-45, 30, 0))

    dome = UsdLux.DomeLight.Define(stage, "/World/Sky")
    dome.CreateIntensityAttr(5000)

    sl = UsdLux.SphereLight.Define(stage, "/World/DoorLight")
    sl.CreateIntensityAttr(60000)
    sl.CreateRadiusAttr(0.3)
    UsdGeom.Xformable(sl.GetPrim()).AddTranslateOp().Set(
        Gf.Vec3f(float(cam_pos[0]), float(cam_pos[1]), float(cam_pos[2]) + 2.0)
    )

    world.reset()
    cam.initialize()

    # Warm up
    print("Warming up...", flush=True)
    for _ in range(60):
        world.step(render=True)

    # Capture
    rgba = cam.get_rgba()
    out_path = os.path.join(_project_dir, "images", "door_verify.png")
    os.makedirs(os.path.dirname(out_path), exist_ok=True)

    if rgba is not None and rgba.max() > 0:
        from PIL import Image, ImageDraw, ImageFont
        img = Image.fromarray(rgba[:, :, :3])
        draw = ImageDraw.Draw(img)
        try:
            font = ImageFont.truetype("/usr/share/fonts/truetype/dejavu/DejaVuSansMono.ttf", 28)
        except (IOError, OSError):
            font = ImageFont.load_default()
        lines = [
            f"Door: {door['name']}",
            f"Size: {dx:.2f} x {dy:.2f} x {dz:.2f} ft",
            f"Center: ({cx:.1f}, {cy:.1f}, {cz:.1f})",
            f"Camera: ({cam_pos[0]:.1f}, {cam_pos[1]:.1f}, {cam_pos[2]:.1f})",
        ]
        y = 20
        for line in lines:
            bbox = draw.textbbox((20, y), line, font=font)
            draw.rectangle([bbox[0]-4, bbox[1]-2, bbox[2]+4, bbox[3]+2], fill=(0,0,0,180))
            draw.text((20, y), line, fill=(255,255,255), font=font)
            y += 36
        img.save(out_path)
        print(f"Screenshot saved: {out_path}", flush=True)
    else:
        print("WARNING: Black frame!", flush=True)

    simulation_app.close()


if __name__ == "__main__":
    main()
