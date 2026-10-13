#!/usr/bin/env python3
"""
Auditorium Preview — single screenshot of the Dierks auditorium from the
camera position discovered by discover_auditorium.py.

No people yet — just the empty room, so we can see the framing before
spending iterations on pose tuning.

Usage:
  sbatch scripts/slurm/submit_isaac_job.sh simulations/environment/auditorium_preview.py \\
      --input assets/environments/msoe_building/DierksHallV2_physics.usd \\
      --anchors logs/dierks_auditorium.json

Output:
  images/dierks_auditorium_preview.png
"""

import argparse
import json
import os
import sys

parser = argparse.ArgumentParser(description="Preview the auditorium camera framing.")
parser.add_argument("--input", required=True, help="Path to DierksHallV2_physics.usd")
parser.add_argument(
    "--anchors",
    default="logs/dierks_auditorium.json",
    help="JSON from discover_auditorium.py",
)
parser.add_argument(
    "--out",
    default="images/dierks_auditorium_preview.png",
    help="Output PNG path",
)
parser.add_argument(
    "--warmup-frames",
    type=int,
    default=30,
    help="Frames to render before capturing (shader warmup)",
)
args, _unknown = parser.parse_known_args()

os.environ["OMNI_KIT_ACCEPT_EULA"] = "YES"

from isaacsim import SimulationApp
simulation_app = SimulationApp({"headless": True})

import numpy as np
from PIL import Image

import omni.usd
from pxr import Sdf, Usd, UsdLux, UsdGeom, Gf
from isaacsim.core.api import World
from isaacsim.sensors.camera import Camera

_script_dir = os.path.dirname(os.path.abspath(__file__))
_project_dir = os.path.abspath(os.path.join(_script_dir, '..', '..'))
sys.path.insert(0, _project_dir)

from sim_utils.building_physics import setup_physics_scene
from sim_utils.test_cameras import look_at_orientation

input_usd = os.path.abspath(args.input)
anchors_path = os.path.abspath(args.anchors) if not os.path.isabs(args.anchors) else args.anchors
out_path = os.path.abspath(args.out) if not os.path.isabs(args.out) else args.out
os.makedirs(os.path.dirname(out_path), exist_ok=True)


def main():
    print(f"Loading anchors: {anchors_path}", flush=True)
    with open(anchors_path) as f:
        anchors = json.load(f)

    cam_spec = anchors["camera"]
    cam_pos = np.array(cam_spec["position"])
    cam_target = np.array(cam_spec["target"])
    focal_mm = cam_spec.get("focal_length_mm", 18.0)
    resolution = tuple(cam_spec.get("resolution", [1920, 1080]))
    print(f"Camera pos:    {cam_pos}", flush=True)
    print(f"Camera target: {cam_target}", flush=True)
    print(f"Focal length:  {focal_mm}mm, resolution={resolution}", flush=True)

    # --- Load building ---
    world = World(stage_units_in_meters=1.0)
    stage = omni.usd.get_context().get_stage()

    print(f"Loading building: {input_usd}", flush=True)
    building_prim = stage.DefinePrim("/World/Building", "Xform")
    building_prim.GetReferences().AddReference(
        input_usd, primPath=Sdf.Path("/World/Building")
    )
    setup_physics_scene(stage)

    # --- Lighting ---
    # Interior auditorium — enclosed, so fall back to heavy artificial light.
    sun = UsdLux.DistantLight.Define(stage, Sdf.Path("/World/Sun"))
    sun.CreateIntensityAttr(2000)
    sun.CreateAngleAttr(1.0)
    UsdGeom.Xformable(sun.GetPrim()).AddRotateXYZOp().Set(Gf.Vec3f(-45, 30, 0))

    dome = UsdLux.DomeLight.Define(stage, Sdf.Path("/World/DomeLight"))
    dome.CreateIntensityAttr(1500)

    # Stage wash — RectLight above the stage pointing straight down.
    stage_center = anchors["stage"]["bbox"]["center"]
    stage_top_z = anchors["stage"]["bbox"]["max"][2]
    rect = UsdLux.RectLight.Define(stage, Sdf.Path("/World/StageWash"))
    rect.CreateIntensityAttr(25000)
    rect.CreateWidthAttr(15.0)
    rect.CreateHeightAttr(4.0)
    rx = UsdGeom.Xformable(rect.GetPrim())
    rx.AddTranslateOp().Set(Gf.Vec3d(stage_center[0], stage_center[1], stage_top_z + 4.0))
    rx.AddRotateXYZOp().Set(Gf.Vec3f(-90, 0, 0))  # face down

    # Camera-fill SphereLight — keeps the front-of-house illuminated.
    fill = UsdLux.SphereLight.Define(stage, Sdf.Path("/World/CamFill"))
    fill.CreateIntensityAttr(20000)
    fill.CreateRadiusAttr(3.0)
    UsdGeom.Xformable(fill.GetPrim()).AddTranslateOp().Set(
        Gf.Vec3d(float(cam_pos[0]), float(cam_pos[1]), float(cam_pos[2]))
    )

    # --- Camera ---
    orient = look_at_orientation(cam_pos, cam_target)
    camera = Camera(
        prim_path="/World/PreviewCam",
        position=cam_pos,
        frequency=30,
        resolution=resolution,
        orientation=orient,
    )
    cam_prim = stage.GetPrimAtPath("/World/PreviewCam")
    UsdGeom.Camera(cam_prim).CreateFocalLengthAttr(float(focal_mm))

    # --- Warm up + capture ---
    world.reset()
    camera.initialize()

    print(f"Warming up {args.warmup_frames} frames...", flush=True)
    rgba = None
    for i in range(args.warmup_frames):
        world.step(render=True)
        rgba = camera.get_rgba()

    if rgba is None:
        print("ERROR: camera returned no RGBA data", flush=True)
        simulation_app.close()
        sys.exit(1)

    rgb = rgba[:, :, :3].astype("uint8")
    Image.fromarray(rgb).save(out_path)
    print(f"Saved: {out_path}  ({rgb.shape[1]}x{rgb.shape[0]})", flush=True)

    simulation_app.close()


if __name__ == "__main__":
    main()
