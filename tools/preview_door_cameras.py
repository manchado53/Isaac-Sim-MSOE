#!/usr/bin/env python3
"""
Render N candidate camera positions around a door cluster as single PNGs.

Given a cluster centroid (from find_door_cluster.py), this script places a
camera at 8 radial positions (N/NE/E/SE/S/SW/W/NW) + a top-down view, all
aimed at the centroid, and renders one frame from each. You pick the best
view by eyeball; the script prints a ready-to-copy CLI snippet of the cam
position + target for the showcase scripts.

Usage:
    sbatch submit_isaac_job.sh tools/preview_door_cameras.py \
        --input assets/environments/msoe_building/MSOEBuildingV4_physics.usd \
        --centroid -16.5,78.2,23.8 \
        --extent 8.2,6.0,8.5

Or pull straight from the cluster JSON:
    sbatch submit_isaac_job.sh tools/preview_door_cameras.py \
        --input assets/environments/msoe_building/MSOEBuildingV4_physics.usd \
        --clusters-json logs/MSOEBuildingV4_physics_door_clusters.json \
        --cluster-rank 1

Output:
    images/diagnostics/preview_<tag>_<angle>.png  (1 per candidate)
    logs/<stem>_camera_preview.txt  (per-angle cam-pos/cam-target to copy)
"""

import argparse
import json
import math
import os
import sys

parser = argparse.ArgumentParser(description="Render camera candidates around a door cluster.")
parser.add_argument("--input", required=True)
parser.add_argument("--centroid", default=None, help="'cx,cy,cz' (required unless --clusters-json)")
parser.add_argument("--extent", default=None, help="'ex,ey,ez' (optional; defaults to 6,6,6)")
parser.add_argument("--clusters-json", default=None, help="Path to *_door_clusters.json")
parser.add_argument("--cluster-rank", type=int, default=1, help="1-based rank inside the JSON")
parser.add_argument("--tag", default="preview", help="Filename tag for the PNGs")
parser.add_argument("--focal-length", type=float, default=18.0)
parser.add_argument("--standoff", type=float, default=5.0,
                    help="Extra distance beyond cluster extent (meters)")
args, _unknown = parser.parse_known_args()

os.environ["OMNI_KIT_ACCEPT_EULA"] = "YES"

from isaacsim import SimulationApp
simulation_app = SimulationApp({"headless": True})

import numpy as np
import omni.usd
from pxr import Sdf, Usd, UsdGeom, UsdLux, UsdPhysics, Gf
import isaacsim.core.utils.numpy.rotations as rot_utils
from isaacsim.core.api import World
from isaacsim.sensors.camera import Camera

_script_dir = os.path.dirname(os.path.abspath(__file__))
_project_dir = os.path.abspath(os.path.join(_script_dir, '..'))
sys.path.insert(0, _project_dir)

from sim_utils.test_cameras import look_at_orientation


def parse3(s):
    return np.array([float(p.strip()) for p in s.split(",")])


def main():
    # Resolve centroid + extent
    if args.clusters_json:
        with open(args.clusters_json) as f:
            data = json.load(f)
        clusters = data.get("clusters", [])
        target = next((c for c in clusters if c["rank"] == args.cluster_rank), None)
        if target is None:
            print(f"ERROR: cluster rank {args.cluster_rank} not in {args.clusters_json}",
                  flush=True)
            simulation_app.close()
            return
        centroid = np.array(target["centroid"])
        extent = np.array(target["extent"])
        print(f"Loaded cluster #{target['rank']}: {target['members']} doors, "
              f"centroid={centroid.tolist()}, extent={extent.tolist()}", flush=True)
    else:
        if not args.centroid:
            print("ERROR: must pass --centroid or --clusters-json", flush=True)
            simulation_app.close()
            return
        centroid = parse3(args.centroid)
        extent = parse3(args.extent) if args.extent else np.array([6.0, 6.0, 6.0])

    input_usd = os.path.abspath(args.input)
    stem = os.path.splitext(os.path.basename(input_usd))[0]
    report_path = os.path.join(_project_dir, "logs", f"{stem}_camera_preview.txt")
    os.makedirs(os.path.dirname(report_path), exist_ok=True)
    diag_dir = os.path.join(_project_dir, "images", "diagnostics")
    os.makedirs(diag_dir, exist_ok=True)

    world = World(stage_units_in_meters=1.0)
    stage = omni.usd.get_context().get_stage()
    building_prim = stage.DefinePrim("/World/Building", "Xform")
    building_prim.GetReferences().AddReference(input_usd, Sdf.Path("/World/Building"))
    print("Building loaded.", flush=True)

    # Freeze rigid bodies so nothing moves (we want a still shot per candidate)
    for prim in stage.Traverse():
        p = str(prim.GetPath())
        if p.startswith("/World/Building") and prim.HasAPI(UsdPhysics.RigidBodyAPI):
            UsdPhysics.RigidBodyAPI(prim).CreateKinematicEnabledAttr(True)

    # Strong interior lighting so walls are bright enough to reason about
    UsdLux.DistantLight.Define(stage, Sdf.Path("/World/Prev_Sun")).CreateIntensityAttr(2500)
    UsdLux.DomeLight.Define(stage, Sdf.Path("/World/Prev_Dome")).CreateIntensityAttr(2000)
    # Sphere light at centroid for interior fill
    sph = UsdLux.SphereLight.Define(stage, Sdf.Path("/World/Prev_Fill"))
    sph.CreateIntensityAttr(40000)
    sph.CreateRadiusAttr(1.5)
    UsdGeom.Xformable(sph.GetPrim()).AddTranslateOp().Set(
        Gf.Vec3f(float(centroid[0]), float(centroid[1]), float(centroid[2])))

    # Candidates: 8 azimuths + 1 top-down
    radius_xy = max(float(extent[0]), float(extent[1])) / 2.0 + args.standoff
    z_eye = float(centroid[2]) + 1.0  # 1 m above centroid
    candidates = []
    labels = ["E", "NE", "N", "NW", "W", "SW", "S", "SE"]
    for i, label in enumerate(labels):
        theta = math.radians(i * 45.0)
        cx = float(centroid[0]) + radius_xy * math.cos(theta)
        cy = float(centroid[1]) + radius_xy * math.sin(theta)
        candidates.append((label, np.array([cx, cy, z_eye]), centroid.copy()))
    # Top-down (5 m above cluster top)
    top_pos = np.array([
        float(centroid[0]),
        float(centroid[1]) + 0.01,  # tiny y offset so yaw is defined
        float(centroid[2]) + float(extent[2]) / 2.0 + 5.0,
    ])
    candidates.append(("TOP", top_pos, centroid.copy()))

    log_lines = []
    log_lines.append(f"# Camera preview for {stem}")
    log_lines.append(f"# centroid={centroid.tolist()} extent={extent.tolist()} standoff={args.standoff}m")
    log_lines.append(f"# radius_xy={radius_xy:.2f}")

    # Create ONE camera prim, move it between candidates
    cam_path = "/World/PreviewCam"
    camera = Camera(
        prim_path=cam_path,
        position=candidates[0][1],
        frequency=30,
        resolution=(1920, 1080),
        orientation=look_at_orientation(candidates[0][1], candidates[0][2]),
    )
    cam_prim = stage.GetPrimAtPath(cam_path)
    if cam_prim.IsValid():
        UsdGeom.Camera(cam_prim).CreateFocalLengthAttr(args.focal_length)

    world.reset()
    camera.initialize()
    # Warm up renderer — first few frames are often black
    for _ in range(30):
        world.step(render=True)

    from PIL import Image
    for label, pos, target in candidates:
        orient = look_at_orientation(pos, target)
        camera.set_world_pose(position=pos, orientation=orient)
        # One render pass at new pose; a couple extra steps let lighting settle.
        for _ in range(4):
            world.step(render=True)
        rgba = camera.get_rgba()
        if rgba is None or rgba.size == 0:
            print(f"  {label}: no rgba returned, skipping", flush=True)
            continue
        fname = f"preview_{args.tag}_{label}.png"
        out = os.path.join(diag_dir, fname)
        Image.fromarray(rgba[:, :, :3].astype("uint8")).save(out)
        pos_str = f"{pos[0]:.2f},{pos[1]:.2f},{pos[2]:.2f}"
        tgt_str = f"{target[0]:.2f},{target[1]:.2f},{target[2]:.2f}"
        print(f"  [{label}] -> {fname}   cam-pos={pos_str}  cam-target={tgt_str}", flush=True)
        log_lines.append(f"[{label}] cam-pos={pos_str} cam-target={tgt_str}")

    with open(report_path, "w") as f:
        f.write("\n".join(log_lines) + "\n")
    print(f"\nWrote {report_path}", flush=True)
    print(f"PNGs in {diag_dir}/preview_{args.tag}_*.png", flush=True)

    simulation_app.close()


if __name__ == "__main__":
    main()
