#!/usr/bin/env python3
"""
360-Degree Building Panorama

Orbital camera tour for building USDs — one smooth 360 revolution centered on
the volume-weighted centroid of the building, with configurable radius,
duration, and gentle vertical bob. Designed for presentation slides.

Three targeted fixes for common 360-video problems:

  1. Building looks too small  -> radius = radius_mult * half_diagonal,
     where half_diagonal is half of sqrt(dx^2 + dy^2).  Default 1.05 gives a
     tight hero framing.
  2. Camera moves too fast     -> duration_sec * fps frames per revolution
     (default 20s @ 30fps = 600 frames, 0.6 deg/frame).
  3. Axis of rotation off-center -> orbit around the volume-weighted mesh
     centroid (computed after hiding trees/site clutter), not the bbox
     midpoint.  Hidden prims are excluded from the centroid and bbox math.

Usage:
    sbatch scripts/slurm/submit_isaac_job.sh simulations/environment/panorama_360.py \
        --input assets/environments/msoe_building/MSOEBuildingV4.usd \
        --hide-patterns "Tree,Planting,Vegetation,Entourage,RPC,Plant,Shrub,Bush,Landscape,Topography,Site_"

Output:
    videos/environment/<stem>_panorama_360.mp4
    logs/<stem>_panorama_360.txt
"""

import argparse
import math
import os
import sys

parser = argparse.ArgumentParser(description="360-degree orbital panorama of a building USD.")
parser.add_argument("--input", required=True, help="Path to the USD file")
parser.add_argument("--video", default=None, help="Output video name override")
parser.add_argument("--report", default=None, help="Report log path override")
parser.add_argument(
    "--hide-patterns",
    default="",
    help="Comma-separated substrings; any prim path under /World/Building "
         "matching any of these is hidden AND excluded from geometry calc. "
         "Example: 'Tree,Planting,Entourage'.",
)
parser.add_argument("--duration-sec", type=float, default=20.0, help="Seconds per full revolution")
parser.add_argument("--fps", type=int, default=30, help="Frames per second")
parser.add_argument("--resolution", default="1920x1080", help="Camera resolution WxH")
parser.add_argument("--framing-mult", type=float, default=1.8,
                    help="Silhouette framing multiplier. radius(angle) = "
                         "facade_along(angle) + silhouette_half(angle)*framing_mult "
                         "+ min_clearance. Larger = further back = smaller apparent building. "
                         "Default 1.8 fills frame comfortably at focal-length 24mm.")
parser.add_argument("--min-clearance", type=float, default=5.0,
                    help="Minimum meters of air between camera and nearest facade point. "
                         "Prevents clipping on tight-framing presets. Default 5m.")
parser.add_argument("--height-frac", type=float, default=0.55, help="Camera Z above bbox_min as fraction of height")
parser.add_argument("--bob-frac", type=float, default=0.08, help="Vertical bob amplitude as fraction of height")
parser.add_argument("--bob-cycles", type=int, default=2, help="Vertical bob cycles per revolution")
parser.add_argument("--focal-length-mm", type=float, default=24.0, help="Camera focal length (mm)")
parser.add_argument("--sun-intensity", type=float, default=3500.0, help="DistantLight intensity")
parser.add_argument("--sky-intensity", type=float, default=1000.0, help="DomeLight intensity")
parser.add_argument("--warmup-frames", type=int, default=60, help="Render warm-up frames before recording")
args, _unknown = parser.parse_known_args()

os.environ["OMNI_KIT_ACCEPT_EULA"] = "YES"

from isaacsim import SimulationApp

simulation_app = SimulationApp({"headless": True})

import numpy as np
import isaacsim.core.utils.numpy.rotations as rot_utils

import omni.usd
from pxr import Gf, Sdf, Usd, UsdGeom, UsdLux
from isaacsim.core.api import World
from isaacsim.core.utils.stage import add_reference_to_stage
from isaacsim.sensors.camera import Camera

_script_dir = os.path.dirname(os.path.abspath(__file__))
_project_dir = os.path.abspath(os.path.join(_script_dir, "..", ".."))
sys.path.insert(0, _project_dir)

from sim_utils.video_capture import VideoRecorder
from sim_utils.panorama_helpers import (
    hide_matching_prims,
    compute_building_geometry,
    sample_orbit_geometry,
    radius_at,
    look_at_orientation,
    add_exterior_lighting,
)

input_usd = os.path.abspath(args.input)
stem = os.path.splitext(os.path.basename(input_usd))[0]

video_name = args.video if args.video else f"environment/{stem}_panorama_360"
report_path = (
    os.path.abspath(args.report)
    if args.report
    else os.path.join(_project_dir, "logs", f"{stem}_panorama_360.txt")
)
os.makedirs(os.path.dirname(report_path), exist_ok=True)

try:
    res_w, res_h = [int(v) for v in args.resolution.lower().split("x")]
except ValueError:
    print(f"ERROR: bad --resolution '{args.resolution}', expected WxH (e.g. 1920x1080)", flush=True)
    simulation_app.close()
    sys.exit(1)

_report_file = None


def log(msg=""):
    print(msg, flush=True)
    if _report_file:
        _report_file.write(msg + "\n")
        _report_file.flush()


def main():
    global _report_file
    _report_file = open(report_path, "w")

    log("=" * 60)
    log("360 Panorama")
    log("=" * 60)
    log(f"Input:      {input_usd}")
    log(f"Video:      {video_name}")
    log(f"Duration:   {args.duration_sec}s @ {args.fps}fps  (resolution {res_w}x{res_h})")
    log(f"Framing-mult:{args.framing_mult}  min-clearance:{args.min_clearance}m  "
        f"height-frac:{args.height_frac}  bob-frac:{args.bob_frac} x {args.bob_cycles}cycles")

    if not os.path.exists(input_usd):
        log(f"ERROR: {input_usd} not found.")
        _report_file.close()
        simulation_app.close()
        sys.exit(1)

    # --- Scene setup ---
    world = World(stage_units_in_meters=1.0)
    stage = omni.usd.get_context().get_stage()

    mpu = UsdGeom.GetStageMetersPerUnit(stage)
    log(f"Stage metersPerUnit: {mpu}")

    log(f"\nLoading {input_usd}")
    add_reference_to_stage(input_usd, prim_path="/World/Building")
    building_prim = stage.GetPrimAtPath("/World/Building")
    if not building_prim.IsValid():
        log("ERROR: /World/Building prim invalid after reference")
        _report_file.close()
        simulation_app.close()
        sys.exit(1)

    # Hide trees/landscape BEFORE measuring the building.
    hide_patterns = [p.strip() for p in args.hide_patterns.split(",") if p.strip()]
    hidden_paths = set()
    if hide_patterns:
        hidden_paths, per_pattern = hide_matching_prims(stage, hide_patterns)
        log(f"\nHide patterns: {hide_patterns}")
        for pat, count in per_pattern.items():
            log(f"  '{pat}' -> {count} prims hidden")
        log(f"Total prims hidden: {len(hidden_paths)}")

    bbox_cache = UsdGeom.BBoxCache(
        Usd.TimeCode.Default(), [UsdGeom.Tokens.default_]
    )

    result = compute_building_geometry(
        building_prim, bbox_cache, hidden_paths=hidden_paths
    )
    if result is None or result[0] is None:
        log("ERROR: no visible mesh geometry under /World/Building")
        _report_file.close()
        simulation_app.close()
        sys.exit(1)

    centroid, bb_min, bb_max, mesh_count, corners_xy = result
    bldg_extent = bb_max - bb_min
    height = float(bldg_extent[2])
    horiz_half_diag = 0.5 * math.sqrt(
        float(bldg_extent[0]) ** 2 + float(bldg_extent[1]) ** 2
    )

    # Sample per-angle facade_along + silhouette_half, then pick the
    # single maximum radius across all angles. That max is the orbit
    # radius — constant, so the camera never zooms in/out as it rotates.
    # For elongated buildings this means the camera sits far enough back
    # to frame the wide silhouette even from the short-side views; the
    # apparent building size is constant across the whole revolution.
    sample_angles, facade_along_arr, silhouette_half_arr = sample_orbit_geometry(
        centroid[:2], corners_xy, num_samples=72
    )
    per_angle_radii = np.array([
        radius_at(
            a, sample_angles, facade_along_arr, silhouette_half_arr,
            args.framing_mult, args.min_clearance,
        )
        for a in sample_angles
    ])
    orbit_radius = float(per_angle_radii.max())

    cam_z_base = float(bb_min[2]) + args.height_frac * height
    bob_amp = args.bob_frac * height

    log(f"\nGeometry (meshes included: {mesh_count})")
    log(f"  bbox_min       : ({bb_min[0]:.2f}, {bb_min[1]:.2f}, {bb_min[2]:.2f})")
    log(f"  bbox_max       : ({bb_max[0]:.2f}, {bb_max[1]:.2f}, {bb_max[2]:.2f})")
    log(f"  extent         : ({bldg_extent[0]:.2f}, {bldg_extent[1]:.2f}, {bldg_extent[2]:.2f})")
    log(f"  centroid       : ({centroid[0]:.2f}, {centroid[1]:.2f}, {centroid[2]:.2f})")
    log(f"  horiz_half_diag: {horiz_half_diag:.2f}")
    log(f"  facade_along   : min={facade_along_arr.min():.1f}  max={facade_along_arr.max():.1f}")
    log(f"  silhouette_half: min={silhouette_half_arr.min():.1f}  max={silhouette_half_arr.max():.1f}")
    log(f"  orbit mode     : CIRCULAR (framing_mult={args.framing_mult}, "
        f"min_clearance={args.min_clearance}m)")
    log(f"  orbit_radius   : {orbit_radius:.2f}  (constant — no zoom in/out)")
    log(f"  per-angle radii: min={per_angle_radii.min():.1f}  max={per_angle_radii.max():.1f} "
        f"(max is used)")
    log(f"  cam_z_base     : {cam_z_base:.2f}")
    log(f"  bob_amp        : {bob_amp:.2f}")

    # Target: centroid XY, slightly-above-base Z so the look-at pitch is
    # gentle (not aiming at the rooftop or the foundation).
    target = np.array([centroid[0], centroid[1], float(bb_min[2]) + 0.45 * height])

    add_exterior_lighting(stage, args.sun_intensity, args.sky_intensity)

    # --- Camera ---
    initial_pos = np.array([centroid[0] + orbit_radius, centroid[1], cam_z_base])
    initial_orient = look_at_orientation(initial_pos, target)
    camera = Camera(
        prim_path="/World/PanoramaCamera",
        position=initial_pos,
        frequency=args.fps,
        resolution=(res_w, res_h),
        orientation=initial_orient,
    )

    world.reset()
    camera.initialize()
    cam_prim = stage.GetPrimAtPath("/World/PanoramaCamera")
    if cam_prim.IsValid():
        UsdGeom.Camera(cam_prim).CreateFocalLengthAttr(args.focal_length_mm)

    # Warm-up so the first recorded frame isn't a black texture.
    log(f"\nWarming up renderer ({args.warmup_frames} frames)...")
    for _ in range(args.warmup_frames):
        world.step(render=True)

    # --- Orbit loop ---
    total_frames = int(round(args.duration_sec * args.fps))
    log(f"\nRecording {total_frames} frames "
        f"({360.0 / total_frames:.2f} deg/frame, "
        f"{args.duration_sec}s @ {args.fps}fps)")

    captured = 0
    with VideoRecorder(video_name, fps=args.fps) as recorder:
        for i in range(total_frames):
            angle = 2.0 * math.pi * i / total_frames

            cam_x = centroid[0] + orbit_radius * math.cos(angle)
            cam_y = centroid[1] + orbit_radius * math.sin(angle)
            cam_z = cam_z_base + bob_amp * math.sin(args.bob_cycles * angle)
            cam_pos = np.array([cam_x, cam_y, cam_z])

            orient = look_at_orientation(cam_pos, target)
            camera.set_world_pose(position=cam_pos, orientation=orient)
            world.step(render=True)

            rgba = camera.get_rgba()
            if rgba is not None and rgba.size > 0:
                recorder.capture_rgba(rgba)
                captured += 1

            if i % 60 == 0 or i == total_frames - 1:
                log(f"  frame {i+1}/{total_frames}  "
                    f"pos=({cam_x:.1f},{cam_y:.1f},{cam_z:.1f})  "
                    f"angle={math.degrees(angle):.1f}")

    log(f"\nCaptured {captured}/{total_frames} frames")
    log("Done.")
    _report_file.close()
    simulation_app.close()


if __name__ == "__main__":
    main()
