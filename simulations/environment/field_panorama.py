#!/usr/bin/env python3
"""
Elevated 360-Degree Field Panorama

Drone-style orbital pass around a sports field / arena. Camera flies high
above the structure looking steeply down at the field surface — you see
the turf, markings, and bleachers instead of a ground-level wall-of-brick
view. Same 20s smooth 360 rotation as panorama_360.py, circular orbit
(no zoom in/out).

Key differences from panorama_360.py:
  * Camera Z is ~height_mult * building_height ABOVE the bbox — not at
    mid-height. Default height_mult=3.0 puts the camera well above bleachers.
  * Look-at target is at the field centroid (ground level), so the look
    direction automatically pitches steeply downward.
  * Tighter framing (smaller framing_mult) because aerial view forgives gaps.

Usage:
    sbatch submit_isaac_job.sh simulations/environment/field_panorama.py \
        --input assets/environments/msoe_building/VietsField.usd \
        --hide-patterns "Tree,Planting,RPC"

Output:
    videos/environment/<stem>_field_panorama.mp4
    logs/<stem>_field_panorama.txt
"""

import argparse
import math
import os
import sys

parser = argparse.ArgumentParser(description="Elevated 360-degree orbital panorama for a field / arena.")
parser.add_argument("--input", required=True, help="Path to the USD file")
parser.add_argument("--video", default=None, help="Output video name override")
parser.add_argument("--report", default=None, help="Report log path override")
parser.add_argument(
    "--hide-patterns",
    default="",
    help="Comma-separated substrings; prim paths under /World/Building "
         "matching any of these are hidden AND excluded from geometry calc.",
)
parser.add_argument("--duration-sec", type=float, default=20.0, help="Seconds per full revolution")
parser.add_argument("--fps", type=int, default=30, help="Frames per second")
parser.add_argument("--resolution", default="1920x1080", help="Camera resolution WxH")
parser.add_argument("--framing-mult", type=float, default=1.4,
                    help="Silhouette framing multiplier. Tighter than panorama_360.py's "
                         "1.8 because aerial view lets the frame contain more.")
parser.add_argument("--min-clearance", type=float, default=5.0,
                    help="Minimum meters between camera and nearest facade point.")
parser.add_argument("--height-mult", type=float, default=3.0,
                    help="Camera altitude = bbox_min_z + height_mult * building_height. "
                         "3.0 places camera 3x the structure's height above its base.")
parser.add_argument("--target-z-frac", type=float, default=0.10,
                    help="Where the camera looks on the Z axis: "
                         "bbox_min_z + target_z_frac * building_height. "
                         "Low values (0.0-0.15) aim at the field surface.")
parser.add_argument("--focal-length-mm", type=float, default=28.0,
                    help="Slightly longer than 24mm for less lens distortion in the aerial.")
parser.add_argument("--sun-intensity", type=float, default=4000.0, help="DistantLight intensity")
parser.add_argument("--sky-intensity", type=float, default=1500.0, help="DomeLight intensity")
parser.add_argument("--warmup-frames", type=int, default=60, help="Render warm-up frames")
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
video_name = args.video if args.video else f"environment/{stem}_field_panorama"
report_path = (
    os.path.abspath(args.report)
    if args.report
    else os.path.join(_project_dir, "logs", f"{stem}_field_panorama.txt")
)
os.makedirs(os.path.dirname(report_path), exist_ok=True)

try:
    res_w, res_h = [int(v) for v in args.resolution.lower().split("x")]
except ValueError:
    print(f"ERROR: bad --resolution '{args.resolution}', expected WxH", flush=True)
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
    log("Elevated Field Panorama")
    log("=" * 60)
    log(f"Input:      {input_usd}")
    log(f"Video:      {video_name}")
    log(f"Duration:   {args.duration_sec}s @ {args.fps}fps  (resolution {res_w}x{res_h})")
    log(f"Height-mult:{args.height_mult}  target-z-frac:{args.target_z_frac}  "
        f"framing-mult:{args.framing_mult}  min-clearance:{args.min_clearance}m")

    if not os.path.exists(input_usd):
        log(f"ERROR: {input_usd} not found.")
        _report_file.close()
        simulation_app.close()
        sys.exit(1)

    world = World(stage_units_in_meters=1.0)
    stage = omni.usd.get_context().get_stage()
    log(f"Stage metersPerUnit: {UsdGeom.GetStageMetersPerUnit(stage)}")

    log(f"\nLoading {input_usd}")
    add_reference_to_stage(input_usd, prim_path="/World/Building")
    building_prim = stage.GetPrimAtPath("/World/Building")
    if not building_prim.IsValid():
        log("ERROR: /World/Building prim invalid after reference")
        _report_file.close()
        simulation_app.close()
        sys.exit(1)

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

    # Circular elevated orbit — single radius across all angles so the
    # camera never zooms in/out. Because the camera sits high and looks
    # down, we want a radius that still frames the widest silhouette.
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

    cam_altitude = float(bb_min[2]) + args.height_mult * height
    target_z = float(bb_min[2]) + args.target_z_frac * height

    log(f"\nGeometry (meshes included: {mesh_count})")
    log(f"  extent          : ({bldg_extent[0]:.2f}, {bldg_extent[1]:.2f}, {bldg_extent[2]:.2f})")
    log(f"  centroid        : ({centroid[0]:.2f}, {centroid[1]:.2f}, {centroid[2]:.2f})")
    log(f"  silhouette_half : min={silhouette_half_arr.min():.1f}  max={silhouette_half_arr.max():.1f}")
    log(f"  orbit_radius    : {orbit_radius:.2f}  (constant)")
    log(f"  cam altitude    : {cam_altitude:.2f} m above world origin "
        f"({args.height_mult}x structure height above ground)")
    log(f"  target Z        : {target_z:.2f}  (pitches camera steeply downward)")

    # Compute approximate look-down pitch for the report.
    dz = target_z - cam_altitude
    approx_pitch = math.degrees(math.atan2(-dz, orbit_radius))
    log(f"  approx pitch    : {approx_pitch:.1f} deg below horizontal")

    target = np.array([centroid[0], centroid[1], target_z])

    add_exterior_lighting(stage, args.sun_intensity, args.sky_intensity)

    initial_pos = np.array([centroid[0] + orbit_radius, centroid[1], cam_altitude])
    initial_orient = look_at_orientation(initial_pos, target)
    camera = Camera(
        prim_path="/World/FieldPanoramaCamera",
        position=initial_pos,
        frequency=args.fps,
        resolution=(res_w, res_h),
        orientation=initial_orient,
    )

    world.reset()
    camera.initialize()
    cam_prim = stage.GetPrimAtPath("/World/FieldPanoramaCamera")
    if cam_prim.IsValid():
        UsdGeom.Camera(cam_prim).CreateFocalLengthAttr(args.focal_length_mm)

    log(f"\nWarming up renderer ({args.warmup_frames} frames)...")
    for _ in range(args.warmup_frames):
        world.step(render=True)

    total_frames = int(round(args.duration_sec * args.fps))
    log(f"\nRecording {total_frames} frames "
        f"({360.0 / total_frames:.2f} deg/frame, {args.duration_sec}s @ {args.fps}fps)")

    captured = 0
    with VideoRecorder(video_name, fps=args.fps) as recorder:
        for i in range(total_frames):
            angle = 2.0 * math.pi * i / total_frames

            cam_x = centroid[0] + orbit_radius * math.cos(angle)
            cam_y = centroid[1] + orbit_radius * math.sin(angle)
            cam_z = cam_altitude
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
