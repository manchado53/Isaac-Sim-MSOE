#!/usr/bin/env python3
"""
Door Chaos Showcase — wrong-axis shear chaos.

Loads the final MSOE Innovation Center physics USD, picks a camera with the
densest visible door cluster, and deliberately *breaks* each door's hinge
physics: rewrites the revolute joint axis to X or Y (instead of Z), removes
angle limits, zeroes drive damping, and flails the target angle randomly.
The result is doors pivoting around floor-parallel axes — cartwheeling,
flipping through walls, oscillating violently — which is what the old
pipeline looked like before the axis-detection fix.

This is the "before" visual for the physics-pipeline story.

Usage:
    sbatch scripts/slurm/submit_isaac_job.sh simulations/environment/door_chaos_showcase.py \
        --input assets/environments/msoe_building/MSOEBuildingV4_physics.usd

Output:
    - videos/environment/<stem>_door_chaos.mp4 (15s @ 30fps, 1920x1080)
"""

import argparse
import math
import os
import random
import sys

parser = argparse.ArgumentParser(description="Wrong-axis door chaos showcase.")
parser.add_argument("--input", required=True, help="Path to physics-ready USD file")
parser.add_argument("--video", default=None, help="Output video name override")
parser.add_argument("--num-doors", type=int, default=12,
                    help="Number of doors to flail near the chosen camera")
parser.add_argument("--seed", type=int, default=7, help="RNG seed for chaos")
parser.add_argument("--cam-pos", default=None,
                    help="Explicit camera position 'X,Y,Z' (meters). "
                         "If provided with --cam-target, skips auto-discovery.")
parser.add_argument("--cam-target", default=None,
                    help="Explicit look-at target 'X,Y,Z' (meters).")
parser.add_argument("--focal-length", type=float, default=18.0,
                    help="Camera focal length in mm (default 18 = wider than Camera's "
                         "built-in 24 mm default; helps fit more doors into frame).")
parser.add_argument(
    "--hide-patterns",
    default="Tree,Planting,Vegetation,Entourage,RPC,Plant,Shrub,Bush,Landscape,Topography,Site_",
    help="Comma-separated substrings; any prim path under /World/Building matching "
         "one of these gets visibility=invisible. Strips interior planting & trees "
         "that obscure door views.",
)
parser.add_argument("--light-boost", type=float, default=1.0,
                    help="Multiplier on interior lighting intensities (default 1.0 = baseline).")
parser.add_argument("--pivot-chaos", action="store_true", default=True,
                    help="Also jitter each joint's local frame — tilts the rotation axis "
                         "off cardinal X/Y/Z and moves the pivot away from the hinge edge.")
parser.add_argument("--no-pivot-chaos", dest="pivot_chaos", action="store_false",
                    help="Disable pivot chaos (doors only get wrong-axis, same hinge).")
parser.add_argument("--pivot-tilt-deg", type=float, default=60.0,
                    help="Max per-axis tilt applied to the joint's local orientation (deg).")
parser.add_argument("--pivot-offset-m", type=float, default=0.6,
                    help="Max per-axis offset (m) applied to the joint's local pivot on the door.")
parser.add_argument("--no-overlay", action="store_true",
                    help="Skip the top-right stats HUD (useful for regression diffs).")
parser.add_argument("--auto-framing", action="store_true",
                    help="When no --cam-pos override is given, recentre on the door "
                         "cluster (old behaviour). Default: use the hero cam preset.")
args, _unknown = parser.parse_known_args()


def parse_xyz(s):
    parts = [p.strip() for p in s.split(",")]
    if len(parts) != 3:
        raise ValueError(f"Expected 'X,Y,Z', got {s!r}")
    return np.array([float(p) for p in parts])

os.environ["OMNI_KIT_ACCEPT_EULA"] = "YES"

from isaacsim import SimulationApp
simulation_app = SimulationApp({"headless": True})

import numpy as np
import omni.usd
from pxr import Gf, Sdf, Usd, UsdGeom, UsdLux, UsdPhysics
import isaacsim.core.utils.numpy.rotations as rot_utils
from isaacsim.core.api import World
from isaacsim.sensors.camera import Camera

_script_dir = os.path.dirname(os.path.abspath(__file__))
_project_dir = os.path.abspath(os.path.join(_script_dir, '..', '..'))
sys.path.insert(0, _project_dir)

from sim_utils.video_capture import VideoRecorder
from sim_utils.building_physics import set_drive_target
from sim_utils.stats_overlay import StatsOverlay
from sim_utils.test_cameras import (
    discover_test_cameras,
    look_at_orientation,
    add_interior_lighting,
)


# ---------------------------------------------------------------------------
# Parameters
# ---------------------------------------------------------------------------
FPS = 30
TOTAL_FRAMES = 450         # 15 s
WARMUP_FRAMES = 30
RESOLUTION = (1920, 1080)

# Hero preset — shared with door_sequence_showcase.py so the two demos frame
# the same corridor. See that file for provenance.
HERO_CAM_POS = (-25.59, 78.17, 24.75)
HERO_CAM_TARGET = (-16.50, 78.17, 23.75)
HERO_LOCATION_LINES = (
    "MSOE Innovation Center",
    "Floor 3 · South Corridor",
)

# Axis distribution: 70% wrong (X or Y), 30% Z. Weighted list for random.choice.
AXIS_POOL = ["X", "X", "X", "Y", "Y", "Y", "Z"]

# Flip targets between these angles (degrees) to make doors flail.
FLIP_ANGLES = [-180.0, -120.0, -60.0, 60.0, 120.0, 180.0]

# Each door independently re-randomizes its drive target every
# FLIP_INTERVAL_SEC ± JITTER seconds.
FLIP_INTERVAL_SEC = 0.8
FLIP_JITTER_SEC = 0.5


# ---------------------------------------------------------------------------
# Door discovery
# ---------------------------------------------------------------------------

def find_door_joint(door_prim):
    """Return the child hinge-joint prim of a door, or None."""
    for child in door_prim.GetChildren():
        name = child.GetName()
        if "Hinge" in name or "Joint" in name:
            if child.IsA(UsdPhysics.RevoluteJoint):
                return child
    return None


def find_drive_on_joint(joint_prim):
    """Return the angular DriveAPI on a joint prim, or None."""
    drive = UsdPhysics.DriveAPI(joint_prim, "angular")
    if drive.GetTargetPositionAttr().IsValid():
        return drive
    return None


def iter_doors(stage):
    """Yield (door_prim, joint_prim, drive) for every physics-controlled door."""
    for prim in stage.Traverse():
        if not prim.HasAPI(UsdPhysics.RigidBodyAPI):
            continue
        joint = find_door_joint(prim)
        if joint is None:
            continue
        drive = find_drive_on_joint(joint)
        if drive is None:
            continue
        yield prim, joint, drive


def world_center(prim, bbox_cache):
    bbox = bbox_cache.ComputeWorldBound(prim).ComputeAlignedRange()
    mn, mx = bbox.GetMin(), bbox.GetMax()
    return np.array([
        (mn[0] + mx[0]) / 2.0,
        (mn[1] + mx[1]) / 2.0,
        (mn[2] + mx[2]) / 2.0,
    ])


def hide_patterns_in_building(stage, patterns):
    """Hide any prim under /World/Building whose path contains one of `patterns`."""
    per_pattern = {p: 0 for p in patterns}
    total = 0
    for prim in stage.Traverse():
        path = str(prim.GetPath())
        if not path.startswith("/World/Building"):
            continue
        for pat in patterns:
            if pat and pat in path:
                imageable = UsdGeom.Imageable(prim)
                if imageable:
                    imageable.CreateVisibilityAttr().Set(UsdGeom.Tokens.invisible)
                per_pattern[pat] += 1
                total += 1
                break
    return total, per_pattern


def add_bright_lighting(stage, cam_pos, cam_target, boost=1.0):
    """Interior lighting tuned for corridor-style atrium door shots.

    Strategy: strong ambient DomeLight + sun DistantLight + a pair of
    ceiling fixtures (one above the cluster centroid, one halfway between
    camera and cluster). No point light at the camera position — it
    bounces off nearby drywall and creates a visible hotspot.
    """
    distant = UsdLux.DistantLight.Define(stage, Sdf.Path("/World/DoorSun"))
    distant.CreateIntensityAttr(3500.0 * boost)
    distant.CreateAngleAttr(1.0)
    UsdGeom.Xformable(distant.GetPrim()).AddRotateXYZOp().Set(Gf.Vec3f(-45, 30, 0))

    dome = UsdLux.DomeLight.Define(stage, Sdf.Path("/World/DoorDome"))
    dome.CreateIntensityAttr(6000.0 * boost)

    # Ceiling fixture #1: directly above cluster centroid
    cam_pos = np.asarray(cam_pos, dtype=float)
    cam_target = np.asarray(cam_target, dtype=float)
    fx1 = UsdLux.SphereLight.Define(stage, Sdf.Path("/World/DoorCeilingA"))
    fx1.CreateIntensityAttr(25000.0 * boost)
    fx1.CreateRadiusAttr(1.5)
    UsdGeom.Xformable(fx1.GetPrim()).AddTranslateOp().Set(
        Gf.Vec3f(float(cam_target[0]), float(cam_target[1]), float(cam_target[2]) + 4.0)
    )

    # Ceiling fixture #2: halfway between camera and target, same height
    # offset above — so the near side of the corridor also reads bright.
    mid = (cam_pos + cam_target) / 2.0
    fx2 = UsdLux.SphereLight.Define(stage, Sdf.Path("/World/DoorCeilingB"))
    fx2.CreateIntensityAttr(18000.0 * boost)
    fx2.CreateRadiusAttr(1.2)
    UsdGeom.Xformable(fx2.GetPrim()).AddTranslateOp().Set(
        Gf.Vec3f(float(mid[0]), float(mid[1]), float(mid[2]) + 4.0)
    )

    # --- Subtle fill lights ---------------------------------------------
    # Tune these two intensity values to brighten/darken the fills.
    FILL_FRONT_LEFT_INTENSITY = 4000.0   # near-camera, camera-left side
    FILL_FAR_BACK_INTENSITY = 12000.0    # deep end of corridor

    forward = cam_target - cam_pos
    fwd_norm = np.linalg.norm(forward)
    forward_unit = forward / fwd_norm if fwd_norm > 1e-6 else np.array([1.0, 0.0, 0.0])
    # Camera-left in world space (cross up × forward with up = +Z).
    left_unit = np.cross(np.array([0.0, 0.0, 1.0]), forward_unit)
    left_norm = np.linalg.norm(left_unit)
    if left_norm > 1e-6:
        left_unit = left_unit / left_norm
    else:
        left_unit = np.array([0.0, 1.0, 0.0])

    # Front-left fill: tucked BEHIND the camera (outside FOV) and up near
    # the ceiling so the sphere is never visible on a wall. Still lights
    # the near-left side of the corridor indirectly.
    fl_pos = cam_pos - 1.5 * forward_unit + 1.5 * left_unit + np.array([0.0, 0.0, 2.0])
    fx_fl = UsdLux.SphereLight.Define(stage, Sdf.Path("/World/DoorFillFrontLeft"))
    fx_fl.CreateIntensityAttr(FILL_FRONT_LEFT_INTENSITY * boost)
    fx_fl.CreateRadiusAttr(0.5)
    UsdGeom.Xformable(fx_fl.GetPrim()).AddTranslateOp().Set(
        Gf.Vec3f(float(fl_pos[0]), float(fl_pos[1]), float(fl_pos[2]))
    )

    # Far-back fill: 5 m past the target along the camera forward axis,
    # roughly at ceiling height of that depth.
    fb_pos = cam_target + 5.0 * forward_unit + np.array([0.0, 0.0, 1.5])
    fx_fb = UsdLux.SphereLight.Define(stage, Sdf.Path("/World/DoorFillFarBack"))
    fx_fb.CreateIntensityAttr(FILL_FAR_BACK_INTENSITY * boost)
    fx_fb.CreateRadiusAttr(1.0)
    UsdGeom.Xformable(fx_fb.GetPrim()).AddTranslateOp().Set(
        Gf.Vec3f(float(fb_pos[0]), float(fb_pos[1]), float(fb_pos[2]))
    )


# ---------------------------------------------------------------------------
# Camera selection
# ---------------------------------------------------------------------------

def choose_camera_with_most_doors(cameras, door_records, search_radius=15.0):
    """Pick the camera whose position is closest to the densest door cluster."""
    best = None
    best_count = -1
    for key, cam in cameras.items():
        pos = np.array(cam["pos"])
        count = sum(
            1
            for _p, _j, _d, center in door_records
            if np.linalg.norm(center - pos) <= search_radius
        )
        if count > best_count:
            best_count = count
            best = (key, cam, count)
    return best


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    rng = random.Random(args.seed)

    input_usd = os.path.abspath(args.input)
    stem = os.path.splitext(os.path.basename(input_usd))[0]
    video_name = args.video if args.video else f"environment/{stem}_door_chaos"

    print("=" * 70, flush=True)
    print("Door Chaos Showcase — wrong-axis shear chaos", flush=True)
    print(f"Input: {input_usd}", flush=True)
    print(f"Output: videos/{video_name}.mp4", flush=True)
    print("=" * 70, flush=True)

    world = World(stage_units_in_meters=1.0)
    stage = omni.usd.get_context().get_stage()

    # Load building
    building_prim = stage.DefinePrim("/World/Building", "Xform")
    building_prim.GetReferences().AddReference(input_usd, Sdf.Path("/World/Building"))
    print("Building loaded.", flush=True)

    # Hide trees, planting, entourage — anything that obstructs the door view.
    hide_patterns = [p.strip() for p in args.hide_patterns.split(",") if p.strip()]
    if hide_patterns:
        total, per_pat = hide_patterns_in_building(stage, hide_patterns)
        print(f"Hide patterns: {hide_patterns}", flush=True)
        for pat, cnt in per_pat.items():
            if cnt:
                print(f"  '{pat}' -> {cnt}", flush=True)
        print(f"Total hidden: {total}", flush=True)

    # Collect every door + its world-space center (for camera proximity)
    bbox_cache = UsdGeom.BBoxCache(Usd.TimeCode.Default(), [UsdGeom.Tokens.default_])
    door_records = []
    for door_prim, joint, drive in iter_doors(stage):
        center = world_center(door_prim, bbox_cache)
        door_records.append((door_prim, joint, drive, center))
    print(f"Discovered {len(door_records)} physics-controlled doors.", flush=True)
    if not door_records:
        print("ERROR: no doors found in stage", flush=True)
        simulation_app.close()
        return

    # Camera selection: explicit override → hero preset.
    if args.cam_pos and args.cam_target:
        cam_pos = parse_xyz(args.cam_pos)
        cam_target = parse_xyz(args.cam_target)
        cam_source = "override"
        print(f"Using explicit camera: pos={cam_pos.tolist()} target={cam_target.tolist()}",
              flush=True)
        pivot = cam_target  # pick doors closest to the target, not the camera
    else:
        cam_pos = np.array(HERO_CAM_POS)
        cam_target = np.array(HERO_CAM_TARGET)
        cam_source = "hero"
        print(f"Using hero preset (cluster #1 W view): "
              f"pos={cam_pos.tolist()} target={cam_target.tolist()}", flush=True)
        pivot = cam_target

    # Z-band filter — exclude doors more than ±2 m above/below the pivot
    # so floor-above / floor-below doors don't sneak into the selection.
    Z_BAND = 2.0
    floor_z = float(pivot[2])
    before_count = len(door_records)
    door_records = [
        r for r in door_records
        if abs(float(r[3][2]) - floor_z) <= Z_BAND
    ]
    filtered = before_count - len(door_records)
    if filtered:
        print(f"Filtered out {filtered} doors outside Z-band [{floor_z - Z_BAND:.1f}, "
              f"{floor_z + Z_BAND:.1f}] m (different floor).", flush=True)

    # Pick the N closest doors to the pivot so the visual cluster is tight.
    door_records.sort(key=lambda r: np.linalg.norm(r[3] - pivot))
    chosen = door_records[: args.num_doors]
    nearest_dists = [float(np.linalg.norm(r[3] - pivot)) for r in chosen]
    print(f"Using closest {len(chosen)} doors. "
          f"Distances from pivot: "
          f"min={min(nearest_dists):.1f}m, max={max(nearest_dists):.1f}m, "
          f"mean={sum(nearest_dists)/len(nearest_dists):.1f}m",
          flush=True)

    # Break physics on each chosen door
    broken = []
    pivot_info = []
    for door_prim, joint, drive, center in chosen:
        joint_api = UsdPhysics.RevoluteJoint(joint)
        axis = rng.choice(AXIS_POOL)
        joint_api.GetAxisAttr().Set(axis)
        # Remove angle clamps so doors can over-rotate.
        joint_api.CreateLowerLimitAttr(-720.0)
        joint_api.CreateUpperLimitAttr(720.0)
        # Zero damping so drives oscillate wildly.
        drive.GetDampingAttr().Set(0.0)

        # Pivot chaos: tilt the joint's local frame + shift the pivot point
        # on the door so rotation happens around an off-center axis that
        # isn't aligned with cardinal X/Y/Z in world space.
        if args.pivot_chaos:
            rx = rng.uniform(-args.pivot_tilt_deg, args.pivot_tilt_deg)
            ry = rng.uniform(-args.pivot_tilt_deg, args.pivot_tilt_deg)
            rz = rng.uniform(-args.pivot_tilt_deg, args.pivot_tilt_deg)
            tilt = (Gf.Rotation(Gf.Vec3d(1, 0, 0), rx)
                    * Gf.Rotation(Gf.Vec3d(0, 1, 0), ry)
                    * Gf.Rotation(Gf.Vec3d(0, 0, 1), rz))
            q = tilt.GetQuat()
            qf = Gf.Quatf(
                float(q.GetReal()),
                Gf.Vec3f(float(q.GetImaginary()[0]),
                         float(q.GetImaginary()[1]),
                         float(q.GetImaginary()[2])),
            )
            try:
                joint_api.CreateLocalRot1Attr().Set(qf)
            except Exception as e:
                print(f"  (localRot1 set failed on {door_prim.GetName()}: {e})", flush=True)

            ox = rng.uniform(-args.pivot_offset_m, args.pivot_offset_m)
            oy = rng.uniform(-args.pivot_offset_m * 0.5, args.pivot_offset_m * 0.5)
            oz = rng.uniform(-args.pivot_offset_m, args.pivot_offset_m)
            try:
                joint_api.CreateLocalPos1Attr().Set(Gf.Vec3f(ox, oy, oz))
            except Exception as e:
                print(f"  (localPos1 set failed on {door_prim.GetName()}: {e})", flush=True)
            pivot_info.append((rx, ry, rz, ox, oy, oz))

        # Give each door an independent flip schedule.
        next_flip = rng.uniform(0.0, FLIP_INTERVAL_SEC)
        broken.append({
            "drive": drive,
            "axis": axis,
            "next_flip_frame": int(next_flip * FPS),
            "target": 0.0,
            "center": center,
        })
    axis_counts = {a: sum(1 for b in broken if b["axis"] == a) for a in ("X", "Y", "Z")}
    print(f"Axis split: {axis_counts}", flush=True)
    if args.pivot_chaos:
        print(f"Pivot chaos enabled: tilt=±{args.pivot_tilt_deg}° offset=±{args.pivot_offset_m} m "
              f"on {len(pivot_info)} doors", flush=True)

    cluster_center = np.mean([b["center"] for b in broken], axis=0)
    if cam_source == "override" or not args.auto_framing:
        # Explicit override or hero preset: respect the framing exactly.
        # No reframing — we already validated this view.
        pass
    else:
        # Auto path: reframe the camera so the chosen door cluster is tight
        # in frame. Aim at cluster_center; pull camera back by 1.5 m.
        to_cluster = cluster_center - cam_pos
        dist = np.linalg.norm(to_cluster)
        if dist > 0.1:
            back_off = 1.5
            cam_pos = cam_pos - (to_cluster / dist) * back_off
        cam_target = cluster_center
    print(f"Final camera: pos={cam_pos.tolist()} target={cam_target.tolist()}", flush=True)
    print(f"Cluster center: {cluster_center.tolist()}", flush=True)

    # Bright interior lighting so the (now tree-free) atrium reads clearly.
    add_bright_lighting(stage, cam_pos, cam_target, boost=args.light_boost)

    orient = look_at_orientation(cam_pos, cam_target)
    camera = Camera(
        prim_path="/World/ChaosCam",
        position=cam_pos,
        frequency=FPS,
        resolution=RESOLUTION,
        orientation=orient,
    )
    # Force wider focal length on the USD camera prim
    cam_prim = stage.GetPrimAtPath("/World/ChaosCam")
    if cam_prim.IsValid():
        UsdGeom.Camera(cam_prim).CreateFocalLengthAttr(args.focal_length)
        print(f"Camera focal length: {args.focal_length} mm", flush=True)

    world.reset()
    camera.initialize()

    # Warm up renderer + settle physics one step
    print("Warming up renderer...", flush=True)
    for _ in range(WARMUP_FRAMES):
        world.step(render=True)

    # Diagnostic: dump a single frame BEFORE the chaos starts so we can
    # verify the camera sees doors at all.
    _diag_rgba = camera.get_rgba()
    if _diag_rgba is not None and _diag_rgba.size > 0:
        try:
            from PIL import Image
            diag_dir = os.path.join(_project_dir, "images", "diagnostics")
            os.makedirs(diag_dir, exist_ok=True)
            diag_path = os.path.join(diag_dir, f"{stem}_chaos_frame0.png")
            Image.fromarray(_diag_rgba[:, :, :3].astype("uint8")).save(diag_path)
            print(f"Diagnostic frame saved: {diag_path}", flush=True)
        except Exception as e:
            print(f"(Could not save diagnostic frame: {e})", flush=True)

    overlay = None
    if not args.no_overlay:
        overlay = StatsOverlay(
            location_lines=HERO_LOCATION_LINES,
            state_label="CORRUPTED",
            state_ok=False,
            door_labels=[f"D{i+1}" for i in range(len(broken))],
        )

    print(f"Recording {TOTAL_FRAMES} frames ({TOTAL_FRAMES / FPS:.1f}s @ {FPS}fps)", flush=True)
    with VideoRecorder(video_name, fps=FPS) as recorder:
        for f in range(TOTAL_FRAMES):
            # Flip door targets on their individual schedules
            for door in broken:
                if f >= door["next_flip_frame"]:
                    door["target"] = rng.choice(FLIP_ANGLES)
                    set_drive_target(door["drive"], door["target"])
                    interval = rng.uniform(
                        FLIP_INTERVAL_SEC - FLIP_JITTER_SEC,
                        FLIP_INTERVAL_SEC + FLIP_JITTER_SEC,
                    )
                    door["next_flip_frame"] = f + max(1, int(interval * FPS))

            # Camera held static — motion made the chaos hard to read. The
            # doors themselves are the moving subject.
            world.step(render=True)
            rgba = camera.get_rgba()
            if rgba is not None and f > 0:
                if overlay is not None:
                    door_angles = [door["target"] for door in broken]
                    rgba = overlay.draw(rgba, door_angles,
                                        frame_idx=f, total_frames=TOTAL_FRAMES)
                recorder.capture_rgba(rgba)

            if f % 60 == 0:
                print(f"  Frame {f}/{TOTAL_FRAMES}", flush=True)

    print(f"Done. Video saved to videos/{video_name}.mp4", flush=True)
    simulation_app.close()


if __name__ == "__main__":
    main()
