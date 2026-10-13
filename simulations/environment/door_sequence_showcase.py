#!/usr/bin/env python3
"""
Door Sequence Showcase — multiple doors opening in a clean wave.

Loads the final MSOE Innovation Center physics USD, picks the camera with
the densest door cluster, and animates every chosen door opening in a
staggered sequence (wave). All physics parameters stay at their production
values (Z-axis hinges, proper damping, angle limits) — this is the
"perfect pipeline" payoff shot.

Usage:
    sbatch scripts/slurm/submit_isaac_job.sh simulations/environment/door_sequence_showcase.py \
        --input assets/environments/msoe_building/MSOEBuildingV4_physics.usd

Output:
    - videos/environment/<stem>_door_sequence.mp4 (15s @ 30fps, 1920x1080)
"""

import argparse
import math
import os
import sys

parser = argparse.ArgumentParser(description="Staggered door-opening showcase.")
parser.add_argument("--input", required=True, help="Path to physics-ready USD file")
parser.add_argument("--video", default=None, help="Output video name override")
parser.add_argument("--num-doors", type=int, default=10,
                    help="Number of doors to animate near the chosen camera")
parser.add_argument("--cam-pos", default=None,
                    help="Explicit camera position 'X,Y,Z' (meters). "
                         "If provided with --cam-target, skips auto-discovery.")
parser.add_argument("--cam-target", default=None,
                    help="Explicit look-at target 'X,Y,Z' (meters).")
parser.add_argument("--focal-length", type=float, default=18.0,
                    help="Camera focal length in mm (default 18 = wider FOV).")
parser.add_argument(
    "--hide-patterns",
    default="Tree,Planting,Vegetation,Entourage,RPC,Plant,Shrub,Bush,Landscape,Topography,Site_",
    help="Comma-separated substrings; matching prims under /World/Building are hidden.",
)
parser.add_argument("--light-boost", type=float, default=1.0,
                    help="Multiplier on interior lighting intensities (default 1.0).")
parser.add_argument("--no-overlay", action="store_true",
                    help="Skip the top-right stats HUD (useful for regression diffs).")
parser.add_argument("--auto-framing", action="store_true",
                    help="When no --cam-pos override is given, recentre on the door "
                         "cluster (old behaviour). Default: use the hero cam preset.")
args, _unknown = parser.parse_known_args()


def parse_xyz(s):
    import numpy as _np
    parts = [p.strip() for p in s.split(",")]
    if len(parts) != 3:
        raise ValueError(f"Expected 'X,Y,Z', got {s!r}")
    return _np.array([float(p) for p in parts])

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
TOTAL_FRAMES = 450           # 15 s
WARMUP_FRAMES = 30
RESOLUTION = (1920, 1080)

OPEN_ANGLE_DEG = 60.0         # reduced from 85° to avoid clipping perpendicular walls
STAGGER_FRAMES = 12           # 0.4 s between successive door starts
OPEN_DURATION_FRAMES = 60     # 2 s for one door to swing open
HOLD_FRAMES = 90              # 3 s at fully open

DOLLY_DISTANCE_M = 1.2        # camera dolly-in over the full clip

# Hero preset for MSOEBuildingV4_physics.usd — cluster #1 (27 doors on Floor 3
# south corridor) viewed from the West. 4+ red-framed doors fit in frame down
# the corridor with no wall occlusion. Source: tools/preview_door_cameras.py
# (W angle) → logs/MSOEBuildingV4_physics_camera_preview.txt.
HERO_CAM_POS = (-25.59, 78.17, 24.75)
HERO_CAM_TARGET = (-16.50, 78.17, 23.75)
HERO_LOCATION_LINES = (
    "MSOE Innovation Center",
    "Floor 3 · South Corridor",
)


def ease_in_out(t):
    """Cubic ease-in-out on [0, 1]."""
    t = max(0.0, min(1.0, t))
    if t < 0.5:
        return 4 * t * t * t
    return 1 - (-2 * t + 2) ** 3 / 2


def _quat_signed_angle_deg(q, axis_char):
    """Signed rotation angle (deg) about the joint's axis, from a quaternion."""
    if q is None:
        return 0.0
    w = float(q.GetReal())
    imag = q.GetImaginary()
    v = np.array([float(imag[0]), float(imag[1]), float(imag[2])])
    w = max(-1.0, min(1.0, w))
    ang = 2.0 * math.acos(w)
    s = math.sqrt(max(0.0, 1.0 - w * w))
    axis = v / s if s > 1e-6 else np.array([0.0, 0.0, 1.0])
    idx = {"X": 0, "Y": 1, "Z": 2}.get(axis_char, 2)
    if abs(axis[idx]) < 0.1:
        return 0.0
    return float(math.degrees(ang) * np.sign(axis[idx]))


def read_joint_angle_deg(door_prim, axis_char):
    """Read the door's current rotation about the joint axis."""
    xform = UsdGeom.Xformable(door_prim)
    for op in xform.GetOrderedXformOps():
        if op.GetOpType() == UsdGeom.XformOp.TypeOrient:
            return _quat_signed_angle_deg(op.Get(), axis_char)
    return 0.0


# ---------------------------------------------------------------------------
# Door discovery (shared shape with door_chaos_showcase.py)
# ---------------------------------------------------------------------------

def find_door_joint(door_prim):
    for child in door_prim.GetChildren():
        name = child.GetName()
        if "Hinge" in name or "Joint" in name:
            if child.IsA(UsdPhysics.RevoluteJoint):
                return child
    return None


def find_drive_on_joint(joint_prim):
    drive = UsdPhysics.DriveAPI(joint_prim, "angular")
    if drive.GetTargetPositionAttr().IsValid():
        return drive
    return None


def iter_doors(stage):
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
    distant = UsdLux.DistantLight.Define(stage, Sdf.Path("/World/DoorSun"))
    distant.CreateIntensityAttr(3500.0 * boost)
    distant.CreateAngleAttr(1.0)
    UsdGeom.Xformable(distant.GetPrim()).AddRotateXYZOp().Set(Gf.Vec3f(-45, 30, 0))

    dome = UsdLux.DomeLight.Define(stage, Sdf.Path("/World/DoorDome"))
    dome.CreateIntensityAttr(6000.0 * boost)

    cam_pos = np.asarray(cam_pos, dtype=float)
    cam_target = np.asarray(cam_target, dtype=float)

    fx1 = UsdLux.SphereLight.Define(stage, Sdf.Path("/World/DoorCeilingA"))
    fx1.CreateIntensityAttr(25000.0 * boost)
    fx1.CreateRadiusAttr(1.5)
    UsdGeom.Xformable(fx1.GetPrim()).AddTranslateOp().Set(
        Gf.Vec3f(float(cam_target[0]), float(cam_target[1]), float(cam_target[2]) + 4.0)
    )

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


def choose_camera_with_most_doors(cameras, door_records, search_radius=15.0):
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
    input_usd = os.path.abspath(args.input)
    stem = os.path.splitext(os.path.basename(input_usd))[0]
    video_name = args.video if args.video else f"environment/{stem}_door_sequence"

    print("=" * 70, flush=True)
    print("Door Sequence Showcase — staggered opening wave", flush=True)
    print(f"Input: {input_usd}", flush=True)
    print(f"Output: videos/{video_name}.mp4", flush=True)
    print("=" * 70, flush=True)

    world = World(stage_units_in_meters=1.0)
    stage = omni.usd.get_context().get_stage()

    building_prim = stage.DefinePrim("/World/Building", "Xform")
    building_prim.GetReferences().AddReference(input_usd, Sdf.Path("/World/Building"))
    print("Building loaded.", flush=True)

    hide_patterns = [p.strip() for p in args.hide_patterns.split(",") if p.strip()]
    if hide_patterns:
        total, per_pat = hide_patterns_in_building(stage, hide_patterns)
        print(f"Hide patterns: {hide_patterns}", flush=True)
        for pat, cnt in per_pat.items():
            if cnt:
                print(f"  '{pat}' -> {cnt}", flush=True)
        print(f"Total hidden: {total}", flush=True)

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

    if args.cam_pos and args.cam_target:
        cam_pos = parse_xyz(args.cam_pos)
        cam_target = parse_xyz(args.cam_target)
        cam_source = "override"
        print(f"Using explicit camera: pos={cam_pos.tolist()} target={cam_target.tolist()}",
              flush=True)
        pivot = cam_target
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

    door_records.sort(key=lambda r: np.linalg.norm(r[3] - pivot))
    # Over-select: pick 2× the target count. After physics settles we verify
    # each one and drop any that are stuck in walls. This keeps the visible
    # animation to doors that actually track their drive commands.
    candidate_count = max(args.num_doors * 2, args.num_doors + 6)
    candidates = door_records[: candidate_count]
    print(f"Selecting from {len(candidates)} candidates to find "
          f"{args.num_doors} verified-working doors.", flush=True)

    # Use candidate centroid for framing so the reframe heuristic still works.
    cluster_center = np.mean([r[3] for r in candidates], axis=0)
    if cam_source == "override" or not args.auto_framing:
        # Explicit override or hero preset — respect the framing exactly.
        pass
    else:
        to_cluster = cluster_center - cam_pos
        dist = np.linalg.norm(to_cluster)
        if dist > 0.1:
            back_off = 1.5
            cam_pos = cam_pos - (to_cluster / dist) * back_off
        cam_target = cluster_center
    forward_dir = (cam_target - cam_pos)
    fwd_norm = np.linalg.norm(forward_dir)
    forward_dir = forward_dir / fwd_norm if fwd_norm > 1e-6 else np.array([1.0, 0.0, 0.0])
    print(f"Final camera: pos={cam_pos.tolist()} target={cam_target.tolist()}", flush=True)
    print(f"Cluster center: {cluster_center.tolist()}", flush=True)

    add_bright_lighting(stage, cam_pos, cam_target, boost=args.light_boost)

    orient = look_at_orientation(cam_pos, cam_target)
    camera = Camera(
        prim_path="/World/SequenceCam",
        position=cam_pos,
        frequency=FPS,
        resolution=RESOLUTION,
        orientation=orient,
    )
    cam_prim = stage.GetPrimAtPath("/World/SequenceCam")
    if cam_prim.IsValid():
        UsdGeom.Camera(cam_prim).CreateFocalLengthAttr(args.focal_length)
        print(f"Camera focal length: {args.focal_length} mm", flush=True)

    world.reset()
    camera.initialize()

    print("Warming up renderer...", flush=True)
    for _ in range(WARMUP_FRAMES):
        world.step(render=True)

    # Diagnostic frame — confirm the camera actually sees doors.
    _diag_rgba = camera.get_rgba()
    if _diag_rgba is not None and _diag_rgba.size > 0:
        try:
            from PIL import Image
            diag_dir = os.path.join(_project_dir, "images", "diagnostics")
            os.makedirs(diag_dir, exist_ok=True)
            diag_path = os.path.join(diag_dir, f"{stem}_sequence_frame0.png")
            Image.fromarray(_diag_rgba[:, :, :3].astype("uint8")).save(diag_path)
            print(f"Diagnostic frame saved: {diag_path}", flush=True)
        except Exception as e:
            print(f"(Could not save diagnostic frame: {e})", flush=True)

    # Self-healing door selection: after warmup, verify each candidate tracks
    # its drive (commanded target=0°). Any door whose actual angle is >5° from
    # 0° is physics-stuck (body spawned inside wall geometry — see
    # logs/door_diagnosis_*.txt). Drop it AND hide its mesh so it doesn't
    # appear as a clipped "phantom" in the background. Keep first N that pass.
    STUCK_DEG = 5.0
    kept = []
    dropped = []
    for door_prim, joint, drive, center in candidates:
        axis_char = UsdPhysics.RevoluteJoint(joint).GetAxisAttr().Get() or "Z"
        angle = read_joint_angle_deg(door_prim, axis_char)
        if abs(angle) <= STUCK_DEG:
            kept.append((door_prim, joint, drive, center))
            if len(kept) >= args.num_doors:
                break
        else:
            dropped.append((door_prim, angle))

    # Hide *any* physics door within 20 m of the pivot that is visibly stuck,
    # not just the ones we tested. This catches background phantoms like
    # Doors_10649156 that we never selected but would show up in frame.
    hidden_phantoms = 0
    SCENE_R = 20.0
    for door_prim, joint, _drive, center in door_records:
        if np.linalg.norm(center - pivot) > SCENE_R:
            continue
        if any(door_prim is k[0] for k in kept):
            continue
        axis_char = UsdPhysics.RevoluteJoint(joint).GetAxisAttr().Get() or "Z"
        angle = read_joint_angle_deg(door_prim, axis_char)
        if abs(angle) > STUCK_DEG:
            img = UsdGeom.Imageable(door_prim)
            if img:
                img.CreateVisibilityAttr().Set(UsdGeom.Tokens.invisible)
            hidden_phantoms += 1

    print(f"Self-heal: kept {len(kept)}/{len(candidates)} candidates "
          f"(dropped {len(dropped)} stuck); hid {hidden_phantoms} phantom doors.",
          flush=True)
    for door_prim, ang in dropped[:5]:
        print(f"  dropped @ {ang:+.1f}°  {door_prim.GetPath()}", flush=True)

    if len(kept) < 2:
        print(f"ERROR: only {len(kept)} working doors found — aborting.", flush=True)
        simulation_app.close()
        return

    chosen = kept
    print(f"Final chosen set: {len(chosen)} doors", flush=True)

    # Build per-door schedules
    open_start_frame = 30  # 1 s of held-closed establishing shot
    schedules = []
    for i, (door_prim, joint, drive, center) in enumerate(chosen):
        t_open = open_start_frame + i * STAGGER_FRAMES
        schedules.append({
            "drive": drive,
            "open_start": t_open,
            "open_end": t_open + OPEN_DURATION_FRAMES,
            "close_start": None,  # filled in below
            "close_end": None,
            "name": door_prim.GetName(),
        })

    last_open_end = max(s["open_end"] for s in schedules)
    close_start_first = last_open_end + HOLD_FRAMES
    # Close in reverse order (last-opened closes first).
    for i, s in enumerate(reversed(schedules)):
        s["close_start"] = close_start_first + i * STAGGER_FRAMES
        s["close_end"] = s["close_start"] + OPEN_DURATION_FRAMES
    final_frame = max(s["close_end"] for s in schedules)
    if final_frame > TOTAL_FRAMES:
        print(f"WARNING: schedule ends at frame {final_frame}, "
              f"clip is {TOTAL_FRAMES} frames — truncating hold.", flush=True)

    def angle_for(sched, frame):
        if frame < sched["open_start"]:
            return 0.0
        if frame < sched["open_end"]:
            t = (frame - sched["open_start"]) / OPEN_DURATION_FRAMES
            return ease_in_out(t) * OPEN_ANGLE_DEG
        if frame < sched["close_start"]:
            return OPEN_ANGLE_DEG
        if frame < sched["close_end"]:
            t = (frame - sched["close_start"]) / OPEN_DURATION_FRAMES
            return OPEN_ANGLE_DEG * (1.0 - ease_in_out(t))
        return 0.0

    overlay = None
    if not args.no_overlay:
        overlay = StatsOverlay(
            location_lines=HERO_LOCATION_LINES,
            state_label="OPERATIONAL",
            state_ok=True,
            door_labels=[f"D{i+1}" for i in range(len(chosen))],
        )

    print(f"Recording {TOTAL_FRAMES} frames ({TOTAL_FRAMES / FPS:.1f}s @ {FPS}fps)", flush=True)
    with VideoRecorder(video_name, fps=FPS) as recorder:
        for f in range(TOTAL_FRAMES):
            door_angles = [angle_for(sched, f) for sched in schedules]
            for sched, angle in zip(schedules, door_angles):
                set_drive_target(sched["drive"], angle)

            # Camera held static so the wave of opening doors reads cleanly.
            world.step(render=True)
            rgba = camera.get_rgba()
            if rgba is not None and f > 0:
                if overlay is not None:
                    rgba = overlay.draw(rgba, door_angles,
                                        frame_idx=f, total_frames=TOTAL_FRAMES)
                recorder.capture_rgba(rgba)

            if f % 60 == 0:
                print(f"  Frame {f}/{TOTAL_FRAMES}", flush=True)

    print(f"Done. Video saved to videos/{video_name}.mp4", flush=True)
    simulation_app.close()


if __name__ == "__main__":
    main()
