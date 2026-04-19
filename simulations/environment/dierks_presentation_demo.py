#!/usr/bin/env python3
"""
Dierks Hall Presentation Demo — 8 presenters on stage + a podium.

Loads the auditorium, drops a podium on the stage-left side, spawns 8
standing characters along the stage front, and renders one screenshot
from the audience camera discovered by discover_auditorium.py.

Usage:
  sbatch submit_isaac_job.sh simulations/environment/dierks_presentation_demo.py \\
      --input assets/environments/msoe_building/DierksHallV2_physics.usd \\
      --anchors logs/dierks_auditorium.json

Output:
  images/dierks_presentation_demo.png
"""

import argparse
import json
import math
import os
import sys

parser = argparse.ArgumentParser(description="Dierks Hall presentation screenshot/video.")
parser.add_argument("--input", required=True, help="Path to DierksHallV2_physics.usd")
parser.add_argument("--anchors", default="logs/dierks_auditorium.json",
                    help="JSON from discover_auditorium.py")
parser.add_argument("--out", default="images/dierks_presentation_demo.png",
                    help="Output PNG path")
parser.add_argument("--warmup-frames", type=int, default=30,
                    help="Frames to render before capturing")
parser.add_argument("--video", action="store_true",
                    help="Also record an MP4 with a dolly-in camera move")
parser.add_argument("--video-name", default="environment/dierks_presentation_demo",
                    help="Video output name (relative to videos/, no extension)")
parser.add_argument("--record-frames", type=int, default=180,
                    help="Frames to record (default 180 = 6s @ 30fps)")
parser.add_argument("--dolly-start-offset", type=float, default=-1.5,
                    help="Camera starts this many meters back from the discovered position")
parser.add_argument("--dolly-end-offset", type=float, default=2.5,
                    help="Camera ends this many meters forward from the discovered position")
args, _unknown = parser.parse_known_args()

os.environ["OMNI_KIT_ACCEPT_EULA"] = "YES"

from isaacsim import SimulationApp
simulation_app = SimulationApp({"headless": True})

import numpy as np
from PIL import Image

import omni.usd
from pxr import Sdf, Usd, UsdLux, UsdGeom, Gf
from isaacsim.core.api import World
from isaacsim.core.api.objects import VisualCuboid
from isaacsim.core.utils import prims as prims_utils
from isaacsim.sensors.camera import Camera
from isaacsim.storage.native import get_assets_root_path

_script_dir = os.path.dirname(os.path.abspath(__file__))
_project_dir = os.path.abspath(os.path.join(_script_dir, '..', '..'))
sys.path.insert(0, _project_dir)

from sim_utils.building_physics import setup_physics_scene
from sim_utils.test_cameras import look_at_orientation
from sim_utils.video_capture import VideoRecorder

# ---------- Paths ----------
input_usd = os.path.abspath(args.input)
anchors_path = args.anchors if os.path.isabs(args.anchors) else os.path.join(_project_dir, args.anchors)
out_path = args.out if os.path.isabs(args.out) else os.path.join(_project_dir, args.out)
os.makedirs(os.path.dirname(out_path), exist_ok=True)


# ---------- Character catalog ----------
# Each slot is a list of candidate USD subpaths. We try them in order and
# use the first one whose reference actually resolves (has child prims after
# load). Last entry in each list is the universal Biped_Setup fallback.
#
# The Isaac Sim 5.1 character catalog is different from 4.x — prior "known
# working" names like M_Business_02 turned out to be absent in 5.1. Rather
# than guess, we list several plausible names per slot and let runtime pick.
_F_BUSINESS = "/Isaac/People/Characters/F_Business_02/F_Business_02.usd"
_F_MEDICAL  = "/Isaac/People/Characters/F_Medical_01/F_Medical_01.usd"
_BIPED      = "/Isaac/People/Characters/Biped_Setup.usd"

# Candidate variants worth probing in 5.1 (will silently skip missing ones).
_M_CANDIDATES = [
    "/Isaac/People/Characters/original_male/original_male.usd",
    "/Isaac/People/Characters/M_Business_02/M_Business_02.usd",
    "/Isaac/People/Characters/M_Business_01/M_Business_01.usd",
    "/Isaac/People/Characters/M_Worker_01/M_Worker_01.usd",
    "/Isaac/People/Characters/M_Police_04/M_Police_04.usd",
    "/Isaac/People/Characters/M_Construction_05/M_Construction_05.usd",
]
_F_CANDIDATES = [
    _F_BUSINESS,
    _F_MEDICAL,
    "/Isaac/People/Characters/F_Business_01/F_Business_01.usd",
    "/Isaac/People/Characters/F_Medical_02/F_Medical_02.usd",
    "/Isaac/People/Characters/original_female/original_female.usd",
]

# Only the two characters that actually resolve in Isaac Sim 5.1 assets —
# Biped_Setup loaded but came in with a broken pivot (either collapsed flat or
# sunk below the stage), so we skip it. Alternate F_Business_02 and
# F_Medical_01 across the row for some visual variety.
CHARACTER_SLOTS = [
    ([_F_BUSINESS], "business_f"),
    ([_F_MEDICAL],  "medical_f"),
    ([_F_BUSINESS], "business_f_2"),
    ([_F_MEDICAL],  "medical_f_2"),
    ([_F_BUSINESS], "business_f_3"),
    ([_F_MEDICAL],  "medical_f_3"),
    ([_F_BUSINESS], "business_f_4"),
    ([_F_MEDICAL],  "medical_f_4"),
]
BIPED_FALLBACK_SUBPATH = _BIPED

# Z offset: characters' root prim sits above the foot by ~0m on the NVIDIA
# biped (scale=1.0, metersPerUnit=1.0). Leave offset at 0 and measure with a
# post-warmup bbox dump if feet clip or float.
CHARACTER_Z_OFFSET = 0.0

# ---------- Podium geometry ----------
# Simple lectern: ~0.5m deep × 0.6m wide × 1.1m tall, dark wood color.
# Placed at stage-left (lower Y) near the front edge, so it frames the scene
# without hiding the presenters.
PODIUM_SIZE = (0.5, 0.6, 1.1)      # (X, Y, Z)
PODIUM_COLOR = (0.28, 0.16, 0.08)  # dark walnut


def load_standing_character(stage, prim_path, character_usd, position, face_target_xy):
    """Load a standing character — translate on ROOT, rotateXYZ on FIRST CHILD.

    Taken from owen-people-data:people_anomaly_scene_demo.py::load_standing_character.
    The metrics-assembler ignores root-level rotation/scale, so they must go
    on the first Xformable child.
    """
    try:
        person = prims_utils.create_prim(prim_path, "Xform", usd_path=character_usd)
    except Exception as e:
        print(f"[present] ERROR loading {prim_path} from {character_usd}: {e}", flush=True)
        return None
    if not person or not person.IsValid():
        return None

    dx = float(face_target_xy[0]) - float(position[0])
    dy = float(face_target_xy[1]) - float(position[1])
    face_angle_deg = math.degrees(math.atan2(dy, dx))

    xformable = UsdGeom.Xformable(person)
    xformable.ClearXformOpOrder()
    xformable.AddTranslateOp(precision=UsdGeom.XformOp.PrecisionDouble).Set(
        Gf.Vec3d(float(position[0]), float(position[1]), float(position[2]))
    )

    for child in person.GetChildren():
        if not child.IsA(UsdGeom.Xformable):
            continue
        r = child.GetAttribute("xformOp:rotateXYZ")
        if r and r.IsValid():
            r.Set(Gf.Vec3f(0.0, 0.0, float(face_angle_deg)))
        s = child.GetAttribute("xformOp:scale")
        if s and s.IsValid():
            cur = s.Get()
            s.Set(type(cur)(1.0, 1.0, 1.0))
        break  # first Xformable child only

    print(f"[present] Standing {prim_path} at "
          f"({position[0]:.2f},{position[1]:.2f},{position[2]:.2f}) "
          f"facing {face_angle_deg:.0f}°", flush=True)
    return person


def _prim_actually_loaded(prim):
    """Return True iff the USD reference resolved (prim has descendants).

    `prims_utils.create_prim` with a bad USD URL succeeds and returns a
    valid (but empty) Xform. We detect the difference by checking for
    child prims — a resolved biped always has SkelRoot/Skeleton/Meshes.
    """
    if prim is None or not prim.IsValid():
        return False
    return len(list(prim.GetChildren())) > 0


def _remove_prim_if_empty(stage, prim_path):
    """Delete a failed-load prim so the next candidate can re-use the path."""
    from pxr import Sdf
    stage.RemovePrim(Sdf.Path(prim_path))


def load_with_fallback(stage, prim_path, asset_root, candidates, position, face_target_xy):
    """Try each candidate USD in order; return the first that actually loads.

    After create_prim, we check that the prim has children. Empty =
    reference didn't resolve; delete the empty prim and try the next.
    If all candidates fail, fall back to Biped_Setup.
    """
    # Skip Biped_Setup fallback: it loads but its pivot is misaligned for our
    # transform logic (collapses flat or sinks below the stage). If the named
    # candidates all fail we'd rather leave that slot empty than add a broken prim.
    for subpath in list(candidates):
        url = asset_root + subpath
        p = load_standing_character(stage, prim_path, url, position, face_target_xy)
        if _prim_actually_loaded(p):
            return p, subpath
        _remove_prim_if_empty(stage, prim_path)
    print(f"[present] ALL candidates failed for {prim_path}", flush=True)
    return None, None


def main():
    print(f"Loading anchors: {anchors_path}", flush=True)
    with open(anchors_path) as f:
        anchors = json.load(f)

    # Camera
    cam_spec = anchors["camera"]
    cam_pos = np.array(cam_spec["position"])
    cam_target = np.array(cam_spec["target"])
    focal_mm = cam_spec.get("focal_length_mm", 18.0)
    resolution = tuple(cam_spec.get("resolution", [1920, 1080]))

    # Anchors
    standing_anchors = anchors["standing_anchors"]
    stage_bbox = anchors["stage"]["bbox"]
    stage_top_z = stage_bbox["max"][2]
    stage_center = stage_bbox["center"]

    # Depth axis = 0 (X) from discovery, row_axis=1 (Y), edge_sign=-1 (audience is -X).
    axes = anchors["axes"]
    depth_axis = axes["depth_axis"]
    row_axis = axes["row_axis"]
    edge_sign = axes["edge_sign"]

    # --- World / building ---
    world = World(stage_units_in_meters=1.0)
    stage = omni.usd.get_context().get_stage()

    print(f"Loading building: {input_usd}", flush=True)
    building_prim = stage.DefinePrim("/World/Building", "Xform")
    building_prim.GetReferences().AddReference(
        input_usd, primPath=Sdf.Path("/World/Building")
    )
    setup_physics_scene(stage)

    # --- Lighting (same as preview) ---
    sun = UsdLux.DistantLight.Define(stage, Sdf.Path("/World/Sun"))
    sun.CreateIntensityAttr(2000)
    sun.CreateAngleAttr(1.0)
    UsdGeom.Xformable(sun.GetPrim()).AddRotateXYZOp().Set(Gf.Vec3f(-45, 30, 0))

    UsdLux.DomeLight.Define(stage, Sdf.Path("/World/DomeLight")).CreateIntensityAttr(1500)

    rect = UsdLux.RectLight.Define(stage, Sdf.Path("/World/StageWash"))
    rect.CreateIntensityAttr(25000)
    rect.CreateWidthAttr(15.0)
    rect.CreateHeightAttr(4.0)
    rx = UsdGeom.Xformable(rect.GetPrim())
    rx.AddTranslateOp().Set(Gf.Vec3d(stage_center[0], stage_center[1], stage_top_z + 4.0))
    rx.AddRotateXYZOp().Set(Gf.Vec3f(-90, 0, 0))

    fill = UsdLux.SphereLight.Define(stage, Sdf.Path("/World/CamFill"))
    fill.CreateIntensityAttr(20000)
    fill.CreateRadiusAttr(3.0)
    UsdGeom.Xformable(fill.GetPrim()).AddTranslateOp().Set(
        Gf.Vec3d(float(cam_pos[0]), float(cam_pos[1]), float(cam_pos[2]))
    )

    # --- Podium ---
    # Place at stage-left (audience's left, so lower Y), close to the stage
    # face so it looks like the front-of-stage lectern position.
    # Stand the podium ON the stage floor (center at stage_top + half_height).
    podium_w_x, podium_w_y, podium_w_z = PODIUM_SIZE  # width along X,Y,Z
    # Use the leftmost standing anchor (first one) as the reference:
    #   X = anchor X, Y = anchor Y - 1.2m (audience-left of that presenter)
    leftmost = standing_anchors[0]
    podium_x = leftmost[0]                  # same depth as presenters
    podium_y = leftmost[row_axis] - 1.2     # 1.2m to audience-left of the first presenter
    podium_z = stage_top_z + podium_w_z / 2.0

    print(f"[present] Podium at ({podium_x:.2f}, {podium_y:.2f}, {podium_z:.2f}), "
          f"size={PODIUM_SIZE}", flush=True)
    VisualCuboid(
        prim_path="/World/Podium",
        name="podium",
        position=np.array([podium_x, podium_y, podium_z]),
        scale=np.array(PODIUM_SIZE),
        color=np.array(PODIUM_COLOR),
    )

    # --- 8 presenters ---
    # All face the audience camera's XY position so they look toward the viewer.
    face_target_xy = (cam_pos[0], cam_pos[1])
    asset_root = get_assets_root_path()
    if not asset_root:
        print("[present] ERROR: get_assets_root_path() returned empty", flush=True)
        simulation_app.close()
        sys.exit(1)
    print(f"[present] Isaac assets root: {asset_root}", flush=True)

    for i, anchor in enumerate(standing_anchors):
        candidates, label = CHARACTER_SLOTS[i % len(CHARACTER_SLOTS)]
        prim_path = f"/World/Presenters/Presenter_{i:02d}_{label}"
        pos = (anchor[0], anchor[1], anchor[2] + CHARACTER_Z_OFFSET)
        _, loaded = load_with_fallback(stage, prim_path, asset_root, candidates, pos, face_target_xy)
        if loaded:
            print(f"[present]   -> slot {i} used: {loaded}", flush=True)

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
    for _ in range(args.warmup_frames):
        world.step(render=True)
        rgba = camera.get_rgba()
    if rgba is None:
        print("ERROR: camera returned no RGBA data", flush=True)
        simulation_app.close()
        sys.exit(1)

    rgb = rgba[:, :, :3].astype("uint8")
    Image.fromarray(rgb).save(out_path)
    print(f"Saved: {out_path}  ({rgb.shape[1]}x{rgb.shape[0]})", flush=True)

    # --- Optional video with a gentle dolly-in ---
    # Camera stays on the line from start_pos → end_pos (pure X translation in
    # our scene since the discovered camera is on the stage's depth axis),
    # target stays fixed on the stage chest height.
    if args.video:
        start_pos = cam_pos + np.array([args.dolly_start_offset, 0.0, 0.0])
        end_pos   = cam_pos + np.array([args.dolly_end_offset,   0.0, 0.0])
        print(f"\n[present] Recording {args.record_frames} frames "
              f"({args.record_frames/30:.1f}s @ 30fps)", flush=True)
        print(f"  start_pos = {start_pos}", flush=True)
        print(f"  end_pos   = {end_pos}", flush=True)
        print(f"  target    = {cam_target}", flush=True)

        with VideoRecorder(args.video_name, fps=30) as recorder:
            for i in range(args.record_frames):
                t = i / max(args.record_frames - 1, 1)   # 0..1 linear
                # Ease in/out (smoothstep) so motion starts and stops gently.
                t_smooth = t * t * (3.0 - 2.0 * t)
                pos = start_pos + t_smooth * (end_pos - start_pos)
                orient_i = look_at_orientation(pos, cam_target)
                camera.set_world_pose(position=pos, orientation=orient_i)
                # Move the camera-fill light with the camera so the scene
                # stays consistently lit from the viewer's POV.
                fill_xf = UsdGeom.Xformable(stage.GetPrimAtPath("/World/CamFill"))
                ops = fill_xf.GetOrderedXformOps()
                if ops:
                    ops[0].Set(Gf.Vec3d(float(pos[0]), float(pos[1]), float(pos[2])))

                world.step(render=True)
                frame_rgba = camera.get_rgba()
                if frame_rgba is not None and i > 0:
                    recorder.capture_rgba(frame_rgba)

                if i % 30 == 0:
                    print(f"  frame {i}/{args.record_frames}", flush=True)

        print(f"Video saved: videos/{args.video_name}.mp4", flush=True)

    # Diagnostic: dump the post-warmup bbox of each presenter to check Z placement.
    print("\n[present] Post-warmup Z-check (bbox min/max):", flush=True)
    bbox_cache = UsdGeom.BBoxCache(Usd.TimeCode.Default(), [UsdGeom.Tokens.default_])
    presenters_root = stage.GetPrimAtPath("/World/Presenters")
    if presenters_root and presenters_root.IsValid():
        for p in presenters_root.GetChildren():
            bb = bbox_cache.ComputeWorldBound(p).ComputeAlignedRange()
            if not bb.IsEmpty():
                mn = bb.GetMin()
                mx = bb.GetMax()
                print(f"  {p.GetName():<30s}  Z=[{mn[2]:.3f}, {mx[2]:.3f}]  "
                      f"height={mx[2]-mn[2]:.2f}", flush=True)

    simulation_app.close()


if __name__ == "__main__":
    main()
