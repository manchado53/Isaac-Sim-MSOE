#!/usr/bin/env python3
"""
Rosie Hero Shot — interior screenshot(s) of the candidate Rosie rooms
identified by discover_rosie.py.

Loads the discover_rosie.py JSON, picks the requested room(s), places a
top-corner wide-angle camera, adds interior lighting, and saves one PNG
per room. All captures run in a single Isaac Sim session so we pay the
~40s startup cost once.

Usage:
  # Render three specific rooms by name substring match
  sbatch submit_isaac_job.sh simulations/environment/rosie_preview.py \\
      --rooms SERVICE_153,STORAGE_147,CONTROL_131

  # Render the top 3 server-shape candidates by default
  sbatch submit_isaac_job.sh simulations/environment/rosie_preview.py --top-n 3

Output:
  images/rosie_<room_name>.png         per-room PNG
  logs/rosie_cameras.json              camera poses for reproducibility
"""

import argparse
import json
import os
import sys
import time

parser = argparse.ArgumentParser(description="Render candidate Rosie room shots.")
parser.add_argument(
    "--input",
    default="assets/environments/msoe_building/DierksHallV2_physics.usd",
    help="Path to DierksHallV2_physics.usd",
)
parser.add_argument(
    "--anchors",
    default="logs/dierks_floor2_rooms.json",
    help="JSON from discover_rosie.py (provides per-room bbox).",
)
parser.add_argument(
    "--rooms",
    default="",
    help="Comma-separated room name substrings to render (case-insensitive). "
         "Matches against 'name' field in the anchors JSON. "
         "If empty, uses --top-n server-shape candidates instead.",
)
parser.add_argument(
    "--top-n",
    type=int,
    default=3,
    help="When --rooms is empty, render the top N server-shape candidates.",
)
parser.add_argument(
    "--out-dir",
    default="images",
    help="Directory for output PNGs.",
)
parser.add_argument(
    "--cameras-json",
    default="logs/rosie_cameras.json",
    help="Where to dump camera poses used.",
)
parser.add_argument(
    "--warmup-frames",
    type=int,
    default=30,
    help="Frames to render per shot before capturing (shader warmup).",
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
out_dir = os.path.abspath(args.out_dir) if not os.path.isabs(args.out_dir) else args.out_dir
cameras_json_path = os.path.abspath(args.cameras_json) if not os.path.isabs(args.cameras_json) else args.cameras_json
os.makedirs(out_dir, exist_ok=True)
os.makedirs(os.path.dirname(cameras_json_path), exist_ok=True)


# Camera framing constants.
# Security-cam-style framing matching sim_utils.test_cameras conventions.
# Use proportional insets (fraction of room dimensions) rather than fixed
# meters — a 0.5m inset in a 5m room is too close to the wall and the first
# render of SERVICE_153 showed the camera clipping through geometry.
CORNER_INSET_FRAC = 0.18   # 18% in from each wall — gives ~1m in a 5.5m room
CEILING_INSET_M   = 0.4
FOCAL_LENGTH_MM   = 12.0   # wide-angle, matches the project's reference screenshots
RESOLUTION        = (1920, 1080)


def pick_rooms(anchors, name_substrings, top_n):
    """Select rooms from the anchors JSON.

    If name_substrings is non-empty, return rooms whose 'name' contains any
    substring (case-insensitive). Otherwise return the first `top_n` rooms
    flagged is_server_room_shape.
    """
    rooms = anchors.get("rooms", [])
    if name_substrings:
        wanted = [s.strip().lower() for s in name_substrings if s.strip()]
        selected = []
        for r in rooms:
            nm = r["name"].lower()
            if any(w in nm for w in wanted):
                selected.append(r)
        return selected
    return [r for r in rooms if r.get("is_server_room_shape")][:top_n]


def camera_anchor_for_room(bbox):
    """Return (camera_pos, target, room_center) for a security-cam-style framing.

    Top corner of the room, 12mm wide-angle, aimed at the room's centroid
    (not the opposite corner). Targeting the opposite corner puts the focal
    axis grazing the far wall, which in a small room (SERVICE_153 was 7×5m)
    means the camera clips into the wall right next to it. Aiming at the
    centroid pulls the frame back into the room volume.
    """
    mn = bbox["min"]
    mx = bbox["max"]
    w  = mx[0] - mn[0]
    d  = mx[1] - mn[1]
    inset_x = max(0.5, CORNER_INSET_FRAC * w)
    inset_y = max(0.5, CORNER_INSET_FRAC * d)

    cam = [
        mn[0] + inset_x,
        mn[1] + inset_y,
        mx[2] - CEILING_INSET_M,
    ]
    center = [
        (mn[0] + mx[0]) / 2.0,
        (mn[1] + mx[1]) / 2.0,
        (mn[2] + mx[2]) / 2.0,
    ]
    # Target: slightly past the centroid toward the far corner, at standing
    # height (~1.5m above the floor). Past-centroid keeps the view angled
    # rather than dead-on, but stays well inside the room.
    target = [
        center[0] + 0.25 * (mx[0] - center[0]),
        center[1] + 0.25 * (mx[1] - center[1]),
        mn[2] + 1.5,
    ]
    return cam, target, center


def sanitize(name):
    """Turn a prim name into a file-safe slug."""
    return "".join(ch if (ch.isalnum() or ch in "-_") else "_" for ch in name)


def add_base_lighting(stage):
    """Lights that stay put for every shot: faint sun + dome.

    Interior rooms are windowless so these barely contribute — the per-room
    sphere light (added/replaced per capture) does the real work.
    """
    sun = UsdLux.DistantLight.Define(stage, Sdf.Path("/World/RosieSun"))
    sun.CreateIntensityAttr(1500)
    sun.CreateAngleAttr(1.0)
    UsdGeom.Xformable(sun.GetPrim()).AddRotateXYZOp().Set(Gf.Vec3f(-45, 30, 0))

    UsdLux.DomeLight.Define(
        stage, Sdf.Path("/World/RosieDome")
    ).CreateIntensityAttr(1000)


def move_fill_light(stage, cam_pos):
    """Create (or update) a SphereLight co-located with the camera.

    Rebuilding the prim each call is cheapest — we only do this a handful
    of times per run.
    """
    path = Sdf.Path("/World/RosieFill")
    existing = stage.GetPrimAtPath(path)
    if existing and existing.IsValid():
        stage.RemovePrim(path)
    fill = UsdLux.SphereLight.Define(stage, path)
    fill.CreateIntensityAttr(60000)
    fill.CreateRadiusAttr(1.5)
    UsdGeom.Xformable(fill.GetPrim()).AddTranslateOp().Set(
        Gf.Vec3d(float(cam_pos[0]), float(cam_pos[1]), float(cam_pos[2]))
    )


def move_floor_light(stage, center_xyz, floor_z):
    """Downward-facing RectLight above the room's center, for even floor fill."""
    path = Sdf.Path("/World/RosieFloorWash")
    existing = stage.GetPrimAtPath(path)
    if existing and existing.IsValid():
        stage.RemovePrim(path)
    rect = UsdLux.RectLight.Define(stage, path)
    rect.CreateIntensityAttr(18000)
    rect.CreateWidthAttr(8.0)
    rect.CreateHeightAttr(8.0)
    rx = UsdGeom.Xformable(rect.GetPrim())
    rx.AddTranslateOp().Set(Gf.Vec3d(float(center_xyz[0]), float(center_xyz[1]), float(floor_z) + 2.8))
    rx.AddRotateXYZOp().Set(Gf.Vec3f(-90, 0, 0))


def main():
    t0 = time.time()
    print(f"[rosie_preview] Loading anchors: {anchors_path}", flush=True)
    with open(anchors_path) as f:
        anchors = json.load(f)

    room_substrs = [s for s in args.rooms.split(",") if s.strip()]
    targets = pick_rooms(anchors, room_substrs, args.top_n)
    if not targets:
        print(f"[rosie_preview] ERROR: no rooms matched. rooms={args.rooms!r}, top_n={args.top_n}",
              flush=True)
        simulation_app.close()
        sys.exit(2)
    print(f"[rosie_preview] Will render {len(targets)} room(s):", flush=True)
    for t in targets:
        print(f"  - {t['name']}  ({t['area_m2']:.1f} m²)", flush=True)

    # --- Load building ---
    world = World(stage_units_in_meters=1.0)
    stage = omni.usd.get_context().get_stage()
    print(f"[rosie_preview] Loading building: {input_usd}", flush=True)
    building_prim = stage.DefinePrim("/World/Building", "Xform")
    building_prim.GetReferences().AddReference(
        input_usd, primPath=Sdf.Path("/World/Building")
    )
    setup_physics_scene(stage)
    add_base_lighting(stage)

    # --- Single camera prim we re-point per shot ---
    # We create it once with the first room's pose; the Camera sensor
    # requires a prim to exist at init time.
    first_cam, first_tgt, _ = camera_anchor_for_room(targets[0]["bbox"])
    cam_pos = np.array(first_cam)
    cam_target = np.array(first_tgt)
    orient = look_at_orientation(cam_pos, cam_target)
    camera = Camera(
        prim_path="/World/RosieCam",
        position=cam_pos,
        frequency=30,
        resolution=RESOLUTION,
        orientation=orient,
    )
    cam_prim = stage.GetPrimAtPath("/World/RosieCam")
    UsdGeom.Camera(cam_prim).CreateFocalLengthAttr(float(FOCAL_LENGTH_MM))

    world.reset()
    camera.initialize()

    # --- Render each room ---
    used_cameras = []
    for i, room in enumerate(targets):
        cam, tgt, center = camera_anchor_for_room(room["bbox"])
        print(f"\n[rosie_preview] [{i+1}/{len(targets)}] {room['name']}", flush=True)
        print(f"  cam    = {cam}", flush=True)
        print(f"  target = {tgt}", flush=True)

        camera.set_world_pose(
            position=np.array(cam),
            orientation=look_at_orientation(np.array(cam), np.array(tgt)),
        )
        move_fill_light(stage, cam)
        move_floor_light(stage, center, room["bbox"]["min"][2])

        # Warmup frames let shaders compile and light transport stabilise.
        rgba = None
        for _ in range(args.warmup_frames):
            world.step(render=True)
            rgba = camera.get_rgba()

        if rgba is None or rgba.size == 0:
            print(f"  WARN: empty RGBA for {room['name']}", flush=True)
            continue

        slug = sanitize(room["name"])
        out_path = os.path.join(out_dir, f"rosie_{slug}.png")
        Image.fromarray(rgba[:, :, :3].astype("uint8")).save(out_path)
        print(f"  saved -> {out_path}", flush=True)

        used_cameras.append({
            "room_name": room["name"],
            "prim_path": room["prim_path"],
            "area_m2": room["area_m2"],
            "position": [float(v) for v in cam],
            "target":   [float(v) for v in tgt],
            "focal_length_mm": FOCAL_LENGTH_MM,
            "resolution": list(RESOLUTION),
            "image_path": os.path.relpath(out_path, _project_dir),
        })

    with open(cameras_json_path, "w") as f:
        json.dump({
            "source_usd": os.path.basename(input_usd),
            "anchors_json": os.path.relpath(anchors_path, _project_dir),
            "cameras": used_cameras,
        }, f, indent=2)

    dt = time.time() - t0
    print(f"\n[rosie_preview] Done. {len(used_cameras)} shot(s) in {dt:.1f}s", flush=True)
    print(f"[rosie_preview] Poses -> {cameras_json_path}", flush=True)

    simulation_app.close()


if __name__ == "__main__":
    main()
