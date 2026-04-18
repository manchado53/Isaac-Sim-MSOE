#!/usr/bin/env python3
"""
People Walking Demo — Character moves along a path on the Innovation Center ground floor.

Baby step #2 of people integration: take the static character from people_demo.py
and make them move along a predefined 3-waypoint path across the Event Floor.

Uses Option A + programmatic translation (W1): update xformOp:translate and
xformOp:orient each frame to interpolate along a linear path. Character will
slide smoothly through space — without omni.anim.people extension enabled
they may stay in a frozen pose, but the spatial/rendering pipeline is proven.

Escalation paths after this works:
  - W2: enable omni.anim.graph.core + omni.anim.people for walking animation
  - W3: full IRA sdg_scheduler.py path with NavMesh

Output:
    videos/environment/people_walking_demo.mp4
    images/people_walking_demo_start.png
    images/people_walking_demo_end.png
"""

import os
import sys
import math

os.environ["OMNI_KIT_ACCEPT_EULA"] = "YES"

from isaacsim import SimulationApp
simulation_app = SimulationApp({"headless": True})

import numpy as np
import carb
import omni.usd
from pxr import Usd, UsdGeom, UsdPhysics, Sdf, Gf, UsdLux

_script_dir = os.path.dirname(os.path.abspath(__file__))
_project_dir = os.path.abspath(os.path.join(_script_dir, '..', '..'))
sys.path.insert(0, _project_dir)

from isaacsim.core.api import World
from isaacsim.core.utils import prims as prims_utils
from isaacsim.storage.native import get_assets_root_path

from sim_utils.test_cameras import (
    TEST_CAMERAS,
    create_test_camera,
    add_interior_lighting,
    set_focal_length_on_prim,
)
from sim_utils.video_capture import VideoRecorder
from sim_utils.building_physics import setup_physics_scene


# --- Ground floor camera "G": Event Floor IC 100 ---
CAM_KEY = "G"
CAM = TEST_CAMERAS[CAM_KEY]
FLOOR_Z = 14.56  # ground floor height in feet

# Waypoints along a path across the Event Floor (in feet).
# Camera looks from (-42.15, 68.44) toward (-27.42, 83.72).
# We'll walk from the far side of the room, across the middle, to the near side,
# keeping everything in camera FOV.
WAYPOINTS = [
    (-38.0, 82.0, FLOOR_Z),   # far-left (visible from camera)
    (-32.0, 78.0, FLOOR_Z),   # middle (near previous static demo position)
    (-28.0, 72.0, FLOOR_Z),   # near-right (closer to camera)
]

# Speed / timing
WALK_SPEED_FT_PER_SEC = 3.0   # ~1 m/s
FPS = 30
WARMUP_FRAMES = 30

# Character asset
CHARACTER_SUBPATH = "/Isaac/People/Characters/F_Business_02/F_Business_02.usd"
BIPED_FALLBACK_SUBPATH = "/Isaac/People/Characters/Biped_Setup.usd"

# Output
VIDEO_NAME = "environment/people_walking_demo"
SCREENSHOT_START = os.path.join(_project_dir, "images", "people_walking_demo_start.png")
SCREENSHOT_END = os.path.join(_project_dir, "images", "people_walking_demo_end.png")


# =============================================================================
# Scene helpers (same as people_demo.py)
# =============================================================================

def freeze_doors(stage):
    for prim in stage.Traverse():
        if prim.HasAPI(UsdPhysics.RigidBodyAPI):
            UsdPhysics.RigidBodyAPI(prim).CreateKinematicEnabledAttr(True)


def add_ground_plane(stage, z):
    UsdGeom.Xform.Define(stage, "/World/GroundPlane")
    mesh = UsdGeom.Mesh.Define(stage, "/World/GroundPlane/Mesh")
    half = 300.0
    mesh.CreatePointsAttr([
        Gf.Vec3f(-half, -half, z), Gf.Vec3f(half, -half, z),
        Gf.Vec3f(half, half, z), Gf.Vec3f(-half, half, z),
    ])
    mesh.CreateFaceVertexCountsAttr([4])
    mesh.CreateFaceVertexIndicesAttr([0, 1, 2, 3])
    mesh.CreateNormalsAttr([Gf.Vec3f(0, 0, 1)] * 4)
    UsdPhysics.CollisionAPI.Apply(mesh.GetPrim())
    coll = UsdPhysics.MeshCollisionAPI.Apply(mesh.GetPrim())
    coll.CreateApproximationAttr("none")


def load_character(stage, prim_path, character_usd, position, face_dir_xy):
    """Load character USD and set initial pose (Z-axis rotation toward face_dir_xy)."""
    print(f"[walking_demo] Loading character: {character_usd}", flush=True)
    try:
        person = prims_utils.create_prim(prim_path, "Xform", usd_path=character_usd)
    except Exception as e:
        print(f"[walking_demo] ERROR loading character: {e}", flush=True)
        return None

    if not person or not person.IsValid():
        return None

    set_character_pose(person, position, face_dir_xy)
    print(f"[walking_demo] Character spawned at {position}", flush=True)
    return person


def set_character_pose(person_prim, position, face_dir_xy):
    """Update character translate + orient to face direction (dx, dy).

    Called every frame to move the character along the path.
    """
    # Translate
    translate_attr = person_prim.GetAttribute("xformOp:translate")
    if translate_attr and translate_attr.IsValid():
        translate_attr.Set(Gf.Vec3d(float(position[0]), float(position[1]), float(position[2])))
    else:
        xformable = UsdGeom.Xformable(person_prim)
        xformable.AddTranslateOp().Set(Gf.Vec3d(float(position[0]), float(position[1]), float(position[2])))

    # Orient (yaw around Z axis toward movement direction)
    dx, dy = float(face_dir_xy[0]), float(face_dir_xy[1])
    if abs(dx) < 1e-6 and abs(dy) < 1e-6:
        return  # no direction to face
    face_angle_deg = math.degrees(math.atan2(dy, dx))
    # Character default forward is +X in its own frame. Rotating around Z by
    # face_angle puts +X into world direction (dx, dy).
    rot_quat = Gf.Rotation(Gf.Vec3d(0, 0, 1), face_angle_deg).GetQuat()

    orient_attr = person_prim.GetAttribute("xformOp:orient")
    if orient_attr and orient_attr.IsValid():
        current = orient_attr.Get()
        if isinstance(current, Gf.Quatf):
            orient_attr.Set(Gf.Quatf(rot_quat))
        else:
            orient_attr.Set(rot_quat)
    else:
        xformable = UsdGeom.Xformable(person_prim)
        xformable.AddOrientOp().Set(Gf.Quatf(rot_quat))


def save_screenshot(camera, output_path):
    rgba = camera.get_rgba()
    if rgba is None:
        return False
    try:
        from PIL import Image
        os.makedirs(os.path.dirname(output_path), exist_ok=True)
        img = Image.fromarray(rgba[:, :, :3])
        img.save(output_path)
        print(f"[walking_demo] Saved: {output_path}", flush=True)
        return True
    except Exception as e:
        print(f"[walking_demo] Screenshot error: {e}", flush=True)
        return False


# =============================================================================
# Path interpolation
# =============================================================================

def build_frames_along_path(waypoints, walk_speed_ft_per_sec, fps):
    """Interpolate evenly-spaced positions + facing directions along a polyline.

    Returns a list of (pos, face_dir_xy) tuples, one per recorded frame.
    """
    # Compute segment distances
    segments = []
    total_dist = 0.0
    for i in range(len(waypoints) - 1):
        p0 = np.array(waypoints[i], dtype=float)
        p1 = np.array(waypoints[i + 1], dtype=float)
        seg_dist = float(np.linalg.norm(p1[:2] - p0[:2]))  # XY distance
        segments.append({"p0": p0, "p1": p1, "dist": seg_dist, "cum": total_dist})
        total_dist += seg_dist

    duration_sec = total_dist / walk_speed_ft_per_sec
    num_frames = max(1, int(duration_sec * fps))
    print(f"[walking_demo] Path: {total_dist:.1f}ft total, {duration_sec:.1f}s, {num_frames} frames", flush=True)

    frames = []
    for f in range(num_frames):
        t = f / max(1, num_frames - 1)  # 0 → 1
        travelled = t * total_dist

        # Find current segment
        seg = segments[-1]
        for s in segments:
            if travelled <= s["cum"] + s["dist"]:
                seg = s
                break

        local_t = (travelled - seg["cum"]) / max(seg["dist"], 1e-6)
        local_t = max(0.0, min(1.0, local_t))
        pos = seg["p0"] + (seg["p1"] - seg["p0"]) * local_t
        direction = (seg["p1"] - seg["p0"])[:2]
        frames.append((pos.tolist(), direction.tolist()))

    return frames


# =============================================================================
# Main
# =============================================================================

def main():
    usd_path = sys.argv[1] if len(sys.argv) > 1 else os.path.join(
        _project_dir, "assets", "environments", "msoe_building",
        "MSOEBuildingV4_physics.usd"
    )
    if not os.path.exists(usd_path):
        print(f"ERROR: USD not found: {usd_path}", flush=True)
        sys.exit(1)

    print(f"[walking_demo] Building USD: {usd_path}", flush=True)

    # --- Resolve character asset path ---
    assets_root = get_assets_root_path()
    if assets_root is None:
        print("[walking_demo] ERROR: get_assets_root_path() returned None", flush=True)
        sys.exit(2)
    character_usd = assets_root + CHARACTER_SUBPATH
    biped_fallback = assets_root + BIPED_FALLBACK_SUBPATH

    # --- Scene setup ---
    world = World(stage_units_in_meters=1.0)
    stage = omni.usd.get_context().get_stage()

    building_prim = stage.DefinePrim("/World/Building", "Xform")
    building_prim.GetReferences().AddReference(
        usd_path, primPath=Sdf.Path("/World/Building")
    )

    setup_physics_scene(stage)
    freeze_doors(stage)

    building_prim = stage.GetPrimAtPath("/World/Building")
    bbox_cache = UsdGeom.BBoxCache(Usd.TimeCode.Default(), [UsdGeom.Tokens.default_])
    bbox = bbox_cache.ComputeWorldBound(building_prim)
    bb_min = bbox.ComputeAlignedRange().GetMin()
    add_ground_plane(stage, bb_min[2] - 0.5)

    add_interior_lighting(stage)
    camera = create_test_camera(CAM_KEY, CAM, stage=stage)

    # --- Build walking path ---
    frames = build_frames_along_path(WAYPOINTS, WALK_SPEED_FT_PER_SEC, FPS)
    start_pos, start_dir = frames[0]

    # --- Load character at start waypoint, facing first segment direction ---
    person = load_character(
        stage, "/World/Person_01", character_usd,
        position=start_pos, face_dir_xy=start_dir,
    )
    if person is None:
        print("[walking_demo] Retrying with Biped_Setup fallback...", flush=True)
        person = load_character(
            stage, "/World/Person_01", biped_fallback,
            position=start_pos, face_dir_xy=start_dir,
        )
    if person is None:
        print("[walking_demo] FATAL: could not load character", flush=True)
        simulation_app.close()
        sys.exit(3)

    # --- Reset + initialize ---
    world.reset()
    camera.initialize()
    set_focal_length_on_prim(stage, camera)

    # --- Warmup ---
    print(f"[walking_demo] Warmup: {WARMUP_FRAMES} frames", flush=True)
    for _ in range(WARMUP_FRAMES):
        world.step(render=True)

    # --- Start screenshot ---
    save_screenshot(camera, SCREENSHOT_START)

    # --- Walk the path while recording ---
    print(f"[walking_demo] Recording {len(frames)} frames to {VIDEO_NAME}.mp4", flush=True)
    with VideoRecorder(VIDEO_NAME, fps=FPS) as recorder:
        for i, (pos, direction) in enumerate(frames):
            set_character_pose(person, pos, direction)
            world.step(render=True)
            try:
                rgba = camera.get_rgba()
                if rgba is not None:
                    recorder.capture_rgba(rgba)
            except Exception:
                pass
            if i % 30 == 0:
                print(f"  Frame {i}/{len(frames)} pos=({pos[0]:.1f}, {pos[1]:.1f})", flush=True)

    # --- End screenshot ---
    save_screenshot(camera, SCREENSHOT_END)

    print("[walking_demo] Done!", flush=True)
    os._exit(0)


if __name__ == "__main__":
    main()
