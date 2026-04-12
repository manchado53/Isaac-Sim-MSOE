#!/usr/bin/env python3
"""
People Crowd Demo — Six static human characters scattered across the
Innovation Center Event Floor (IC 100), viewed from the "G" security
camera.

Builds on people_demo.py:
  * Uses the measured CHARACTER_Z_OFFSET (see diagnose_character_bbox.py)
    to plant the characters' feet on the floor instead of floating.
  * Mixes six different NVIDIA character USDs (business/medical/
    construction/police, male + female). Falls back to Biped_Setup if a
    specific asset fails to resolve.
  * Spawns characters in natural-looking positions across X=[-38, -25],
    Y=[70, 85], each facing a different direction.

Output:
    images/people_crowd_demo.png
    videos/environment/people_crowd_demo.mp4
"""

import os
import sys
import math
import random

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
FLOOR_Z = 14.56

# Character Z offset (feet) — see people_demo.py / diagnose_character_bbox.py.
CHARACTER_Z_OFFSET = -0.875
PERSON_Z = FLOOR_Z + CHARACTER_Z_OFFSET

PERSON_SCALE = 1.0

# Character asset catalog. Tried in order; Biped_Setup is the universal fallback.
BIPED_FALLBACK_SUBPATH = "/Isaac/People/Characters/Biped_Setup.usd"
CHARACTERS = [
    ("F_Business_02",    (-36.0, 76.0), "/Isaac/People/Characters/F_Business_02/F_Business_02.usd"),
    ("F_Medical_01",     (-32.0, 80.0), "/Isaac/People/Characters/F_Medical_01/F_Medical_01.usd"),
    ("M_Business_02",    (-28.0, 74.0), "/Isaac/People/Characters/M_Business_02/M_Business_02.usd"),
    ("M_Construction_05",(-37.5, 82.0), "/Isaac/People/Characters/M_Construction_05/M_Construction_05.usd"),
    ("F_Police_01",      (-30.5, 71.0), "/Isaac/People/Characters/F_Police_01/F_Police_01.usd"),
    ("M_Police_04",      (-26.0, 82.5), "/Isaac/People/Characters/M_Police_04/M_Police_04.usd"),
]

# Output
RECORD_FRAMES = 150
WARMUP_FRAMES = 30
SCREENSHOT_PATH = os.path.join(_project_dir, "images", "people_crowd_demo.png")
VIDEO_NAME = "environment/people_crowd_demo"


def freeze_doors(stage):
    """Freeze all dynamic bodies so the scene is static."""
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


def load_character(stage, prim_path, character_usd, position, face_angle_deg, scale=1.0):
    """Load a character USD as a static reference prim.

    Args:
        position: (x, y, z) in stage units (feet for Revit buildings)
        face_angle_deg: rotation about Z axis (degrees)
    """
    print(f"[crowd] Loading {prim_path}: {character_usd}", flush=True)
    try:
        person = prims_utils.create_prim(
            prim_path,
            "Xform",
            usd_path=character_usd,
        )
    except Exception as e:
        print(f"[crowd] ERROR loading character: {e}", flush=True)
        return None

    if not person or not person.IsValid():
        print(f"[crowd] ERROR: character prim invalid at {prim_path}", flush=True)
        return None

    translate_attr = person.GetAttribute("xformOp:translate")
    if translate_attr and translate_attr.IsValid():
        translate_attr.Set(Gf.Vec3d(float(position[0]), float(position[1]), float(position[2])))
    else:
        UsdGeom.Xformable(person).AddTranslateOp().Set(
            Gf.Vec3d(float(position[0]), float(position[1]), float(position[2]))
        )

    rot_quat = Gf.Rotation(Gf.Vec3d(0, 0, 1), float(face_angle_deg)).GetQuat()
    orient_attr = person.GetAttribute("xformOp:orient")
    if orient_attr and orient_attr.IsValid():
        current = orient_attr.Get()
        if isinstance(current, Gf.Quatf):
            orient_attr.Set(Gf.Quatf(rot_quat))
        else:
            orient_attr.Set(rot_quat)
    else:
        UsdGeom.Xformable(person).AddOrientOp().Set(Gf.Quatf(rot_quat))

    if abs(scale - 1.0) > 1e-6:
        scale_attr = person.GetAttribute("xformOp:scale")
        if scale_attr and scale_attr.IsValid():
            scale_attr.Set(Gf.Vec3d(scale, scale, scale))
        else:
            UsdGeom.Xformable(person).AddScaleOp().Set(Gf.Vec3d(scale, scale, scale))

    return person


def save_screenshot(camera, output_path):
    rgba = camera.get_rgba()
    if rgba is None:
        print(f"[crowd] WARNING: camera returned no rgba data", flush=True)
        return False
    try:
        from PIL import Image
        os.makedirs(os.path.dirname(output_path), exist_ok=True)
        img = Image.fromarray(rgba[:, :, :3])
        img.save(output_path)
        mean_val = float(np.mean(rgba[:, :, :3]))
        print(f"[crowd] Saved screenshot: {output_path} (mean brightness={mean_val:.1f})", flush=True)
        return True
    except Exception as e:
        print(f"[crowd] ERROR saving screenshot: {e}", flush=True)
        return False


def main():
    usd_path = sys.argv[1] if len(sys.argv) > 1 else os.path.join(
        _project_dir, "assets", "environments", "msoe_building",
        "MSOEBuildingV4_physics.usd"
    )
    if not os.path.exists(usd_path):
        print(f"ERROR: USD not found: {usd_path}", flush=True)
        sys.exit(1)

    print(f"[crowd] Building USD: {usd_path}", flush=True)

    assets_root = get_assets_root_path()
    if assets_root is None:
        print("[crowd] ERROR: get_assets_root_path() returned None", flush=True)
        sys.exit(2)
    biped_fallback = assets_root + BIPED_FALLBACK_SUBPATH
    print(f"[crowd] Assets root: {assets_root}", flush=True)

    # --- Scene setup ---
    world = World(stage_units_in_meters=1.0)
    stage = omni.usd.get_context().get_stage()

    building_prim = stage.DefinePrim("/World/Building", "Xform")
    building_prim.GetReferences().AddReference(
        usd_path, primPath=Sdf.Path("/World/Building")
    )

    setup_physics_scene(stage)
    freeze_doors(stage)

    # Ground plane below the building min
    building_prim = stage.GetPrimAtPath("/World/Building")
    bbox_cache = UsdGeom.BBoxCache(Usd.TimeCode.Default(), [UsdGeom.Tokens.default_])
    bbox = bbox_cache.ComputeWorldBound(building_prim)
    bb_min = bbox.ComputeAlignedRange().GetMin()
    add_ground_plane(stage, bb_min[2] - 0.5)

    add_interior_lighting(stage)
    camera = create_test_camera(CAM_KEY, CAM, stage=stage)

    # --- Spawn characters ---
    # Seed for reproducible facing angles
    rng = random.Random(42)
    loaded_info = []
    fallback_count = 0
    for idx, (name, (x, y), subpath) in enumerate(CHARACTERS):
        asset_url = assets_root + subpath
        prim_path = f"/World/Person_{idx:02d}"

        # Random-ish facing angle; bias toward facing the camera so most faces visible
        # Compute angle from person -> camera, then add a jitter up to +/-90 deg.
        dx = CAM["pos"][0] - x
        dy = CAM["pos"][1] - y
        base_angle = math.degrees(math.atan2(dy, dx))
        face_angle = base_angle + rng.uniform(-90.0, 90.0)

        pos = (x, y, PERSON_Z)
        person = load_character(stage, prim_path, asset_url, pos, face_angle, PERSON_SCALE)
        used_fallback = False
        if person is None:
            print(f"[crowd] {name} failed — falling back to Biped_Setup", flush=True)
            person = load_character(stage, prim_path, biped_fallback, pos, face_angle, PERSON_SCALE)
            used_fallback = True
            fallback_count += 1
        if person is None:
            print(f"[crowd] FATAL: even fallback failed for {name}", flush=True)
            continue
        loaded_info.append({
            "name": name, "prim": prim_path, "pos": pos,
            "angle": face_angle, "fallback": used_fallback,
        })
        print(f"[crowd] Placed {name} at ({x:.1f}, {y:.1f}, {PERSON_Z:.2f}), "
              f"facing {face_angle:.1f}°, fallback={used_fallback}", flush=True)

    print(f"[crowd] Loaded {len(loaded_info)}/{len(CHARACTERS)} characters "
          f"({fallback_count} fallback)", flush=True)

    # --- Reset + initialize ---
    world.reset()
    camera.initialize()
    set_focal_length_on_prim(stage, camera)

    # --- Warmup ---
    print(f"[crowd] Warmup: {WARMUP_FRAMES} frames", flush=True)
    for _ in range(WARMUP_FRAMES):
        world.step(render=True)

    # --- Screenshot ---
    save_screenshot(camera, SCREENSHOT_PATH)

    # --- Record video ---
    print(f"[crowd] Recording {RECORD_FRAMES} frames to {VIDEO_NAME}.mp4", flush=True)
    with VideoRecorder(VIDEO_NAME, fps=30) as recorder:
        for i in range(RECORD_FRAMES):
            world.step(render=True)
            try:
                rgba = camera.get_rgba()
                if rgba is not None:
                    recorder.capture_rgba(rgba)
            except Exception:
                pass
            if i % 30 == 0:
                print(f"  Frame {i}/{RECORD_FRAMES}", flush=True)

    print("[crowd] Done!", flush=True)
    os._exit(0)


if __name__ == "__main__":
    main()
