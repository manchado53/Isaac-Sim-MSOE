#!/usr/bin/env python3
"""
People Demo — Single static human character on the Innovation Center ground floor.

Baby step #1 of people integration: place one standing person in the Event
Floor (IC 100), visible from the "G" security camera, capture a screenshot
and short video to verify the character asset pipeline works.

Uses Option A (static character USD reference) — no animation pipeline,
no NavMesh, no omni.anim.people extension. Character is loaded the same
way we load robot USDs.

Output:
    images/people_demo_ground_floor.png
    videos/environment/people_demo.mp4
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
FLOOR_Z = 14.56  # ground floor height constant (feet) from test_cameras target Z

# Scale — BOTH the building (V4_physics) and the character USDs have
# metersPerUnit=1.0. The character is 1.73 stage-units tall at scale=1.0,
# which is 1.73 m = 5.67 ft (correct adult height). NO conversion needed.
# (Earlier commits applied 3.28× based on wrong assumption that building
# was in feet — that made characters 18.6 ft tall. Don't re-introduce it.)
PERSON_SCALE = 1.0

# Character Z offset (stage units = meters). Real walkable floor via raycast
# is at Z=13.56. Character rig origin is at its feet but the bbox-min sits
# ~0.125 below the origin (measured). To plant feet on floor with translate:
#   translate_Z = floor_raycast + 0.125 = 13.685
# relative to FLOOR_Z=14.56 (camera target, above floor by 1 m): offset = -0.875
CHARACTER_Z_OFFSET = -0.875

# Person placement: 55% of the way from camera position to camera target.
# Camera pos = (-42.15, 68.44, 16.44); target = (-27.42, 83.72, 14.56)
# Person XY ends up near (-34.0, 76.8).
T = 0.55
PERSON_X = CAM["pos"][0] + (CAM["target"][0] - CAM["pos"][0]) * T
PERSON_Y = CAM["pos"][1] + (CAM["target"][1] - CAM["pos"][1]) * T
PERSON_Z = FLOOR_Z + CHARACTER_Z_OFFSET

# Character asset path. We try a specific named character first, fall back
# to the default Biped_Setup.usd if that fails to resolve.
CHARACTER_SUBPATH = "/Isaac/People/Characters/F_Business_02/F_Business_02.usd"
BIPED_FALLBACK_SUBPATH = "/Isaac/People/Characters/Biped_Setup.usd"

# Output
RECORD_FRAMES = 150  # 5s at 30fps
WARMUP_FRAMES = 30
SCREENSHOT_PATH = os.path.join(_project_dir, "images", "people_demo_ground_floor.png")
VIDEO_NAME = "environment/people_demo"


def freeze_doors(stage):
    """Freeze all dynamic bodies so the scene is static."""
    for prim in stage.Traverse():
        if prim.HasAPI(UsdPhysics.RigidBodyAPI):
            UsdPhysics.RigidBodyAPI(prim).CreateKinematicEnabledAttr(True)


def add_ground_plane(stage, z):
    """Add a large static ground plane at Z=z (in feet)."""
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


def load_character(stage, prim_path, character_usd, position, face_target, scale=1.0):
    """Load a character USD as a static reference prim.

    The NVIDIA character USD composition honors xformOp:translate on the
    root prim but IGNORES the root's rotation/scale for rendering (metrics
    assembler override). So we put translation on the ROOT, and rotation+
    scale on the CHILD xform (the character body), which actually takes
    effect. Same pattern as lying_person_demo.py.

    Args:
        stage: USD stage
        prim_path: Where to place the character in the stage hierarchy
        character_usd: URL or path to character USD
        position: (x, y, z) in stage units (feet for Revit buildings)
        face_target: (x, y, z) point the character should face (Z axis rotation)
        scale: Uniform scale factor applied to the character (CHILD prim)

    Returns:
        The character prim, or None if loading failed.
    """
    print(f"[people_demo] Loading character: {character_usd}", flush=True)
    try:
        person = prims_utils.create_prim(
            prim_path,
            "Xform",
            usd_path=character_usd,
        )
    except Exception as e:
        print(f"[people_demo] ERROR loading character: {e}", flush=True)
        return None

    if not person or not person.IsValid():
        print(f"[people_demo] ERROR: character prim invalid at {prim_path}", flush=True)
        return None

    # Compute facing yaw (degrees around +Z)
    dx = float(face_target[0]) - float(position[0])
    dy = float(face_target[1]) - float(position[1])
    face_angle_deg = math.degrees(math.atan2(dy, dx))

    # --- Root prim: translate only (rotation/scale ignored by composition) ---
    xformable = UsdGeom.Xformable(person)
    xformable.ClearXformOpOrder()
    translate_op = xformable.AddTranslateOp(
        precision=UsdGeom.XformOp.PrecisionDouble
    )
    translate_op.Set(Gf.Vec3d(float(position[0]), float(position[1]),
                               float(position[2])))

    # --- Child prim: rotation (yaw) + scale ---
    for child in person.GetChildren():
        if not child.IsA(UsdGeom.Xformable):
            continue

        # Yaw around Z: face the target
        rot_attr = child.GetAttribute("xformOp:rotateXYZ")
        if rot_attr and rot_attr.IsValid():
            rot_attr.Set(Gf.Vec3f(0.0, 0.0, float(face_angle_deg)))

        # Uniform scale to convert meters → feet (building is feet-scale)
        s_attr = child.GetAttribute("xformOp:scale")
        if s_attr and s_attr.IsValid():
            current_scale = s_attr.Get()
            new_scale = type(current_scale)(scale, scale, scale)
            s_attr.Set(new_scale)

    print(f"[people_demo] Character placed at ({position[0]:.2f}, {position[1]:.2f}, {position[2]:.2f}), "
          f"facing {face_angle_deg:.1f}°, scale={scale}", flush=True)
    return person


def save_screenshot(camera, output_path):
    """Save the current camera frame as a PNG."""
    rgba = camera.get_rgba()
    if rgba is None:
        print(f"[people_demo] WARNING: camera returned no rgba data", flush=True)
        return False
    try:
        from PIL import Image
        os.makedirs(os.path.dirname(output_path), exist_ok=True)
        img = Image.fromarray(rgba[:, :, :3])
        img.save(output_path)
        mean_val = float(np.mean(rgba[:, :, :3]))
        print(f"[people_demo] Saved screenshot: {output_path} (mean brightness={mean_val:.1f})", flush=True)
        return True
    except ImportError:
        np.save(output_path.replace(".png", ".npy"), rgba)
        print(f"[people_demo] Saved rgba.npy (PIL not available)", flush=True)
        return True
    except Exception as e:
        print(f"[people_demo] ERROR saving screenshot: {e}", flush=True)
        return False


def main():
    # --- Args ---
    usd_path = sys.argv[1] if len(sys.argv) > 1 else os.path.join(
        _project_dir, "assets", "environments", "msoe_building",
        "MSOEBuildingV4_physics.usd"
    )
    if not os.path.exists(usd_path):
        print(f"ERROR: USD not found: {usd_path}", flush=True)
        sys.exit(1)

    print(f"[people_demo] Building USD: {usd_path}", flush=True)

    # --- Resolve character asset path ---
    assets_root = get_assets_root_path()
    if assets_root is None:
        print("[people_demo] ERROR: get_assets_root_path() returned None — "
              "no internet or missing Nucleus. Cannot load character.", flush=True)
        sys.exit(2)

    character_usd = assets_root + CHARACTER_SUBPATH
    biped_fallback = assets_root + BIPED_FALLBACK_SUBPATH
    print(f"[people_demo] Assets root: {assets_root}", flush=True)

    # --- Scene setup ---
    world = World(stage_units_in_meters=1.0)
    stage = omni.usd.get_context().get_stage()

    building_prim = stage.DefinePrim("/World/Building", "Xform")
    building_prim.GetReferences().AddReference(
        usd_path, primPath=Sdf.Path("/World/Building")
    )

    setup_physics_scene(stage)
    freeze_doors(stage)

    # Ground plane slightly below the building min
    building_prim = stage.GetPrimAtPath("/World/Building")
    bbox_cache = UsdGeom.BBoxCache(Usd.TimeCode.Default(), [UsdGeom.Tokens.default_])
    bbox = bbox_cache.ComputeWorldBound(building_prim)
    bb_min = bbox.ComputeAlignedRange().GetMin()
    add_ground_plane(stage, bb_min[2] - 0.5)

    # --- Lighting ---
    add_interior_lighting(stage)

    # --- Camera ---
    camera = create_test_camera(CAM_KEY, CAM, stage=stage)

    # --- Load character ---
    person_pos = (PERSON_X, PERSON_Y, PERSON_Z)
    person = load_character(
        stage, "/World/Person_01", character_usd,
        position=person_pos, face_target=CAM["pos"], scale=PERSON_SCALE,
    )
    if person is None:
        print(f"[people_demo] Retrying with Biped_Setup fallback...", flush=True)
        person = load_character(
            stage, "/World/Person_01", biped_fallback,
            position=person_pos, face_target=CAM["pos"], scale=PERSON_SCALE,
        )
    if person is None:
        print(f"[people_demo] FATAL: could not load any character asset", flush=True)
        simulation_app.close()
        sys.exit(3)

    # --- Reset + initialize ---
    world.reset()
    camera.initialize()
    set_focal_length_on_prim(stage, camera)

    # --- Warmup ---
    print(f"[people_demo] Warmup: {WARMUP_FRAMES} frames", flush=True)
    for _ in range(WARMUP_FRAMES):
        world.step(render=True)

    # --- Screenshot ---
    save_screenshot(camera, SCREENSHOT_PATH)

    # --- Record video ---
    print(f"[people_demo] Recording {RECORD_FRAMES} frames to {VIDEO_NAME}.mp4", flush=True)
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

    print("[people_demo] Done!", flush=True)
    os._exit(0)


if __name__ == "__main__":
    main()
