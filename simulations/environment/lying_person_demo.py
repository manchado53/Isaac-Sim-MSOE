#!/usr/bin/env python3
"""
Lying Person Demo — Single static human character laid horizontally on the
Innovation Center ground floor to simulate an injured/unconscious person
(anomaly class for training data).

This is the BABY STEP of Option B (skeleton/pose manipulation). Instead of
manipulating individual joints via UsdSkel.Animation, we simply rotate the
entire character prim 90° around a horizontal axis so the T-pose character
lies flat on their back. The character remains in T-pose but is horizontal.

Rotation choice:
    We rotate 90° around the +Y axis. Before rotation the character stands
    with their height along +Z. After a +90° rotation around +Y, the
    character's +Z (head) direction maps to +X, while their original +Y
    (front/back) maps to -Z. This results in the character lying with
    their BACK facing +Z (supine, face up) — which is what we want for
    an "injured person on the floor" anomaly.

    (X-axis rotation also works but would put the character on their side
    or face down depending on sign. +Y/+90° is the cleanest "on their back"
    pose.)

Z placement after rotation:
    A standing biped origin sits at the feet (Z=0 local). After a 90° Y
    rotation, what used to be the feet-to-head axis now extends along +X
    in world space. The character's torso "thickness" (originally ~1 ft
    in X) is now the vertical extent. To put the back flush against the
    floor we need to lift the prim by roughly half the body thickness,
    which for a typical NVIDIA biped is approximately 0.5 - 1.0 feet.
    This is tunable via LYING_Z_OFFSET.

Output:
    images/lying_person_demo.png
    videos/environment/lying_person_demo.mp4
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

# Place the lying person near the camera (closer than standing demo so the
# lying body is clearly visible). Camera G pos=(-42.15, 68.44); target=
# (-27.42, 83.72). 35% along that line gives ~(-37, 74) — closer to camera.
PERSON_X = -37.0
PERSON_Y = 74.0

# Rotation axis for lying-down pose. We rotate the entire character
# 90° around the world X axis so the character's spine (originally +Z) tilts
# to -Y, laying them on their back with head pointing -Y, feet +Y, chest up.
# (Y axis rotation turned out to just swap width/height in the bbox due to
# T-pose symmetry — X rotation actually puts them flat on the floor.)
LYING_ROT_AXIS = Gf.Vec3d(1, 0, 0)
LYING_ROT_DEG = 90.0

# Vertical offset above the floor for the character's origin AFTER rotation.
# After a 90° X rotation, the character's spine (originally +Z) points along
# -Y; their body-depth axis (originally +Y front-of-body) now points along +Z.
# So chest/face end up on top, back faces down. We lift the rotated origin
# by roughly half the body thickness (~0.7 ft after meters→feet scaling) so
# the back rests just above the floor.
# Tune this if the character clips into the floor or floats noticeably.
LYING_Z_OFFSET = 0.85
PERSON_Z = FLOOR_Z + LYING_Z_OFFSET

# Body-orientation hint: we want the character's head to point roughly toward
# the camera so the face is visible. Because the full 3-axis pose is produced
# by a single Y rotation, we add an additional Z-axis "yaw" about the world
# up axis to spin the lying body around so the head points where we want.
# face_yaw_deg = angle (degrees) around +Z to rotate the lying body.
# 0 => head points along +X (toward east); 180 => head points along -X.
FACE_YAW_DEG = 180.0  # head points toward -X (roughly toward camera at -42, 68)

# Scale — character USD is in meters, building is in feet. The metrics
# assembler does NOT auto-convert in this case (verified via bbox diagnostic:
# character was 1.76 ft tall instead of 5.6 ft). Scale up by 1/0.3048 ≈ 3.28
# to bring the character to its natural ~5.6 ft height in feet units.
PERSON_SCALE = 1.0 / 0.3048

# Character asset
CHARACTER_SUBPATH = "/Isaac/People/Characters/F_Business_02/F_Business_02.usd"
BIPED_FALLBACK_SUBPATH = "/Isaac/People/Characters/Biped_Setup.usd"

# Output
RECORD_FRAMES = 150  # 5s at 30fps
WARMUP_FRAMES = 30
SCREENSHOT_PATH = os.path.join(_project_dir, "images", "lying_person_demo.png")
VIDEO_NAME = "environment/lying_person_demo"


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


def load_lying_character(stage, prim_path, character_usd, position,
                          rot_axis, rot_deg, face_yaw_deg, scale=1.0):
    """Load a character USD and lay them horizontally on their back.

    The combined rotation is: (yaw around +Z) * (tilt around rot_axis).
    This first tilts the character from standing (T-pose) to horizontal,
    then spins the horizontal body around the world Z axis so the head
    points in the desired direction.

    Args:
        stage: USD stage
        prim_path: Where to place the character in the stage hierarchy
        character_usd: URL or path to character USD
        position: (x, y, z) in stage units (feet)
        rot_axis: Gf.Vec3d rotation axis for the tilt (expect horizontal axis)
        rot_deg: rotation angle in degrees (90 lies them down)
        face_yaw_deg: additional Z-axis yaw in degrees to orient head direction
        scale: uniform scale factor

    Returns:
        Character prim, or None on failure.
    """
    print(f"[lying_person_demo] Loading character: {character_usd}", flush=True)
    try:
        person = prims_utils.create_prim(
            prim_path,
            "Xform",
            usd_path=character_usd,
        )
    except Exception as e:
        print(f"[lying_person_demo] ERROR loading character: {e}", flush=True)
        return None

    if not person or not person.IsValid():
        print(f"[lying_person_demo] ERROR: prim invalid at {prim_path}", flush=True)
        return None

    # --- Parent root: translate only ---
    # The NVIDIA character USD composition honors xformOp:translate on the
    # root prim we create, but IGNORES the root's rotation/scale for rendering
    # (likely because of metrics-assembler unit conversion overriding them).
    # So we put translation on the parent, and rotation+scale on the CHILD
    # xform (character body), which actually takes effect.
    xformable = UsdGeom.Xformable(person)
    xformable.ClearXformOpOrder()
    translate_op = xformable.AddTranslateOp(
        precision=UsdGeom.XformOp.PrecisionDouble
    )
    translate_op.Set(Gf.Vec3d(float(position[0]), float(position[1]),
                               float(position[2])))

    # Tilt mapping from the rot_axis vector to the rotateXYZ Euler triple.
    # The child uses rotateXYZ order, so pick the component matching our axis.
    if float(rot_axis[0]) != 0.0:
        tilt_euler = Gf.Vec3f(float(rot_deg), 0.0, 0.0)
    elif float(rot_axis[1]) != 0.0:
        tilt_euler = Gf.Vec3f(0.0, float(rot_deg), 0.0)
    else:
        tilt_euler = Gf.Vec3f(0.0, 0.0, float(rot_deg))

    # Compose yaw_Z into the Euler: since rotateXYZ order means the yaw is
    # the LAST component applied (Z), we can stuff face_yaw_deg into the Z
    # component when the tilt axis is X or Y.
    if float(rot_axis[2]) == 0.0:
        tilt_euler = Gf.Vec3f(tilt_euler[0], tilt_euler[1],
                               float(face_yaw_deg))

    print(f"[lying_person_demo] Post-set xformOpOrder on {prim_path}: "
          f"{list(xformable.GetXformOpOrderAttr().Get() or [])}", flush=True)

    # Apply tilt + yaw + scale to the character's immediate child xform.
    for child in person.GetChildren():
        if not child.IsA(UsdGeom.Xformable):
            continue
        child_xformable = UsdGeom.Xformable(child)
        child_ops = list(child_xformable.GetXformOpOrderAttr().Get() or [])
        print(f"[lying_person_demo]   child '{child.GetName()}' "
              f"({child.GetTypeName()}) xformOps={child_ops}", flush=True)

        rot_attr = child.GetAttribute("xformOp:rotateXYZ")
        if rot_attr and rot_attr.IsValid():
            print(f"[lying_person_demo]   child rotateXYZ was {rot_attr.Get()}, "
                  f"setting to {tilt_euler}", flush=True)
            rot_attr.Set(tilt_euler)

        s_attr = child.GetAttribute("xformOp:scale")
        if s_attr and s_attr.IsValid():
            current_scale = s_attr.Get()
            new_scale = type(current_scale)(scale, scale, scale)
            print(f"[lying_person_demo]   child scale was {current_scale}, "
                  f"setting to {new_scale}", flush=True)
            s_attr.Set(new_scale)

        t_attr = child.GetAttribute("xformOp:translate")
        if t_attr and t_attr.IsValid():
            print(f"[lying_person_demo]   child translate={t_attr.Get()}",
                  flush=True)

    print(f"[lying_person_demo] Placed lying character at "
          f"({position[0]:.2f}, {position[1]:.2f}, {position[2]:.2f}), "
          f"tilt {rot_deg}° around {tuple(rot_axis)}, "
          f"yaw {face_yaw_deg}°, scale={scale}", flush=True)
    return person


def save_screenshot(camera, output_path):
    """Save the current camera frame as a PNG."""
    rgba = camera.get_rgba()
    if rgba is None:
        print(f"[lying_person_demo] WARNING: camera returned no rgba", flush=True)
        return False
    try:
        from PIL import Image
        os.makedirs(os.path.dirname(output_path), exist_ok=True)
        img = Image.fromarray(rgba[:, :, :3])
        img.save(output_path)
        mean_val = float(np.mean(rgba[:, :, :3]))
        print(f"[lying_person_demo] Saved screenshot: {output_path} "
              f"(mean brightness={mean_val:.1f})", flush=True)
        return True
    except ImportError:
        np.save(output_path.replace(".png", ".npy"), rgba)
        print(f"[lying_person_demo] Saved rgba.npy (PIL not available)", flush=True)
        return True
    except Exception as e:
        print(f"[lying_person_demo] ERROR saving screenshot: {e}", flush=True)
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

    print(f"[lying_person_demo] Building USD: {usd_path}", flush=True)

    # --- Resolve character asset path ---
    assets_root = get_assets_root_path()
    if assets_root is None:
        print("[lying_person_demo] ERROR: get_assets_root_path() returned None. "
              "Cannot load character.", flush=True)
        sys.exit(2)

    character_usd = assets_root + CHARACTER_SUBPATH
    biped_fallback = assets_root + BIPED_FALLBACK_SUBPATH
    print(f"[lying_person_demo] Assets root: {assets_root}", flush=True)

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

    # --- Load lying character ---
    person_pos = (PERSON_X, PERSON_Y, PERSON_Z)
    person = load_lying_character(
        stage, "/World/Person_01", character_usd,
        position=person_pos,
        rot_axis=LYING_ROT_AXIS, rot_deg=LYING_ROT_DEG,
        face_yaw_deg=FACE_YAW_DEG,
        scale=PERSON_SCALE,
    )
    if person is None:
        print(f"[lying_person_demo] Retrying with Biped_Setup fallback...", flush=True)
        person = load_lying_character(
            stage, "/World/Person_01", biped_fallback,
            position=person_pos,
            rot_axis=LYING_ROT_AXIS, rot_deg=LYING_ROT_DEG,
            face_yaw_deg=FACE_YAW_DEG,
            scale=PERSON_SCALE,
        )
    if person is None:
        print(f"[lying_person_demo] FATAL: could not load any character asset", flush=True)
        simulation_app.close()
        sys.exit(3)

    # --- Reset + initialize ---
    world.reset()
    camera.initialize()
    set_focal_length_on_prim(stage, camera)

    # --- Warmup ---
    print(f"[lying_person_demo] Warmup: {WARMUP_FRAMES} frames", flush=True)
    for _ in range(WARMUP_FRAMES):
        world.step(render=True)

    # --- Diagnostic: world-space bounding box after warmup ---
    try:
        bbox_cache_p = UsdGeom.BBoxCache(Usd.TimeCode.Default(),
                                         [UsdGeom.Tokens.default_])
        pbbox = bbox_cache_p.ComputeWorldBound(person)
        pmin = pbbox.ComputeAlignedRange().GetMin()
        pmax = pbbox.ComputeAlignedRange().GetMax()
        print(f"[lying_person_demo] Character world bbox: "
              f"min=({pmin[0]:.2f},{pmin[1]:.2f},{pmin[2]:.2f}) "
              f"max=({pmax[0]:.2f},{pmax[1]:.2f},{pmax[2]:.2f}) "
              f"size=({pmax[0]-pmin[0]:.2f},{pmax[1]-pmin[1]:.2f},{pmax[2]-pmin[2]:.2f})",
              flush=True)
    except Exception as e:
        print(f"[lying_person_demo] WARN: bbox diag failed: {e}", flush=True)

    # --- Screenshot ---
    save_screenshot(camera, SCREENSHOT_PATH)

    # --- Record video ---
    print(f"[lying_person_demo] Recording {RECORD_FRAMES} frames to {VIDEO_NAME}.mp4",
          flush=True)
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

    print("[lying_person_demo] Done!", flush=True)
    os._exit(0)


if __name__ == "__main__":
    main()
