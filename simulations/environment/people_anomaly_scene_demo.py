#!/usr/bin/env python3
"""
People Anomaly Scene — the combined training scene.

Places multiple standing people + one injured person lying on the floor
in the Event Floor IC 100. This is the real anomaly detection training
scene: "which frames contain a fallen/injured person?"

Uses the corrected patterns from:
  - people_demo.py (child-prim scale=3.28, Z offset for feet-on-floor)
  - lying_person_natural_demo.py (UsdSkel joint manipulation for arms-at-sides)

Output:
    images/people_anomaly_scene.png
    videos/environment/people_anomaly_scene.mp4
"""

import os
import sys
import math

os.environ["OMNI_KIT_ACCEPT_EULA"] = "YES"

from isaacsim import SimulationApp
simulation_app = SimulationApp({"headless": True})

import numpy as np
import omni.usd
from pxr import Usd, UsdGeom, UsdPhysics, UsdSkel, Sdf, Gf, Vt

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
FLOOR_Z = 14.56   # camera target Z (for reference)
REAL_FLOOR_Z = 13.56  # actual walkable surface (from raycast)

# Scale=1.0: stage + character both metersPerUnit=1.0, no conversion.
# See CLAUDE.md "CRITICAL: metersPerUnit varies per USD" for why.
PERSON_SCALE = 1.0

# Standing: translate_Z = 13.685 plants feet on real floor (13.56 + 0.125).
# Relative to FLOOR_Z=14.56, offset = -0.875.
STANDING_Z = FLOOR_Z - 0.875  # = 13.685

# Lying: translate_Z = 13.83 puts back on real floor at 13.56
# (measured offset -0.73 from lying_person_natural_demo.py)
LYING_Z = FLOOR_Z - 0.73
LYING_ROT_DEG = 90.0       # 90° tilt around X axis (supine)
LYING_FACE_YAW = 180.0

# Shoulder overrides for the lying character (UsdSkel)
L_UPPERARM_IDX = 50
R_UPPERARM_IDX = 78
L_SHOULDER_ANGLE_DEG = -90.0
R_SHOULDER_ANGLE_DEG = 90.0
SHOULDER_AXIS_LOCAL = (0.0, 0.0, 1.0)

# Scene layout (coords in feet, XY in the Event Floor)
#
# Camera G looks from (-42.15, 68.44) toward (-27.42, 83.72).
# Arrange characters along and across this line so they're all in FOV.
#
# Lying person near the far end (closer to camera target).
# Standing people scattered at varied distances and facing directions.
LYING_POS = (-32.0, 78.5)    # roughly center of Event Floor

STANDING_PEOPLE = [
    # (xy, face_target_xy, character_subpath, label)
    ((-29.0, 72.0), (-42.15, 68.44), "/Isaac/People/Characters/F_Medical_01/F_Medical_01.usd",    "medical"),
    ((-35.0, 70.0), (-27.42, 83.72), "/Isaac/People/Characters/M_Business_02/M_Business_02.usd", "business"),
    ((-37.0, 82.0), (-29.0, 72.0),   "/Isaac/People/Characters/M_Police_04/M_Police_04.usd",     "police"),
]

LYING_CHARACTER_SUBPATH = "/Isaac/People/Characters/F_Business_02/F_Business_02.usd"
BIPED_FALLBACK_SUBPATH = "/Isaac/People/Characters/Biped_Setup.usd"

# Output
RECORD_FRAMES = 90  # 3s at 30fps — static scene
WARMUP_FRAMES = 30
SCREENSHOT_PATH = os.path.join(_project_dir, "images", "people_anomaly_scene.png")
VIDEO_NAME = "environment/people_anomaly_scene"


# -----------------------------------------------------------------------------
# Scene helpers
# -----------------------------------------------------------------------------

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


def load_standing_character(stage, prim_path, character_usd, position, face_target, scale):
    """Load a standing character (root translate, child rotateXYZ + scale)."""
    try:
        person = prims_utils.create_prim(prim_path, "Xform", usd_path=character_usd)
    except Exception as e:
        print(f"[anomaly] ERROR loading {prim_path}: {e}", flush=True)
        return None
    if not person or not person.IsValid():
        return None

    dx = float(face_target[0]) - float(position[0])
    dy = float(face_target[1]) - float(position[1])
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
            s.Set(type(cur)(scale, scale, scale))

    print(f"[anomaly] Standing {prim_path} at ({position[0]:.1f},{position[1]:.1f},{position[2]:.2f}) "
          f"facing {face_angle_deg:.0f}°", flush=True)
    return person


def decompose_mat4(m):
    t = m.ExtractTranslation()
    r = m.ExtractRotation().GetQuat()
    col0 = Gf.Vec3d(m[0][0], m[0][1], m[0][2])
    col1 = Gf.Vec3d(m[1][0], m[1][1], m[1][2])
    col2 = Gf.Vec3d(m[2][0], m[2][1], m[2][2])
    return t, r, Gf.Vec3f(float(col0.GetLength()),
                           float(col1.GetLength()),
                           float(col2.GetLength()))


def load_lying_character(stage, prim_path, character_usd, position,
                          rot_deg, face_yaw_deg, scale):
    """Load character, body-tilt supine, and pose arms at sides via UsdSkel."""
    try:
        person = prims_utils.create_prim(prim_path, "Xform", usd_path=character_usd)
    except Exception as e:
        print(f"[anomaly] ERROR loading {prim_path}: {e}", flush=True)
        return None
    if not person or not person.IsValid():
        return None

    xformable = UsdGeom.Xformable(person)
    xformable.ClearXformOpOrder()
    xformable.AddTranslateOp(precision=UsdGeom.XformOp.PrecisionDouble).Set(
        Gf.Vec3d(float(position[0]), float(position[1]), float(position[2]))
    )

    # Tilt 90° around X + yaw around Z to orient head
    for child in person.GetChildren():
        if not child.IsA(UsdGeom.Xformable):
            continue
        r = child.GetAttribute("xformOp:rotateXYZ")
        if r and r.IsValid():
            r.Set(Gf.Vec3f(float(rot_deg), 0.0, float(face_yaw_deg)))
        s = child.GetAttribute("xformOp:scale")
        if s and s.IsValid():
            cur = s.Get()
            s.Set(type(cur)(scale, scale, scale))

    # Find skeleton and bind a UsdSkel.Animation with shoulder overrides
    skel_prim = None
    for p in Usd.PrimRange(person):
        if p.GetTypeName() == "Skeleton":
            skel_prim = p
            break
    if skel_prim is None:
        print(f"[anomaly] WARN: no Skeleton in {prim_path}; falling back to T-pose", flush=True)
        return person

    sk = UsdSkel.Skeleton(skel_prim)
    joints = list(sk.GetJointsAttr().Get() or [])
    rest_xforms = list(sk.GetRestTransformsAttr().Get() or [])
    if not joints or len(rest_xforms) != len(joints):
        print(f"[anomaly] WARN: skeleton data malformed; skipping pose", flush=True)
        return person

    overrides = {
        L_UPPERARM_IDX: (SHOULDER_AXIS_LOCAL, L_SHOULDER_ANGLE_DEG),
        R_UPPERARM_IDX: (SHOULDER_AXIS_LOCAL, R_SHOULDER_ANGLE_DEG),
    }
    rotations = []
    translations = []
    scales = []
    for i, rest_m in enumerate(rest_xforms):
        t, r, sc = decompose_mat4(rest_m)
        rq = Gf.Quatf(r)
        if i in overrides:
            axis, angle = overrides[i]
            extra = Gf.Quatf(Gf.Rotation(Gf.Vec3d(*axis), float(angle)).GetQuat())
            rq = rq * extra
        rotations.append(rq)
        translations.append(Gf.Vec3f(float(t[0]), float(t[1]), float(t[2])))
        scales.append(Gf.Vec3h(float(sc[0]), float(sc[1]), float(sc[2])))

    anim_path = skel_prim.GetPath().AppendChild("LyingPose")
    anim = UsdSkel.Animation.Define(stage, anim_path)
    anim.CreateJointsAttr().Set(Vt.TokenArray(joints))
    anim.CreateRotationsAttr().Set(Vt.QuatfArray(rotations))
    anim.CreateTranslationsAttr().Set(Vt.Vec3fArray(translations))
    anim.CreateScalesAttr().Set(Vt.Vec3hArray(scales))
    UsdSkel.BindingAPI.Apply(skel_prim).CreateAnimationSourceRel().SetTargets([anim_path])

    print(f"[anomaly] Lying {prim_path} at ({position[0]:.1f},{position[1]:.1f},{position[2]:.2f}) "
          f"with arm overrides", flush=True)
    return person


def save_screenshot(camera, output_path):
    rgba = camera.get_rgba()
    if rgba is None:
        return False
    try:
        from PIL import Image
        os.makedirs(os.path.dirname(output_path), exist_ok=True)
        Image.fromarray(rgba[:, :, :3]).save(output_path)
        print(f"[anomaly] Saved: {output_path}", flush=True)
        return True
    except Exception as e:
        print(f"[anomaly] Screenshot error: {e}", flush=True)
        return False


# -----------------------------------------------------------------------------
# Main
# -----------------------------------------------------------------------------

def main():
    usd_path = sys.argv[1] if len(sys.argv) > 1 else os.path.join(
        _project_dir, "assets", "environments", "msoe_building",
        "MSOEBuildingV4_physics.usd"
    )
    if not os.path.exists(usd_path):
        print(f"ERROR: USD not found: {usd_path}", flush=True)
        sys.exit(1)

    assets_root = get_assets_root_path()
    if assets_root is None:
        print("[anomaly] ERROR: assets root unavailable", flush=True)
        sys.exit(2)

    world = World(stage_units_in_meters=1.0)
    stage = omni.usd.get_context().get_stage()

    building_prim = stage.DefinePrim("/World/Building", "Xform")
    building_prim.GetReferences().AddReference(
        usd_path, primPath=Sdf.Path("/World/Building")
    )
    setup_physics_scene(stage)
    freeze_doors(stage)

    bbox_cache = UsdGeom.BBoxCache(Usd.TimeCode.Default(), [UsdGeom.Tokens.default_])
    bb = bbox_cache.ComputeWorldBound(stage.GetPrimAtPath("/World/Building"))
    add_ground_plane(stage, bb.ComputeAlignedRange().GetMin()[2] - 0.5)

    add_interior_lighting(stage)
    camera = create_test_camera(CAM_KEY, CAM, stage=stage)

    # Parent prim for people
    prims_utils.create_prim("/World/People", "Xform")

    # Load standing characters
    for i, (xy, target_xy, subpath, label) in enumerate(STANDING_PEOPLE):
        prim_path = f"/World/People/Standing_{i+1}_{label}"
        asset_url = assets_root + subpath
        pos = (xy[0], xy[1], STANDING_Z)
        target = (target_xy[0], target_xy[1], STANDING_Z)
        person = load_standing_character(stage, prim_path, asset_url, pos, target, PERSON_SCALE)
        if person is None:
            # Fallback
            load_standing_character(
                stage, prim_path, assets_root + BIPED_FALLBACK_SUBPATH,
                pos, target, PERSON_SCALE,
            )

    # Load lying (injured) character
    lying_prim_path = "/World/People/Injured_01"
    lying_url = assets_root + LYING_CHARACTER_SUBPATH
    lying_pos = (LYING_POS[0], LYING_POS[1], LYING_Z)
    person_lying = load_lying_character(
        stage, lying_prim_path, lying_url, lying_pos,
        LYING_ROT_DEG, LYING_FACE_YAW, PERSON_SCALE,
    )
    if person_lying is None:
        load_lying_character(
            stage, lying_prim_path, assets_root + BIPED_FALLBACK_SUBPATH,
            lying_pos, LYING_ROT_DEG, LYING_FACE_YAW, PERSON_SCALE,
        )

    world.reset()
    camera.initialize()
    set_focal_length_on_prim(stage, camera)

    print(f"[anomaly] Warmup: {WARMUP_FRAMES} frames", flush=True)
    for _ in range(WARMUP_FRAMES):
        world.step(render=True)

    save_screenshot(camera, SCREENSHOT_PATH)

    print(f"[anomaly] Recording {RECORD_FRAMES} frames to {VIDEO_NAME}.mp4", flush=True)
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

    print("[anomaly] Done!", flush=True)
    os._exit(0)


if __name__ == "__main__":
    main()
