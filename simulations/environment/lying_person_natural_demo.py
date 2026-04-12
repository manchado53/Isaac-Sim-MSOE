#!/usr/bin/env python3
"""
Lying Person (Natural) Demo — UsdSkel-posed injured person on ground floor.

Step up from lying_person_demo.py: uses UsdSkel.Animation joint manipulation
to pose the character with arms at sides (natural supine) instead of T-pose
arms sticking out. This is the real "Option B" from our research.

Approach:
  1. Apply body-level 90° tilt (same as lying_person_demo.py) to lay the
     character flat on their back.
  2. Add a UsdSkel.Animation prim bound to the character's Skeleton.
     - For most joints: copy rest transforms (no change)
     - For L_Upperarm (index 50) + R_Upperarm (index 78): apply additional
       rotation to swing arm from T-pose-out-to-side DOWN along the body
  3. Iterate on the shoulder rotation axis/angle to find the natural pose.

Joint info from logs/character_skeleton.json (F_Business_02):
  - Skeleton has 101 joints (NVIDIA biped style)
  - L_Clavicle=49, L_Upperarm=50, L_Forearm=51, L_Hand=55
  - R_Clavicle=77, R_Upperarm=78, R_Forearm=79, R_Hand=83
  - Joints are at their REST positions in a T-pose by default
  - Each joint's local +Y axis points along the bone direction

Tuning parameters (adjust these if pose looks wrong):
  - SHOULDER_EXTRA_AXIS / SHOULDER_EXTRA_ANGLE_DEG: rotation to bring arm
    from T-pose horizontal to downward-along-body
  - SHOULDER_PRE_MULTIPLY: whether to apply extra before or after rest rot
    (changes whether the rotation is in joint-local or parent-local frame)

Output:
    images/lying_person_natural_demo.png
    videos/environment/lying_person_natural_demo.mp4
    logs/lying_person_natural_demo.json  — diagnostic joint info
"""

import os
import sys
import math
import json

os.environ["OMNI_KIT_ACCEPT_EULA"] = "YES"

from isaacsim import SimulationApp
simulation_app = SimulationApp({"headless": True})

import numpy as np
import carb
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
FLOOR_Z = 14.56  # ground floor height in feet

# Position (same as lying_person_demo.py)
PERSON_X = -37.0
PERSON_Y = 74.0

# Body-level tilt (rotate character supine)
LYING_ROT_AXIS = Gf.Vec3d(1, 0, 0)
LYING_ROT_DEG = 90.0
# Z offset (in METERS — stage units for V4_physics.usd).
# Real walkable floor Z = 13.56 (from raycast).
# Measured at offset=0: bbox z_min=14.29 (body floating 0.73m above floor).
# Compensation: offset = -0.73 puts back on floor.
LYING_Z_OFFSET = -0.73
PERSON_Z = FLOOR_Z + LYING_Z_OFFSET
FACE_YAW_DEG = 180.0
# Scale=1.0: stage + character both metersPerUnit=1.0, no conversion needed.
# (Earlier 3.28× was based on wrong assumption that stage was in feet.)
PERSON_SCALE = 1.0

# --- UsdSkel joint manipulation parameters ---
# Joint indices from logs/character_skeleton.json
L_UPPERARM_IDX = 50
R_UPPERARM_IDX = 78

# Rotation to bring upperarm from T-pose (arm out to side) down along body.
# Joint's local +Y points along the bone; in T-pose, local +Y → world +X (left arm).
# We want arm to point along body length (which after body-tilt becomes into floor).
# Rotation around local +X axis by ~-85° should swing local +Y from +X-ish toward +Z-ish
# which in the T-pose frame is "downward along the body."
#
# CRUCIAL: This value needs experimental tuning. Start with -80° and iterate.
# Mirror for right arm.
SHOULDER_ROT_AXIS_LOCAL = (0.0, 0.0, 1.0)   # rotate around local Z
L_SHOULDER_ANGLE_DEG = -90.0                # left arm: -90° for arms fully at sides
R_SHOULDER_ANGLE_DEG = 90.0                 # right arm: opposite sign

# Forearm bend (0 = keep arm straight along upper arm direction)
L_FOREARM_IDX = 51
R_FOREARM_IDX = 79
FOREARM_ROT_AXIS_LOCAL = (1.0, 0.0, 0.0)   # around local X
FOREARM_ANGLE_DEG = 15.0                    # slight elbow bend (was 0)

# --- Extended joint overrides for natural relaxed supine pose ---
# Indices from logs/character_skeleton.json. Axes & signs may flip during
# iteration — each joint's local axis convention in this NVIDIA biped is not
# obvious without visualization. First-pass values; expect to tune.

# Head / neck — turn head to one side
NECK_IDX = 36                               # NeckTwist01
NECK_AXIS_LOCAL = (0.0, 1.0, 0.0)           # around local Y (twist-like)
NECK_ANGLE_DEG = 20.0

# Spine — slight curve, NOT ramrod straight
SPINE01_IDX = 34
SPINE02_IDX = 35
SPINE_AXIS_LOCAL = (1.0, 0.0, 0.0)          # around local X (forward bend)
SPINE_ANGLE_DEG = 5.0

# Legs — asymmetric knee bend so legs aren't rigid
L_THIGH_IDX = 3
L_CALF_IDX = 4
L_FOOT_IDX = 5
R_THIGH_IDX = 18
R_CALF_IDX = 19
R_FOOT_IDX = 21
LEG_AXIS_LOCAL = (1.0, 0.0, 0.0)            # around local X (pitch)
L_THIGH_ANGLE_DEG = -12.0
L_CALF_ANGLE_DEG = 18.0
R_THIGH_ANGLE_DEG = -8.0                    # smaller → asymmetric
R_CALF_ANGLE_DEG = 10.0

# Feet — flopped outward (both directions for asymmetry)
FOOT_AXIS_LOCAL = (0.0, 0.0, 1.0)           # around local Z (yaw)
L_FOOT_ANGLE_DEG = -15.0
R_FOOT_ANGLE_DEG = 12.0

# Character asset
CHARACTER_SUBPATH = "/Isaac/People/Characters/F_Business_02/F_Business_02.usd"
BIPED_FALLBACK_SUBPATH = "/Isaac/People/Characters/Biped_Setup.usd"

# Output
RECORD_FRAMES = 90  # 3s at 30fps — static scene, shorter is fine
WARMUP_FRAMES = 30
SCREENSHOT_PATH = os.path.join(_project_dir, "images", "lying_person_natural_demo.png")
VIDEO_NAME = "environment/lying_person_natural_demo"
DIAG_JSON_PATH = os.path.join(_project_dir, "logs", "lying_person_natural_demo.json")


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


def load_body_tilted_character(stage, prim_path, character_usd, position,
                                rot_axis, rot_deg, face_yaw_deg, scale):
    """Load character with body tilted (from lying_person_demo.py).

    Returns (prim, bbox_min, bbox_max) for diagnostic purposes.
    """
    print(f"[natural] Loading character: {character_usd}", flush=True)
    try:
        person = prims_utils.create_prim(prim_path, "Xform", usd_path=character_usd)
    except Exception as e:
        print(f"[natural] ERROR loading character: {e}", flush=True)
        return None

    if not person or not person.IsValid():
        return None

    # Root: translate only
    xformable = UsdGeom.Xformable(person)
    xformable.ClearXformOpOrder()
    translate_op = xformable.AddTranslateOp(precision=UsdGeom.XformOp.PrecisionDouble)
    translate_op.Set(Gf.Vec3d(float(position[0]), float(position[1]), float(position[2])))

    # Child: rotation + scale
    if float(rot_axis[0]) != 0.0:
        tilt_euler = Gf.Vec3f(float(rot_deg), 0.0, 0.0)
    elif float(rot_axis[1]) != 0.0:
        tilt_euler = Gf.Vec3f(0.0, float(rot_deg), 0.0)
    else:
        tilt_euler = Gf.Vec3f(0.0, 0.0, float(rot_deg))
    if float(rot_axis[2]) == 0.0:
        tilt_euler = Gf.Vec3f(tilt_euler[0], tilt_euler[1], float(face_yaw_deg))

    for child in person.GetChildren():
        if not child.IsA(UsdGeom.Xformable):
            continue
        rot_attr = child.GetAttribute("xformOp:rotateXYZ")
        if rot_attr and rot_attr.IsValid():
            rot_attr.Set(tilt_euler)
        s_attr = child.GetAttribute("xformOp:scale")
        if s_attr and s_attr.IsValid():
            current_scale = s_attr.Get()
            s_attr.Set(type(current_scale)(scale, scale, scale))

    print(f"[natural] Placed tilted character at {position}, tilt={rot_deg}°", flush=True)
    return person


def decompose_mat4(m):
    """Extract translation, rotation (Quatf), scale from a Gf.Matrix4d."""
    t = m.ExtractTranslation()
    r = m.ExtractRotation().GetQuat()
    # Scale: use column norms (assume no shear)
    col0 = Gf.Vec3d(m[0][0], m[0][1], m[0][2])
    col1 = Gf.Vec3d(m[1][0], m[1][1], m[1][2])
    col2 = Gf.Vec3d(m[2][0], m[2][1], m[2][2])
    sx = col0.GetLength()
    sy = col1.GetLength()
    sz = col2.GetLength()
    return t, r, Gf.Vec3f(float(sx), float(sy), float(sz))


def apply_shoulder_override(rest_quat, axis_local, angle_deg, pre_multiply=False):
    """Compose an additional rotation onto a rest quaternion.

    Args:
        rest_quat: Gf.Quatf or Quatd for the joint's rest rotation
        axis_local: 3-tuple for rotation axis in joint-local frame
        angle_deg: angle in degrees
        pre_multiply: if True, extra * rest (extra in parent frame).
                      if False, rest * extra (extra in joint-local frame).
    """
    extra = Gf.Rotation(Gf.Vec3d(*axis_local), float(angle_deg)).GetQuat()
    # Convert to Quatf for output consistency
    rq = Gf.Quatf(rest_quat) if not isinstance(rest_quat, Gf.Quatf) else rest_quat
    eq = Gf.Quatf(extra)
    if pre_multiply:
        return eq * rq
    return rq * eq


def find_skeleton(person_prim):
    """Find the first Skeleton prim under the character."""
    for prim in Usd.PrimRange(person_prim):
        if prim.GetTypeName() == "Skeleton":
            return prim
    return None


def build_skel_animation(stage, skeleton_prim, overrides, diag_out=None):
    """Build a UsdSkel.Animation prim that reproduces rest pose with joint overrides.

    Args:
        stage: USD stage
        skeleton_prim: Skeleton prim
        overrides: dict[joint_index -> (axis_local, angle_deg, pre_multiply)]
        diag_out: list to append joint-override diagnostics to

    Returns:
        SkelAnimation prim path (string)
    """
    skel = UsdSkel.Skeleton(skeleton_prim)
    joints = list(skel.GetJointsAttr().Get() or [])
    rest_xforms = list(skel.GetRestTransformsAttr().Get() or [])

    if not joints or len(rest_xforms) != len(joints):
        print(f"[natural] ERROR: skeleton has {len(joints)} joints "
              f"but {len(rest_xforms)} rest transforms", flush=True)
        return None

    print(f"[natural] Skeleton has {len(joints)} joints; building animation...",
          flush=True)

    rotations = []
    translations = []
    scales = []
    for i, rest_m in enumerate(rest_xforms):
        t, r, s = decompose_mat4(rest_m)
        rq = Gf.Quatf(r)
        if i in overrides:
            axis_local, angle_deg, pre_mul = overrides[i]
            rq = apply_shoulder_override(rq, axis_local, angle_deg, pre_mul)
            if diag_out is not None:
                diag_out.append({
                    "joint_index": i,
                    "joint_path": str(joints[i]),
                    "rest_quat_wxyz": [float(r.GetReal()),
                                        float(r.GetImaginary()[0]),
                                        float(r.GetImaginary()[1]),
                                        float(r.GetImaginary()[2])],
                    "axis_local": list(axis_local),
                    "angle_deg": angle_deg,
                    "pre_multiply": pre_mul,
                    "new_quat_wxyz": [float(rq.GetReal()),
                                       float(rq.GetImaginary()[0]),
                                       float(rq.GetImaginary()[1]),
                                       float(rq.GetImaginary()[2])],
                })
        rotations.append(rq)
        translations.append(Gf.Vec3f(float(t[0]), float(t[1]), float(t[2])))
        scales.append(Gf.Vec3h(float(s[0]), float(s[1]), float(s[2])))

    # Create SkelAnimation prim under the skeleton
    anim_path = skeleton_prim.GetPath().AppendChild("CustomLyingPose")
    anim_prim = UsdSkel.Animation.Define(stage, anim_path)
    anim_prim.CreateJointsAttr().Set(Vt.TokenArray(joints))
    anim_prim.CreateRotationsAttr().Set(Vt.QuatfArray(rotations))
    anim_prim.CreateTranslationsAttr().Set(Vt.Vec3fArray(translations))
    anim_prim.CreateScalesAttr().Set(Vt.Vec3hArray(scales))

    # Bind animation to skeleton
    binding_api = UsdSkel.BindingAPI.Apply(skeleton_prim)
    rel = binding_api.CreateAnimationSourceRel()
    rel.SetTargets([anim_path])

    print(f"[natural] Bound SkelAnimation at {anim_path} with "
          f"{len(overrides)} joint overrides", flush=True)
    return str(anim_path)


def save_screenshot(camera, output_path):
    rgba = camera.get_rgba()
    if rgba is None:
        return False
    try:
        from PIL import Image
        os.makedirs(os.path.dirname(output_path), exist_ok=True)
        Image.fromarray(rgba[:, :, :3]).save(output_path)
        print(f"[natural] Saved: {output_path}", flush=True)
        return True
    except Exception as e:
        print(f"[natural] Screenshot error: {e}", flush=True)
        return False


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
        print("[natural] ERROR: assets root unavailable", flush=True)
        sys.exit(2)
    character_usd = assets_root + CHARACTER_SUBPATH
    biped_fallback = assets_root + BIPED_FALLBACK_SUBPATH

    # Scene setup
    world = World(stage_units_in_meters=1.0)
    stage = omni.usd.get_context().get_stage()

    building_prim = stage.DefinePrim("/World/Building", "Xform")
    building_prim.GetReferences().AddReference(
        usd_path, primPath=Sdf.Path("/World/Building")
    )
    setup_physics_scene(stage)
    freeze_doors(stage)

    bbox_cache = UsdGeom.BBoxCache(Usd.TimeCode.Default(), [UsdGeom.Tokens.default_])
    bldg_bbox = bbox_cache.ComputeWorldBound(stage.GetPrimAtPath("/World/Building"))
    bb_min = bldg_bbox.ComputeAlignedRange().GetMin()
    add_ground_plane(stage, bb_min[2] - 0.5)

    add_interior_lighting(stage)
    camera = create_test_camera(CAM_KEY, CAM, stage=stage)

    # Load character with body tilt
    person = load_body_tilted_character(
        stage, "/World/Person_01", character_usd,
        position=(PERSON_X, PERSON_Y, PERSON_Z),
        rot_axis=LYING_ROT_AXIS, rot_deg=LYING_ROT_DEG,
        face_yaw_deg=FACE_YAW_DEG, scale=PERSON_SCALE,
    )
    if person is None:
        print("[natural] Retrying with Biped_Setup fallback...", flush=True)
        person = load_body_tilted_character(
            stage, "/World/Person_01", biped_fallback,
            position=(PERSON_X, PERSON_Y, PERSON_Z),
            rot_axis=LYING_ROT_AXIS, rot_deg=LYING_ROT_DEG,
            face_yaw_deg=FACE_YAW_DEG, scale=PERSON_SCALE,
        )
    if person is None:
        print("[natural] FATAL: could not load character", flush=True)
        simulation_app.close()
        sys.exit(3)

    # Find the skeleton
    skeleton_prim = find_skeleton(person)
    if skeleton_prim is None:
        print("[natural] FATAL: no Skeleton found under character", flush=True)
        simulation_app.close()
        sys.exit(4)
    print(f"[natural] Skeleton: {skeleton_prim.GetPath()}", flush=True)

    # Build joint override map — extended relaxed supine pose
    overrides = {
        # Shoulders: arms at sides (proven)
        L_UPPERARM_IDX: (SHOULDER_ROT_AXIS_LOCAL, L_SHOULDER_ANGLE_DEG, False),
        R_UPPERARM_IDX: (SHOULDER_ROT_AXIS_LOCAL, R_SHOULDER_ANGLE_DEG, False),
        # Elbows: slight bend
        L_FOREARM_IDX:  (FOREARM_ROT_AXIS_LOCAL,  FOREARM_ANGLE_DEG,     False),
        R_FOREARM_IDX:  (FOREARM_ROT_AXIS_LOCAL, -FOREARM_ANGLE_DEG,     False),
        # Head: turned to one side
        NECK_IDX:       (NECK_AXIS_LOCAL,          NECK_ANGLE_DEG,        False),
        # Spine: slight curve
        SPINE01_IDX:    (SPINE_AXIS_LOCAL,         SPINE_ANGLE_DEG,       False),
        SPINE02_IDX:    (SPINE_AXIS_LOCAL,         SPINE_ANGLE_DEG,       False),
        # Legs: asymmetric bend
        L_THIGH_IDX:    (LEG_AXIS_LOCAL,           L_THIGH_ANGLE_DEG,     False),
        L_CALF_IDX:     (LEG_AXIS_LOCAL,           L_CALF_ANGLE_DEG,      False),
        R_THIGH_IDX:    (LEG_AXIS_LOCAL,           R_THIGH_ANGLE_DEG,     False),
        R_CALF_IDX:     (LEG_AXIS_LOCAL,           R_CALF_ANGLE_DEG,      False),
        # Feet: flopped outward
        L_FOOT_IDX:     (FOOT_AXIS_LOCAL,          L_FOOT_ANGLE_DEG,      False),
        R_FOOT_IDX:     (FOOT_AXIS_LOCAL,          R_FOOT_ANGLE_DEG,      False),
    }

    diag_out = []
    anim_path = build_skel_animation(stage, skeleton_prim, overrides, diag_out)

    # Dump diagnostic
    os.makedirs(os.path.dirname(DIAG_JSON_PATH), exist_ok=True)
    with open(DIAG_JSON_PATH, "w") as f:
        json.dump({
            "skeleton_path": str(skeleton_prim.GetPath()),
            "anim_path": anim_path,
            "overrides": diag_out,
            "shoulder_axis_local": SHOULDER_ROT_AXIS_LOCAL,
            "l_angle_deg": L_SHOULDER_ANGLE_DEG,
            "r_angle_deg": R_SHOULDER_ANGLE_DEG,
        }, f, indent=2)

    # Reset + initialize
    world.reset()
    camera.initialize()
    set_focal_length_on_prim(stage, camera)

    # Warmup (skinning needs frames to update)
    print(f"[natural] Warmup: {WARMUP_FRAMES} frames", flush=True)
    for _ in range(WARMUP_FRAMES):
        world.step(render=True)

    # Character bbox after pose
    try:
        bbox_cache_p = UsdGeom.BBoxCache(Usd.TimeCode.Default(),
                                         [UsdGeom.Tokens.default_])
        pbbox = bbox_cache_p.ComputeWorldBound(person)
        pmin = pbbox.ComputeAlignedRange().GetMin()
        pmax = pbbox.ComputeAlignedRange().GetMax()
        print(f"[natural] Character world bbox: "
              f"min=({pmin[0]:.2f},{pmin[1]:.2f},{pmin[2]:.2f}) "
              f"max=({pmax[0]:.2f},{pmax[1]:.2f},{pmax[2]:.2f})", flush=True)
    except Exception as e:
        print(f"[natural] bbox diag failed: {e}", flush=True)

    save_screenshot(camera, SCREENSHOT_PATH)

    print(f"[natural] Recording {RECORD_FRAMES} frames", flush=True)
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

    print("[natural] Done!", flush=True)
    os._exit(0)


if __name__ == "__main__":
    main()
