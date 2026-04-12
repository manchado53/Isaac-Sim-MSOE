#!/usr/bin/env python3
"""
Falling Person Demo — keyframed "limp collapse" of a character to the floor.

Visually fakes a ragdoll: over ~2 seconds, the character goes from standing
(T-pose, upright) to our natural relaxed-lying pose (supine with bent
knees/arms at sides), while translating downward with gravity-like easing.

Not true physics — no articulated rigid bodies — but visually convincing
for "person collapses to the floor." Each frame we:
  1. Update `xformOp:translate` Z on the root (fall downward)
  2. Update `xformOp:rotateXYZ` on the child (tilt from 0° → 90° X)
  3. Update the bound `UsdSkel.Animation` joint rotations (rest → relaxed-lying)

Interpolation:
  - Z: quadratic easing (accelerating, gravity-like)
  - Tilt and joints: smooth-step (ease in-out)

Approximate timeline (30 fps):
    frames 0-10      : Standing still (pre-fall, 0.33s)
    frames 10-70     : Fall + collapse (2.0s, Z↓ and tilt 0→90°)
    frames 70-120    : Settled on floor (1.67s, hold)

Output:
    videos/environment/falling_person_demo.mp4
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


# --- Camera + building ---
CAM_KEY = "G"
CAM = TEST_CAMERAS[CAM_KEY]
FLOOR_Z = 14.56         # camera target Z (reference)
REAL_FLOOR_Z = 13.56    # actual walkable surface (from raycast)

# --- Character placement (meters) ---
PERSON_X = -37.0
PERSON_Y = 74.0

# Standing Z (feet on floor): rig origin = floor + ~0.125m
STANDING_Z = REAL_FLOOR_Z + 0.125   # = 13.685

# Lying Z (back on floor): translate = floor + 0.27m
LYING_Z = REAL_FLOOR_Z + 0.27       # = 13.83

# --- Timing ---
FPS = 30
PRE_FALL_FRAMES = 15        # 0.5s standing still
FALL_FRAMES = 60            # 2.0s collapsing
SETTLE_FRAMES = 45          # 1.5s on floor
TOTAL_FRAMES = PRE_FALL_FRAMES + FALL_FRAMES + SETTLE_FRAMES
WARMUP_FRAMES = 20

# --- Pose parameters (match lying_person_natural_demo.py final values) ---
FACE_YAW_DEG = 180.0        # head direction during fall (kept constant)

# Joint overrides — FINAL values to reach at end of fall
SHOULDER_AXIS_LOCAL = (0.0, 0.0, 1.0)
L_UPPERARM_IDX, L_SHOULDER_ANGLE_DEG = 50, -90.0
R_UPPERARM_IDX, R_SHOULDER_ANGLE_DEG = 78,  90.0

FOREARM_AXIS_LOCAL = (1.0, 0.0, 0.0)
L_FOREARM_IDX, L_FOREARM_ANGLE_DEG = 51,  15.0
R_FOREARM_IDX, R_FOREARM_ANGLE_DEG = 79, -15.0

NECK_IDX, NECK_AXIS_LOCAL, NECK_ANGLE_DEG = 36, (0.0, 1.0, 0.0), 20.0

SPINE01_IDX, SPINE02_IDX = 34, 35
SPINE_AXIS_LOCAL, SPINE_ANGLE_DEG = (1.0, 0.0, 0.0), 5.0

LEG_AXIS_LOCAL = (1.0, 0.0, 0.0)
L_THIGH_IDX, L_THIGH_ANGLE_DEG = 3, -12.0
L_CALF_IDX,  L_CALF_ANGLE_DEG  = 4,  18.0
R_THIGH_IDX, R_THIGH_ANGLE_DEG = 18, -8.0
R_CALF_IDX,  R_CALF_ANGLE_DEG  = 19, 10.0

FOOT_AXIS_LOCAL = (0.0, 0.0, 1.0)
L_FOOT_IDX, L_FOOT_ANGLE_DEG = 5,  -15.0
R_FOOT_IDX, R_FOOT_ANGLE_DEG = 21,  12.0

# Collected target overrides for each tunable joint
TARGET_OVERRIDES = {
    L_UPPERARM_IDX: (SHOULDER_AXIS_LOCAL, L_SHOULDER_ANGLE_DEG),
    R_UPPERARM_IDX: (SHOULDER_AXIS_LOCAL, R_SHOULDER_ANGLE_DEG),
    L_FOREARM_IDX:  (FOREARM_AXIS_LOCAL,  L_FOREARM_ANGLE_DEG),
    R_FOREARM_IDX:  (FOREARM_AXIS_LOCAL,  R_FOREARM_ANGLE_DEG),
    NECK_IDX:       (NECK_AXIS_LOCAL,     NECK_ANGLE_DEG),
    SPINE01_IDX:    (SPINE_AXIS_LOCAL,    SPINE_ANGLE_DEG),
    SPINE02_IDX:    (SPINE_AXIS_LOCAL,    SPINE_ANGLE_DEG),
    L_THIGH_IDX:    (LEG_AXIS_LOCAL,      L_THIGH_ANGLE_DEG),
    L_CALF_IDX:     (LEG_AXIS_LOCAL,      L_CALF_ANGLE_DEG),
    R_THIGH_IDX:    (LEG_AXIS_LOCAL,      R_THIGH_ANGLE_DEG),
    R_CALF_IDX:     (LEG_AXIS_LOCAL,      R_CALF_ANGLE_DEG),
    L_FOOT_IDX:     (FOOT_AXIS_LOCAL,     L_FOOT_ANGLE_DEG),
    R_FOOT_IDX:     (FOOT_AXIS_LOCAL,     R_FOOT_ANGLE_DEG),
}

CHARACTER_SUBPATH = "/Isaac/People/Characters/F_Business_02/F_Business_02.usd"
BIPED_FALLBACK_SUBPATH = "/Isaac/People/Characters/Biped_Setup.usd"

VIDEO_NAME = "environment/falling_person_demo"


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
    UsdPhysics.MeshCollisionAPI.Apply(mesh.GetPrim()).CreateApproximationAttr("none")


def decompose_mat4(m):
    t = m.ExtractTranslation()
    r = m.ExtractRotation().GetQuat()
    col0 = Gf.Vec3d(m[0][0], m[0][1], m[0][2])
    col1 = Gf.Vec3d(m[1][0], m[1][1], m[1][2])
    col2 = Gf.Vec3d(m[2][0], m[2][1], m[2][2])
    return t, r, Gf.Vec3f(float(col0.GetLength()),
                           float(col1.GetLength()),
                           float(col2.GetLength()))


def find_skeleton(root_prim):
    for p in Usd.PrimRange(root_prim):
        if p.GetTypeName() == "Skeleton":
            return p
    return None


def smoothstep(t):
    """Ease in-out cubic: 3t^2 - 2t^3. Input clamped to [0,1]."""
    t = max(0.0, min(1.0, t))
    return t * t * (3.0 - 2.0 * t)


def quadratic_fall(t):
    """Accelerating fall (gravity-like). Input clamped to [0,1]."""
    t = max(0.0, min(1.0, t))
    return t * t


def slerp_quat(q0, q1, t):
    """Spherical interpolation between two Gf.Quatf. Handles sign flip."""
    # Dot product
    dot = (q0.GetReal() * q1.GetReal()
           + q0.GetImaginary()[0] * q1.GetImaginary()[0]
           + q0.GetImaginary()[1] * q1.GetImaginary()[1]
           + q0.GetImaginary()[2] * q1.GetImaginary()[2])
    # If dot < 0, invert one to take shortest path
    if dot < 0.0:
        q1 = Gf.Quatf(-q1.GetReal(), -q1.GetImaginary())
        dot = -dot
    # Linear for very close quats
    if dot > 0.9995:
        r = q0.GetReal() + (q1.GetReal() - q0.GetReal()) * t
        i0 = q0.GetImaginary()[0] + (q1.GetImaginary()[0] - q0.GetImaginary()[0]) * t
        i1 = q0.GetImaginary()[1] + (q1.GetImaginary()[1] - q0.GetImaginary()[1]) * t
        i2 = q0.GetImaginary()[2] + (q1.GetImaginary()[2] - q0.GetImaginary()[2]) * t
        q = Gf.Quatf(r, Gf.Vec3f(i0, i1, i2))
        # Normalize
        norm = math.sqrt(r*r + i0*i0 + i1*i1 + i2*i2)
        if norm > 1e-8:
            return Gf.Quatf(r/norm, Gf.Vec3f(i0/norm, i1/norm, i2/norm))
        return q
    dot = max(-1.0, min(1.0, dot))
    theta_0 = math.acos(dot)
    theta = theta_0 * t
    sin_theta = math.sin(theta)
    sin_theta_0 = math.sin(theta_0)
    s0 = math.cos(theta) - dot * sin_theta / sin_theta_0
    s1 = sin_theta / sin_theta_0
    r = s0 * q0.GetReal() + s1 * q1.GetReal()
    i0 = s0 * q0.GetImaginary()[0] + s1 * q1.GetImaginary()[0]
    i1 = s0 * q0.GetImaginary()[1] + s1 * q1.GetImaginary()[1]
    i2 = s0 * q0.GetImaginary()[2] + s1 * q1.GetImaginary()[2]
    return Gf.Quatf(r, Gf.Vec3f(i0, i1, i2))


def save_screenshot(camera, output_path):
    rgba = camera.get_rgba()
    if rgba is None:
        return False
    try:
        from PIL import Image
        os.makedirs(os.path.dirname(output_path), exist_ok=True)
        Image.fromarray(rgba[:, :, :3]).save(output_path)
        return True
    except Exception:
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
        print("[fall] assets root unavailable", flush=True)
        sys.exit(2)
    character_usd = assets_root + CHARACTER_SUBPATH

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

    # --- Load character standing ---
    print(f"[fall] Loading character: {character_usd}", flush=True)
    person = prims_utils.create_prim("/World/Falling_Person", "Xform",
                                      usd_path=character_usd)
    if not person or not person.IsValid():
        print("[fall] FATAL: character load failed", flush=True)
        simulation_app.close()
        sys.exit(3)

    # Root: translate op (we'll animate Z during fall)
    root_xform = UsdGeom.Xformable(person)
    root_xform.ClearXformOpOrder()
    translate_op = root_xform.AddTranslateOp(
        precision=UsdGeom.XformOp.PrecisionDouble
    )
    translate_op.Set(Gf.Vec3d(PERSON_X, PERSON_Y, STANDING_Z))

    # Child: rotateXYZ (start at 0,0,FACE_YAW), scale 1.0 (no conversion needed)
    child_xform_op = None
    for child in person.GetChildren():
        if child.IsA(UsdGeom.Xformable):
            r = child.GetAttribute("xformOp:rotateXYZ")
            if r and r.IsValid():
                r.Set(Gf.Vec3f(0.0, 0.0, FACE_YAW_DEG))
                child_xform_op = r  # remember which one we tweaked
            # scale stays at its native (1,1,1) — no change
    if child_xform_op is None:
        print("[fall] FATAL: could not find child rotateXYZ", flush=True)
        simulation_app.close()
        sys.exit(4)

    # --- Find skeleton and build animation prim that we'll poke each frame ---
    skel_prim = find_skeleton(person)
    if skel_prim is None:
        print("[fall] FATAL: no Skeleton under character", flush=True)
        simulation_app.close()
        sys.exit(5)

    sk = UsdSkel.Skeleton(skel_prim)
    joints = list(sk.GetJointsAttr().Get() or [])
    rest_xforms = list(sk.GetRestTransformsAttr().Get() or [])
    n_joints = len(joints)
    print(f"[fall] Skeleton: {n_joints} joints", flush=True)

    # Precompute rest quats for all joints
    rest_quats = []
    rest_translations = []
    rest_scales = []
    for m in rest_xforms:
        t, r, s = decompose_mat4(m)
        rest_quats.append(Gf.Quatf(r))
        rest_translations.append(Gf.Vec3f(float(t[0]), float(t[1]), float(t[2])))
        rest_scales.append(Gf.Vec3h(float(s[0]), float(s[1]), float(s[2])))

    # Precompute TARGET quats (rest * extra) for each overridden joint
    target_quats = list(rest_quats)  # copy
    for idx, (axis, angle_deg) in TARGET_OVERRIDES.items():
        if idx < n_joints:
            extra = Gf.Quatf(Gf.Rotation(Gf.Vec3d(*axis), float(angle_deg)).GetQuat())
            target_quats[idx] = rest_quats[idx] * extra

    # Create + bind SkelAnimation
    anim_path = skel_prim.GetPath().AppendChild("FallAnim")
    anim = UsdSkel.Animation.Define(stage, anim_path)
    anim.CreateJointsAttr().Set(Vt.TokenArray(joints))
    anim.CreateRotationsAttr().Set(Vt.QuatfArray(rest_quats))
    anim.CreateTranslationsAttr().Set(Vt.Vec3fArray(rest_translations))
    anim.CreateScalesAttr().Set(Vt.Vec3hArray(rest_scales))
    UsdSkel.BindingAPI.Apply(skel_prim).CreateAnimationSourceRel().SetTargets([anim_path])
    rotations_attr = anim.GetRotationsAttr()

    # --- Reset + initialize ---
    world.reset()
    camera.initialize()
    set_focal_length_on_prim(stage, camera)

    # Warmup: skinning settles
    for _ in range(WARMUP_FRAMES):
        world.step(render=True)

    # --- Animated fall loop ---
    print(f"[fall] Recording {TOTAL_FRAMES} frames "
          f"(pre:{PRE_FALL_FRAMES}, fall:{FALL_FRAMES}, settle:{SETTLE_FRAMES})",
          flush=True)
    with VideoRecorder(VIDEO_NAME, fps=FPS) as recorder:
        for i in range(TOTAL_FRAMES):
            # Determine phase and interpolation parameter t in [0,1]
            if i < PRE_FALL_FRAMES:
                # Standing still
                t_pos = 0.0
                t_pose = 0.0
            elif i < PRE_FALL_FRAMES + FALL_FRAMES:
                # Falling + collapsing
                raw = (i - PRE_FALL_FRAMES) / float(FALL_FRAMES)
                t_pos = quadratic_fall(raw)      # gravity-accelerating
                t_pose = smoothstep(raw)          # smooth pose blend
            else:
                # Settled on floor
                t_pos = 1.0
                t_pose = 1.0

            # Position: interpolate translate Z from STANDING_Z down to LYING_Z
            cur_z = STANDING_Z + (LYING_Z - STANDING_Z) * t_pos
            translate_op.Set(Gf.Vec3d(PERSON_X, PERSON_Y, cur_z))

            # Body tilt: 0° → 90° around X, face-yaw constant
            tilt_x = 90.0 * t_pose
            child_xform_op.Set(Gf.Vec3f(tilt_x, 0.0, FACE_YAW_DEG))

            # Joint rotations: slerp rest → target per joint
            if t_pose >= 0.9999:
                new_rotations = target_quats
            elif t_pose <= 0.0001:
                new_rotations = rest_quats
            else:
                new_rotations = []
                for j in range(n_joints):
                    if j in TARGET_OVERRIDES:
                        new_rotations.append(slerp_quat(rest_quats[j],
                                                         target_quats[j], t_pose))
                    else:
                        new_rotations.append(rest_quats[j])
            rotations_attr.Set(Vt.QuatfArray(new_rotations))

            world.step(render=True)
            try:
                rgba = camera.get_rgba()
                if rgba is not None:
                    recorder.capture_rgba(rgba)
            except Exception:
                pass
            if i % 20 == 0:
                print(f"  frame {i:3d}/{TOTAL_FRAMES}  t_pos={t_pos:.2f}  "
                      f"t_pose={t_pose:.2f}  z={cur_z:.2f}  tilt={tilt_x:.1f}°",
                      flush=True)

    # Final screenshot for quick preview
    save_screenshot(camera, os.path.join(_project_dir, "images",
                                          "falling_person_demo_final.png"))
    print("[fall] Done!", flush=True)
    os._exit(0)


if __name__ == "__main__":
    main()
