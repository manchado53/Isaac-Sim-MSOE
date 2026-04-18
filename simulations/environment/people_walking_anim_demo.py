#!/usr/bin/env python3
"""
People Walking (Animation-enabled) Demo — W2.5 attempt.

Baby step #2.5: same as people_walking_demo.py but enables the omni.anim.*
extensions so the character's AnimationGraph evaluates. If successful,
the character should play idle/walk animation (breathing, limb movement)
while we update their world position each frame.

Key differences from people_walking_demo.py:
  - Enables omni.anim.graph.core, omni.anim.people, omni.anim.navigation.core,
    omni.anim.retarget.core, omni.kit.scripting BEFORE loading the character
  - Disables navmesh requirement (navmesh_enabled = False)
  - Sits a warmup longer to let the AnimationGraph initialize
  - Extra debug logging

Risk: NVIDIA forum reports headless animation can remain in T-pose.
Expected outcome: either (a) character animates idle + slides along path
(partial success), or (b) character stays in T-pose (failure, need W3).

Output:
    videos/environment/people_walking_anim_demo.mp4
    images/people_walking_anim_demo_start.png
    images/people_walking_anim_demo_mid.png
    images/people_walking_anim_demo_end.png
"""

import os
import sys
import math

os.environ["OMNI_KIT_ACCEPT_EULA"] = "YES"

# Launch SimulationApp first. We enable anim extensions immediately after.
from isaacsim import SimulationApp
simulation_app = SimulationApp({"headless": True})

# --- Enable animation extensions BEFORE touching the stage ---
print("[anim_demo] Enabling animation extensions...", flush=True)
import omni.kit.app
_ext_manager = omni.kit.app.get_app().get_extension_manager()
_ANIM_EXTENSIONS = [
    "omni.kit.scripting",        # BehaviorScript runtime
    "omni.anim.timeline",
    "omni.anim.graph.core",       # AnimationGraph evaluation
    "omni.anim.retarget.core",    # Skeleton retargeting
    "omni.anim.navigation.core",  # Navigation (even if NavMesh disabled)
    "omni.anim.people",           # Character command system
]
for _ext in _ANIM_EXTENSIONS:
    ok = _ext_manager.set_extension_enabled_immediate(_ext, True)
    print(f"[anim_demo]   {_ext}: {'OK' if ok else 'FAILED'}", flush=True)

# Configure people/nav settings BEFORE loading anything
import carb
_settings = carb.settings.get_settings()
_settings.set("/exts/omni.anim.people/navigation_settings/navmesh_enabled", False)
_settings.set("/exts/omni.anim.people/navigation_settings/dynamic_avoidance_enabled", False)
_settings.set("/persistent/exts/omni.anim.navigation.core/navMesh/viewNavMesh", False)
_settings.set("/rtx/raytracing/fractionalCutoutOpacity", True)  # Needed per sdg_scheduler
print("[anim_demo] Settings configured.", flush=True)

# Pump app update so extension loading takes effect
for _ in range(5):
    simulation_app.update()

# --- Now imports that may depend on extensions ---
import numpy as np
import omni.usd
from pxr import Usd, UsdGeom, UsdPhysics, Sdf, Gf

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
FLOOR_Z = 14.56  # feet

WAYPOINTS = [
    (-38.0, 82.0, FLOOR_Z),
    (-32.0, 78.0, FLOOR_Z),
    (-28.0, 72.0, FLOOR_Z),
]
WALK_SPEED_FT_PER_SEC = 3.0
FPS = 30
WARMUP_FRAMES = 60  # Longer warmup for AnimationGraph to initialize

CHARACTER_SUBPATH = "/Isaac/People/Characters/F_Business_02/F_Business_02.usd"
BIPED_FALLBACK_SUBPATH = "/Isaac/People/Characters/Biped_Setup.usd"

VIDEO_NAME = "environment/people_walking_anim_demo"
IMG_DIR = os.path.join(_project_dir, "images")
SCREENSHOT_START = os.path.join(IMG_DIR, "people_walking_anim_demo_start.png")
SCREENSHOT_MID = os.path.join(IMG_DIR, "people_walking_anim_demo_mid.png")
SCREENSHOT_END = os.path.join(IMG_DIR, "people_walking_anim_demo_end.png")


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


def set_character_pose(person_prim, position, face_dir_xy):
    """Update xformOp:translate and xformOp:orient on a character prim."""
    translate_attr = person_prim.GetAttribute("xformOp:translate")
    if translate_attr and translate_attr.IsValid():
        translate_attr.Set(Gf.Vec3d(float(position[0]), float(position[1]), float(position[2])))
    else:
        UsdGeom.Xformable(person_prim).AddTranslateOp().Set(
            Gf.Vec3d(float(position[0]), float(position[1]), float(position[2]))
        )

    dx, dy = float(face_dir_xy[0]), float(face_dir_xy[1])
    if abs(dx) < 1e-6 and abs(dy) < 1e-6:
        return
    face_angle_deg = math.degrees(math.atan2(dy, dx))
    rot_quat = Gf.Rotation(Gf.Vec3d(0, 0, 1), face_angle_deg).GetQuat()

    orient_attr = person_prim.GetAttribute("xformOp:orient")
    if orient_attr and orient_attr.IsValid():
        current = orient_attr.Get()
        if isinstance(current, Gf.Quatf):
            orient_attr.Set(Gf.Quatf(rot_quat))
        else:
            orient_attr.Set(rot_quat)
    else:
        UsdGeom.Xformable(person_prim).AddOrientOp().Set(Gf.Quatf(rot_quat))


def log_skeleton_info(stage, prim_path):
    """Diagnostic: inspect the character's SkelRoot / AnimationGraph status."""
    from pxr import UsdSkel
    root = stage.GetPrimAtPath(prim_path)
    if not root or not root.IsValid():
        print(f"[anim_demo] Skeleton info: prim {prim_path} invalid", flush=True)
        return
    skel_count = 0
    anim_count = 0
    anim_graph_count = 0
    for p in Usd.PrimRange(root):
        t = p.GetTypeName()
        if t == "SkelRoot":
            skel_count += 1
        elif t == "SkelAnimation":
            anim_count += 1
        elif "AnimationGraph" in t or "AnimGraph" in t:
            anim_graph_count += 1
    print(f"[anim_demo] Skeleton inspection: SkelRoot={skel_count}, "
          f"SkelAnimation={anim_count}, AnimationGraph={anim_graph_count}",
          flush=True)


def load_character(stage, prim_path, character_usd, position, face_dir_xy):
    print(f"[anim_demo] Loading character: {character_usd}", flush=True)
    try:
        person = prims_utils.create_prim(prim_path, "Xform", usd_path=character_usd)
    except Exception as e:
        print(f"[anim_demo] ERROR loading character: {e}", flush=True)
        return None
    if not person or not person.IsValid():
        return None
    set_character_pose(person, position, face_dir_xy)
    return person


def save_screenshot(camera, output_path):
    rgba = camera.get_rgba()
    if rgba is None:
        return False
    try:
        from PIL import Image
        os.makedirs(os.path.dirname(output_path), exist_ok=True)
        Image.fromarray(rgba[:, :, :3]).save(output_path)
        print(f"[anim_demo] Saved: {output_path}", flush=True)
        return True
    except Exception as e:
        print(f"[anim_demo] Screenshot error: {e}", flush=True)
        return False


def build_frames_along_path(waypoints, walk_speed_ft_per_sec, fps):
    segments = []
    total_dist = 0.0
    for i in range(len(waypoints) - 1):
        p0 = np.array(waypoints[i], dtype=float)
        p1 = np.array(waypoints[i + 1], dtype=float)
        seg_dist = float(np.linalg.norm(p1[:2] - p0[:2]))
        segments.append({"p0": p0, "p1": p1, "dist": seg_dist, "cum": total_dist})
        total_dist += seg_dist
    duration_sec = total_dist / walk_speed_ft_per_sec
    num_frames = max(1, int(duration_sec * fps))
    print(f"[anim_demo] Path: {total_dist:.1f}ft total, {duration_sec:.1f}s, {num_frames} frames", flush=True)

    frames = []
    for f in range(num_frames):
        t = f / max(1, num_frames - 1)
        travelled = t * total_dist
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


def main():
    usd_path = sys.argv[1] if len(sys.argv) > 1 else os.path.join(
        _project_dir, "assets", "environments", "msoe_building",
        "MSOEBuildingV4_physics.usd"
    )
    if not os.path.exists(usd_path):
        print(f"ERROR: USD not found: {usd_path}", flush=True)
        sys.exit(1)
    print(f"[anim_demo] Building USD: {usd_path}", flush=True)

    assets_root = get_assets_root_path()
    if assets_root is None:
        print("[anim_demo] ERROR: get_assets_root_path() returned None", flush=True)
        sys.exit(2)
    character_usd = assets_root + CHARACTER_SUBPATH
    biped_fallback = assets_root + BIPED_FALLBACK_SUBPATH

    world = World(stage_units_in_meters=1.0)
    stage = omni.usd.get_context().get_stage()

    building_prim = stage.DefinePrim("/World/Building", "Xform")
    building_prim.GetReferences().AddReference(
        usd_path, primPath=Sdf.Path("/World/Building")
    )
    setup_physics_scene(stage)
    freeze_doors(stage)

    bbox_cache = UsdGeom.BBoxCache(Usd.TimeCode.Default(), [UsdGeom.Tokens.default_])
    bbox = bbox_cache.ComputeWorldBound(stage.GetPrimAtPath("/World/Building"))
    bb_min = bbox.ComputeAlignedRange().GetMin()
    add_ground_plane(stage, bb_min[2] - 0.5)

    add_interior_lighting(stage)
    camera = create_test_camera(CAM_KEY, CAM, stage=stage)

    # IMPORTANT: omni.anim.people expects characters under /World/Characters
    # Create the parent prim so the people system recognizes them
    prims_utils.create_prim("/World/Characters", "Xform")

    frames = build_frames_along_path(WAYPOINTS, WALK_SPEED_FT_PER_SEC, FPS)
    start_pos, start_dir = frames[0]

    person = load_character(
        stage, "/World/Characters/Person_01", character_usd,
        position=start_pos, face_dir_xy=start_dir,
    )
    if person is None:
        print("[anim_demo] Retrying with Biped_Setup fallback...", flush=True)
        person = load_character(
            stage, "/World/Characters/Person_01", biped_fallback,
            position=start_pos, face_dir_xy=start_dir,
        )
    if person is None:
        print("[anim_demo] FATAL: could not load character", flush=True)
        simulation_app.close()
        sys.exit(3)

    log_skeleton_info(stage, "/World/Characters/Person_01")

    world.reset()
    camera.initialize()
    set_focal_length_on_prim(stage, camera)

    # Longer warmup so AnimationGraph has time to initialize
    print(f"[anim_demo] Warmup: {WARMUP_FRAMES} frames", flush=True)
    for i in range(WARMUP_FRAMES):
        world.step(render=True)
        if i == 0 or i == WARMUP_FRAMES // 2 or i == WARMUP_FRAMES - 1:
            print(f"  warmup frame {i}", flush=True)

    save_screenshot(camera, SCREENSHOT_START)

    mid_frame = len(frames) // 2
    print(f"[anim_demo] Recording {len(frames)} frames to {VIDEO_NAME}.mp4", flush=True)
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
            if i == mid_frame:
                save_screenshot(camera, SCREENSHOT_MID)
            if i % 30 == 0:
                print(f"  Frame {i}/{len(frames)} pos=({pos[0]:.1f}, {pos[1]:.1f})", flush=True)

    save_screenshot(camera, SCREENSHOT_END)

    print("[anim_demo] Done!", flush=True)
    os._exit(0)


if __name__ == "__main__":
    main()
