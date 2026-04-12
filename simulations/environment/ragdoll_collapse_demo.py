#!/usr/bin/env python3
"""
Ragdoll Collapse Demo — physics-driven capsule ragdoll falls to the floor.

Builds an 11-capsule humanoid with spherical + revolute joints, lets it
stand kinematically for 1 second, then flips to dynamic so PhysX gravity
pulls it to the floor. Records the collapse from camera G.

This is the FIRST real physics ragdoll in the project — prior "fallen
person" demos all used keyframed UsdSkel animations. No mesh is rendered;
the capsules are the visual.

Output:
    videos/environment/ragdoll_collapse_demo.mp4
    images/ragdoll_collapse_final.png
    images/ragdoll_standing.png   (captured right before the flip)
"""

import os
import sys

os.environ["OMNI_KIT_ACCEPT_EULA"] = "YES"

from isaacsim import SimulationApp
simulation_app = SimulationApp({"headless": True})

import numpy as np
import omni.usd
from pxr import Usd, UsdGeom, UsdPhysics, Sdf, Gf

_script_dir = os.path.dirname(os.path.abspath(__file__))
_project_dir = os.path.abspath(os.path.join(_script_dir, '..', '..'))
sys.path.insert(0, _project_dir)

from isaacsim.core.api import World

from sim_utils.test_cameras import (
    TEST_CAMERAS,
    create_test_camera,
    add_interior_lighting,
    set_focal_length_on_prim,
)
from sim_utils.video_capture import VideoRecorder
from sim_utils.building_physics import setup_physics_scene
from sim_utils.ragdoll import build_character_ragdoll, set_ragdoll_kinematic


# --- Camera + placement ---
CAM_KEY = "G"
CAM = TEST_CAMERAS[CAM_KEY]
FLOOR_Z = 14.56          # camera target Z (reference)
REAL_FLOOR_Z = 13.56     # walkable surface Z (from raycast)

# Character root position (feet-origin). Same XY as other demos.
PERSON_X = -37.0
PERSON_Y = 74.0
RAGDOLL_ROOT_Z = REAL_FLOOR_Z  # bottom of lowest capsule sits at floor

# --- Timing (30 fps) ---
FPS = 30
WARMUP_FRAMES = 20        # let physics settle in kinematic mode
STANDING_FRAMES = 30      # 1.0 s standing (kinematic)
COLLAPSE_FRAMES = 150     # 5.0 s falling + settling
TOTAL_RECORD_FRAMES = STANDING_FRAMES + COLLAPSE_FRAMES

# --- Output paths ---
VIDEO_NAME = "environment/ragdoll_collapse_demo"
SCREENSHOT_STANDING = os.path.join(_project_dir, "images", "ragdoll_standing.png")
SCREENSHOT_FINAL = os.path.join(_project_dir, "images", "ragdoll_collapse_final.png")


def add_ground_plane(stage, z):
    """Add a large static collision ground plane at Z=z."""
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


def freeze_building_doors(stage):
    """Freeze building rigid bodies so doors don't swing open."""
    for prim in stage.Traverse():
        if prim.HasAPI(UsdPhysics.RigidBodyAPI):
            path = str(prim.GetPath())
            if path.startswith("/World/Ragdoll"):
                continue  # leave ragdoll bodies alone
            UsdPhysics.RigidBodyAPI(prim).CreateKinematicEnabledAttr(True)


def save_screenshot(camera, output_path):
    rgba = camera.get_rgba()
    if rgba is None:
        return False
    try:
        from PIL import Image
        os.makedirs(os.path.dirname(output_path), exist_ok=True)
        Image.fromarray(rgba[:, :, :3]).save(output_path)
        print(f"[ragdoll] Saved: {output_path}", flush=True)
        return True
    except Exception as e:
        print(f"[ragdoll] Screenshot error: {e}", flush=True)
        return False


def dump_ragdoll_bbox(stage, bodies):
    """Print the world-space bounding box of the ragdoll (all capsules)."""
    try:
        cache = UsdGeom.BBoxCache(Usd.TimeCode.Default(),
                                   [UsdGeom.Tokens.default_])
        zs_min = []
        zs_max = []
        for name, prim in bodies.items():
            bb = cache.ComputeWorldBound(prim).ComputeAlignedRange()
            zs_min.append(bb.GetMin()[2])
            zs_max.append(bb.GetMax()[2])
        print(f"[ragdoll] bbox Z range: "
              f"min={min(zs_min):.2f} max={max(zs_max):.2f}", flush=True)
    except Exception as e:
        print(f"[ragdoll] bbox diag failed: {e}", flush=True)


def main():
    usd_path = sys.argv[1] if len(sys.argv) > 1 else os.path.join(
        _project_dir, "assets", "environments", "msoe_building",
        "MSOEBuildingV4_physics.usd"
    )
    if not os.path.exists(usd_path):
        print(f"ERROR: USD not found: {usd_path}", flush=True)
        sys.exit(1)
    print(f"[ragdoll] Building USD: {usd_path}", flush=True)

    # --- Scene setup ---
    world = World(stage_units_in_meters=1.0)
    stage = omni.usd.get_context().get_stage()

    building_prim = stage.DefinePrim("/World/Building", "Xform")
    building_prim.GetReferences().AddReference(
        usd_path, primPath=Sdf.Path("/World/Building")
    )
    setup_physics_scene(stage)
    freeze_building_doors(stage)

    bbox_cache = UsdGeom.BBoxCache(Usd.TimeCode.Default(),
                                     [UsdGeom.Tokens.default_])
    bb = bbox_cache.ComputeWorldBound(stage.GetPrimAtPath("/World/Building"))
    add_ground_plane(stage, bb.ComputeAlignedRange().GetMin()[2] - 0.5)

    add_interior_lighting(stage)
    camera = create_test_camera(CAM_KEY, CAM, stage=stage)

    # --- Build ragdoll (kinematic = standing still) ---
    print(f"[ragdoll] Building ragdoll at ({PERSON_X}, {PERSON_Y}, "
          f"{RAGDOLL_ROOT_Z})", flush=True)
    bodies = build_character_ragdoll(
        stage,
        root_prim_path="/World/Ragdoll",
        world_position=(PERSON_X, PERSON_Y, RAGDOLL_ROOT_Z),
        kinematic=True,
    )
    print(f"[ragdoll] Built {len(bodies)} bodies", flush=True)

    # --- Reset + initialize ---
    world.reset()
    camera.initialize()
    set_focal_length_on_prim(stage, camera)

    # --- Warmup (kinematic bodies just hold their initial pose) ---
    print(f"[ragdoll] Warmup: {WARMUP_FRAMES} frames", flush=True)
    for _ in range(WARMUP_FRAMES):
        world.step(render=True)
    dump_ragdoll_bbox(stage, bodies)

    # --- Record ---
    print(f"[ragdoll] Recording {TOTAL_RECORD_FRAMES} frames "
          f"(stand:{STANDING_FRAMES}, collapse:{COLLAPSE_FRAMES})",
          flush=True)
    with VideoRecorder(VIDEO_NAME, fps=FPS) as recorder:
        for i in range(TOTAL_RECORD_FRAMES):
            # At frame STANDING_FRAMES, flip all bodies to dynamic
            if i == STANDING_FRAMES:
                print(f"[ragdoll] >>> COLLAPSE at frame {i}", flush=True)
                save_screenshot(camera, SCREENSHOT_STANDING)
                set_ragdoll_kinematic(bodies, False)

            world.step(render=True)
            try:
                rgba = camera.get_rgba()
                if rgba is not None:
                    recorder.capture_rgba(rgba)
            except Exception:
                pass

            if i % 30 == 0 or i == STANDING_FRAMES:
                dump_ragdoll_bbox(stage, bodies)

    # Final screenshot
    save_screenshot(camera, SCREENSHOT_FINAL)
    dump_ragdoll_bbox(stage, bodies)
    print("[ragdoll] Done!", flush=True)
    os._exit(0)


if __name__ == "__main__":
    main()
