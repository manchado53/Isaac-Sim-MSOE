#!/usr/bin/env python3
"""
Quick demo: colorful balls dropping on the Innovation Center ground floor.

Produces: videos/validation/ball_drop_ground_floor.mp4
"""

import os
import sys
import math

os.environ["OMNI_KIT_ACCEPT_EULA"] = "YES"

from isaacsim import SimulationApp
simulation_app = SimulationApp({"headless": True})

import numpy as np
import omni.usd
from pxr import Usd, UsdGeom, UsdPhysics, Sdf, Gf, UsdLux, UsdShade

_script_dir = os.path.dirname(os.path.abspath(__file__))
_project_dir = os.path.abspath(os.path.join(_script_dir, '..', '..'))
sys.path.insert(0, _project_dir)

from isaacsim.core.api import World
from isaacsim.core.api.objects import DynamicSphere
from isaacsim.sensors.camera import Camera
from sim_utils.video_capture import VideoRecorder
from sim_utils.building_physics import setup_physics_scene
from sim_utils.test_cameras import look_at_orientation

# Ground floor camera "G": Event Floor IC 100
# pos: (-42.15, 68.44, 16.44), target: (-27.42, 83.72, 14.56)
FLOOR_Z = 14.56  # ground floor height in feet
DROP_CENTER = (-33.0, 77.0)  # midpoint between cam and target on XY
DROP_HEIGHT = 6.0  # feet above floor

# Ball config
BALL_RADIUS = 0.4  # feet (~5 inch balls)
BALL_COLORS = [
    (1.0, 0.15, 0.15),   # red
    (0.2, 0.3, 1.0),     # blue
    (0.1, 0.8, 0.2),     # green
    (1.0, 0.9, 0.1),     # yellow
    (1.0, 0.4, 0.7),     # pink
    (0.0, 0.9, 0.9),     # cyan
]

RECORD_FRAMES = 360  # 12s at 30fps


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


def main():
    usd_path = sys.argv[1] if len(sys.argv) > 1 else os.path.join(
        _project_dir, "assets", "environments", "msoe_building",
        "innovation_center_physics.usd"
    )

    if not os.path.exists(usd_path):
        print(f"ERROR: USD not found: {usd_path}", flush=True)
        sys.exit(1)

    print(f"[ball_drop] Loading: {usd_path}", flush=True)

    world = World(stage_units_in_meters=1.0)
    stage = omni.usd.get_context().get_stage()

    building_prim = stage.DefinePrim("/World/Building", "Xform")
    building_prim.GetReferences().AddReference(
        usd_path, primPath=Sdf.Path("/World/Building")
    )

    setup_physics_scene(stage)
    freeze_doors(stage)

    # Ground plane
    building_prim = stage.GetPrimAtPath("/World/Building")
    bbox_cache = UsdGeom.BBoxCache(Usd.TimeCode.Default(), [UsdGeom.Tokens.default_])
    bbox = bbox_cache.ComputeWorldBound(building_prim)
    bb_min = bbox.ComputeAlignedRange().GetMin()
    add_ground_plane(stage, bb_min[2] - 0.5)

    # --- Soft interior lighting ---
    distant = UsdLux.DistantLight.Define(stage, Sdf.Path("/World/Sun"))
    distant.CreateIntensityAttr(2000)
    distant.CreateAngleAttr(1.0)
    UsdGeom.Xformable(distant.GetPrim()).AddRotateXYZOp().Set((-45, 30, 0))

    UsdLux.DomeLight.Define(stage, Sdf.Path("/World/Dome")).CreateIntensityAttr(1000)

    # Fill light near drop zone
    fill = UsdLux.SphereLight.Define(stage, Sdf.Path("/World/FillLight"))
    fill.CreateIntensityAttr(12000)
    fill.CreateRadiusAttr(2.0)
    UsdGeom.Xformable(fill.GetPrim()).AddTranslateOp().Set(
        (DROP_CENTER[0], DROP_CENTER[1], FLOOR_Z + 5.0)
    )

    # --- Camera: low angle looking at the drop zone ---
    # Position behind and slightly to the side, at ~3ft height
    cam_pos = np.array([DROP_CENTER[0] - 8.0, DROP_CENTER[1] - 6.0, FLOOR_Z + 3.0])
    cam_target = np.array([DROP_CENTER[0], DROP_CENTER[1], FLOOR_Z + 2.0])
    cam_orient = look_at_orientation(cam_pos, cam_target)

    camera = Camera(
        prim_path="/World/BallDropCam",
        position=cam_pos,
        frequency=30,
        resolution=(1920, 1080),
        orientation=cam_orient,
    )
    # Wide angle to capture the full drop zone
    cam_prim = stage.GetPrimAtPath("/World/BallDropCam")
    UsdGeom.Camera(cam_prim).CreateFocalLengthAttr(14.0)

    # --- Spawn balls in a loose cluster ---
    print("[ball_drop] Spawning 6 balls...", flush=True)
    cx, cy = DROP_CENTER
    offsets = [
        (-1.0, -1.0), (1.0, -1.0), (0.0, 0.5),
        (-1.2, 1.2), (1.2, 1.0), (0.0, -0.3),
    ]
    for i, (ox, oy) in enumerate(offsets):
        # Stagger drop heights so they don't all land at once
        h = DROP_HEIGHT + i * 0.8
        DynamicSphere(
            prim_path=f"/World/Ball_{i}",
            name=f"ball_{i}",
            position=np.array([cx + ox, cy + oy, FLOOR_Z + h]),
            radius=BALL_RADIUS,
            color=np.array(BALL_COLORS[i]),
            mass=0.5,
        )
        print(f"  Ball {i}: ({cx+ox:.1f}, {cy+oy:.1f}, {FLOOR_Z+h:.1f})", flush=True)

    # --- Simulate and record ---
    world.reset()
    camera.initialize()

    # Warmup
    for _ in range(30):
        world.step(render=True)

    print(f"[ball_drop] Recording {RECORD_FRAMES} frames...", flush=True)
    recorder = VideoRecorder("validation/ball_drop_ground_floor", fps=30)
    recorder.__enter__()

    for i in range(RECORD_FRAMES):
        world.step(render=True)
        if i > 0:
            try:
                rgba = camera.get_rgba()
                if rgba is not None:
                    recorder.capture_rgba(rgba)
            except Exception:
                pass
        if i % 60 == 0:
            print(f"  Frame {i}/{RECORD_FRAMES}", flush=True)

    try:
        recorder.__exit__(None, None, None)
    except Exception:
        pass

    print("[ball_drop] Done!", flush=True)
    os._exit(0)


if __name__ == "__main__":
    main()
