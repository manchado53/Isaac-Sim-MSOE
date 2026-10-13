#!/usr/bin/env python3
"""
Generic Building Physics Demo — Orbital Camera Tour

Discovers floor surfaces dynamically, drops colorful objects on multiple
floors, lets them settle, then records a 360-degree orbital camera tour.

Usage:
    sbatch scripts/slurm/submit_isaac_job.sh simulations/environment/building_physics_demo.py \
        --input assets/environments/msoe_building/MSOEBuildingTake2_physics.usd

Output:
    - Video: videos/environment/<stem>_demo.mp4  (15s @ 30fps, 1920x1080)
    - Report: logs/<stem>_demo.txt
"""

import argparse
import os
import sys
import math

parser = argparse.ArgumentParser(description="Building physics demo with orbital camera tour.")
parser.add_argument("--input", required=True, help="Path to physics-ready USD file")
parser.add_argument("--video", default=None, help="Output video name (default: <stem>_demo)")
parser.add_argument("--report", default=None, help="Report log path (default: logs/<stem>_demo.txt)")
args, _unknown = parser.parse_known_args()

os.environ["OMNI_KIT_ACCEPT_EULA"] = "YES"

from isaacsim import SimulationApp

simulation_app = SimulationApp({"headless": True})

import numpy as np
import isaacsim.core.utils.numpy.rotations as rot_utils

import omni.usd
from pxr import Sdf, UsdLux, UsdGeom, Usd, UsdPhysics, Gf
from isaacsim.core.api import World
from isaacsim.core.api.objects import DynamicCuboid, DynamicSphere
from isaacsim.sensors.camera import Camera

_script_dir = os.path.dirname(os.path.abspath(__file__))
_project_dir = os.path.abspath(os.path.join(_script_dir, '..', '..'))
sys.path.insert(0, _project_dir)

from sim_utils.video_capture import VideoRecorder
from sim_utils.building_physics import (
    setup_physics_scene, discover_floor_levels,
)

# Resolve paths
input_usd = os.path.abspath(args.input)
stem = os.path.splitext(os.path.basename(input_usd))[0]

video_name = args.video if args.video else f"environment/{stem}_demo"

if args.report:
    report_path = os.path.abspath(args.report)
else:
    report_path = os.path.join(_project_dir, "logs", f"{stem}_demo.txt")

os.makedirs(os.path.dirname(report_path), exist_ok=True)

_report_file = None


def log(msg=""):
    print(msg, flush=True)
    if _report_file:
        _report_file.write(msg + "\n")
        _report_file.flush()


OBJ_SIZE = 0.3
FLOOR_COLORS = [
    ([1.0, 0.15, 0.15], [1.0, 0.5, 0.0]),     # red / orange
    ([0.2, 0.3, 1.0],   [0.4, 0.7, 1.0]),      # blue / light blue
    ([0.1, 0.8, 0.2],   [0.5, 1.0, 0.2]),      # green / lime
    ([1.0, 0.4, 0.7],   [0.9, 0.0, 0.6]),      # pink / magenta
    ([1.0, 0.9, 0.1],   [0.85, 0.65, 0.1]),    # yellow / gold
    ([0.0, 0.9, 0.9],   [0.0, 0.6, 0.6]),      # cyan / teal
]


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


def freeze_doors(stage):
    """Make all RigidBodyAPI prims kinematic so doors don't fly."""
    frozen = 0
    for prim in stage.Traverse():
        if prim.HasAPI(UsdPhysics.RigidBodyAPI):
            rb = UsdPhysics.RigidBodyAPI(prim)
            rb.CreateKinematicEnabledAttr(True)
            frozen += 1
    return frozen


def look_at_orientation(cam_pos, target):
    dx = target[0] - cam_pos[0]
    dy = target[1] - cam_pos[1]
    dz = target[2] - cam_pos[2]
    yaw = math.degrees(math.atan2(dy, dx))
    pitch = math.degrees(math.atan2(-dz, math.sqrt(dx**2 + dy**2)))
    return rot_utils.euler_angles_to_quats(
        np.array([0, pitch, yaw]), degrees=True
    )


def main():
    global _report_file
    _report_file = open(report_path, "w")

    log("=" * 60)
    log("Building Physics Demo — Orbital Camera Tour")
    log("=" * 60)
    log(f"Input:  {input_usd}")
    log(f"Video:  {video_name}")

    if not os.path.exists(input_usd):
        log(f"ERROR: {input_usd} not found.")
        log("Run setup_building_physics.py first.")
        _report_file.close()
        simulation_app.close()
        sys.exit(1)

    # --- Phase 1: Scene Setup ---
    log("\n--- Phase 1: Scene Setup ---")

    world = World(stage_units_in_meters=1.0)
    stage = omni.usd.get_context().get_stage()

    log(f"Loading: {input_usd}")
    building_prim = stage.DefinePrim("/World/Building", "Xform")
    building_prim.GetReferences().AddReference(
        input_usd, primPath=Sdf.Path("/World/Building")
    )
    log("Building loaded.")

    setup_physics_scene(stage)

    # Building bbox and ground plane
    building_prim = stage.GetPrimAtPath("/World/Building")
    bbox_cache = UsdGeom.BBoxCache(Usd.TimeCode.Default(), [UsdGeom.Tokens.default_])
    bbox = bbox_cache.ComputeWorldBound(building_prim)
    box_range = bbox.ComputeAlignedRange()
    bb_min = box_range.GetMin()
    bb_max = box_range.GetMax()

    bldg_center = np.array([(bb_min[i] + bb_max[i]) / 2.0 for i in range(3)])
    bldg_extent = np.array([bb_max[i] - bb_min[i] for i in range(3)])
    ground_z = bb_min[2] - 0.5

    log(f"Building: ({bb_min[0]:.1f}, {bb_min[1]:.1f}, {bb_min[2]:.1f}) to "
        f"({bb_max[0]:.1f}, {bb_max[1]:.1f}, {bb_max[2]:.1f})")
    log(f"Ground plane at Z={ground_z:.1f}")

    add_ground_plane(stage, ground_z)
    frozen = freeze_doors(stage)
    log(f"Froze {frozen} dynamic doors")

    # Lighting
    distant = UsdLux.DistantLight.Define(stage, Sdf.Path("/World/Sun"))
    distant.CreateIntensityAttr(3000)
    distant.CreateAngleAttr(1.0)
    xform = UsdGeom.Xformable(distant.GetPrim())
    xform.AddRotateXYZOp().Set(Gf.Vec3f(-45, 30, 0))
    UsdLux.DomeLight.Define(stage, Sdf.Path("/World/DomeLight")).CreateIntensityAttr(500)

    # Discover floors
    log("\nDiscovering floor levels...")
    floors = discover_floor_levels(stage)
    log(f"Found {len(floors)} floor levels:")
    for fl in floors:
        log(f"  {fl['name']:12s}  Z={fl['z']:.2f}  "
            f"center=({fl['center_x']:.1f}, {fl['center_y']:.1f})  "
            f"({fl['count']} elements)")

    if not floors:
        log("ERROR: No floor levels found!")
        _report_file.close()
        simulation_app.close()
        sys.exit(1)

    # Pick up to 6 evenly spaced floors
    if len(floors) > 6:
        step = len(floors) / 6.0
        demo_floors = [floors[int(i * step)] for i in range(6)]
    else:
        demo_floors = floors

    # --- Spawn objects ---
    log(f"\n{'=' * 60}")
    log(f"SPAWNING OBJECTS ON {len(demo_floors)} FLOORS")
    log("=" * 60)

    all_objects = []
    for i, fl in enumerate(demo_floors):
        cube_color, sphere_color = FLOOR_COLORS[i % len(FLOOR_COLORS)]
        drop_z = fl["z"] + 2.0
        offsets = [(-1.0, -1.0), (1.0, -1.0), (-1.0, 1.0), (1.0, 1.0)]

        for j in range(2):
            ox, oy = offsets[j]
            name = f"cube_{fl['name']}_{j}"
            cube = world.scene.add(DynamicCuboid(
                prim_path=f"/World/{name}", name=name,
                position=np.array([fl["center_x"] + ox, fl["center_y"] + oy, drop_z]),
                scale=np.array([OBJ_SIZE] * 3),
                color=np.array(cube_color),
            ))
            all_objects.append({
                "name": name, "obj": cube,
                "floor_z": fl["z"], "floor_name": fl["name"],
            })

        for j in range(2):
            ox, oy = offsets[j + 2]
            name = f"sphere_{fl['name']}_{j}"
            sphere = world.scene.add(DynamicSphere(
                prim_path=f"/World/{name}", name=name,
                position=np.array([fl["center_x"] + ox, fl["center_y"] + oy, drop_z]),
                radius=OBJ_SIZE / 2.0,
                color=np.array(sphere_color),
            ))
            all_objects.append({
                "name": name, "obj": sphere,
                "floor_z": fl["z"], "floor_name": fl["name"],
            })

        log(f"  {fl['name']:12s}  drop_z={drop_z:.1f}  (4 objects)")

    log(f"\nTotal objects: {len(all_objects)}")

    # Orbital camera setup
    horiz_diag = math.sqrt(float(bldg_extent[0])**2 + float(bldg_extent[1])**2)
    radius = horiz_diag * 1.8
    cam_height = float(bldg_center[2]) + float(bldg_extent[2]) * 0.5

    initial_pos = np.array([
        float(bldg_center[0]) + radius,
        float(bldg_center[1]),
        cam_height,
    ])
    initial_orient = look_at_orientation(initial_pos, bldg_center)

    camera = Camera(
        prim_path="/World/OrbitCamera",
        position=initial_pos,
        frequency=30,
        resolution=(1920, 1080),
        orientation=initial_orient,
    )
    log(f"  Orbit radius: {radius:.1f}, height: ~{cam_height:.1f}")

    # --- Phase 2: Settle ---
    log(f"\n--- Phase 2: Physics Settling ---")

    world.reset()
    camera.initialize()

    log("Warming up (30 frames)...")
    for _ in range(30):
        world.step(render=True)

    settle_frames = 300
    log(f"Settling ({settle_frames} frames)...")
    for i in range(settle_frames):
        world.step(render=True)
        if i % 60 == 0:
            log(f"  Frame {i}/{settle_frames}")

    # Validate
    log(f"\n{'=' * 60}")
    log("POST-SETTLING VALIDATION")
    log("=" * 60)

    all_pass = True
    for obj_info in all_objects:
        pos, _ = obj_info["obj"].get_world_pose()
        z = pos[2]
        passed = z > (obj_info["floor_z"] - 1.0)
        if not passed:
            all_pass = False
        log(f"  {obj_info['name']:28s}  Z={z:7.2f}  "
            f"floor={obj_info['floor_z']:.2f}  "
            f"[{'PASS' if passed else 'FAIL'}]")

    log(f"\nOVERALL: {'PASS' if all_pass else 'FAIL'}")

    # --- Phase 3: Orbital Recording ---
    log(f"\n--- Phase 3: Orbital Camera Recording ---")

    orbit_frames = 450
    log(f"Recording {orbit_frames} frames ({orbit_frames / 30:.0f}s at 30fps)...")

    with VideoRecorder(video_name, fps=30) as recorder:
        for i in range(orbit_frames):
            angle = 2 * math.pi * i / orbit_frames
            height_offset = float(bldg_extent[2]) * 0.15 * math.sin(2 * angle)
            cam_x = float(bldg_center[0]) + radius * math.cos(angle)
            cam_y = float(bldg_center[1]) + radius * math.sin(angle)
            cam_z = cam_height + height_offset

            cam_pos = np.array([cam_x, cam_y, cam_z])
            orient = look_at_orientation(cam_pos, bldg_center)
            camera.set_world_pose(position=cam_pos, orientation=orient)

            world.step(render=True)
            rgba = camera.get_rgba()
            if rgba is not None and i > 0:
                recorder.capture_rgba(rgba)

            if i % 90 == 0:
                log(f"  Frame {i}/{orbit_frames}  "
                    f"pos=({cam_x:.0f}, {cam_y:.0f}, {cam_z:.0f})")

    log("\nRecording complete.")
    log(f"  Video  -> videos/{video_name}.mp4")  # video_name includes environment/ prefix
    log(f"  Report -> {report_path}")
    log("=" * 60)

    _report_file.close()
    simulation_app.close()

    if not all_pass:
        sys.exit(1)


if __name__ == "__main__":
    main()
