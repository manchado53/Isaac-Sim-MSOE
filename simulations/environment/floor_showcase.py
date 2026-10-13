#!/usr/bin/env python3
"""
Innovation Center Floor Showcase — Roof Off, Balls Drop

Hides the roof of the Innovation Center to expose all floor levels, then
drops colorful balls onto every discovered floor. An orbital camera records
a 360-degree tour showing the balls resting on each floor.

Run on Rosie:
    sbatch scripts/slurm/submit_isaac_job.sh simulations/environment/floor_showcase.py

Output:
    videos/environment/floor_showcase.mp4  (20s @ 30fps, 1920x1080)
    logs/floor_showcase.txt
"""

import os
import sys
import math
import colorsys

os.environ["OMNI_KIT_ACCEPT_EULA"] = "YES"

from isaacsim import SimulationApp

simulation_app = SimulationApp({"headless": True})

import numpy as np
import isaacsim.core.utils.numpy.rotations as rot_utils

import omni.usd
from pxr import Sdf, UsdLux, UsdGeom, Usd, UsdPhysics, Gf
from isaacsim.core.api import World
from isaacsim.core.api.objects import DynamicSphere
from isaacsim.sensors.camera import Camera

_script_dir = os.path.dirname(os.path.abspath(__file__))
_project_dir = os.path.abspath(os.path.join(_script_dir, '..', '..'))
sys.path.insert(0, _project_dir)

from sim_utils.video_capture import VideoRecorder
from sim_utils.building_physics import (
    setup_physics_scene, iter_building_elements,
)

_report_path = os.path.join(_project_dir, "logs", "floor_showcase.txt")
os.makedirs(os.path.dirname(_report_path), exist_ok=True)
_report_file = None


def log(msg=""):
    print(msg, flush=True)
    if _report_file:
        _report_file.write(msg + "\n")
        _report_file.flush()


# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

BALL_RADIUS = 0.4
WARMUP_FRAMES = 30
SETTLE_FRAMES = 300
RECORD_FRAMES = 600
FPS = 30

# Triangle offsets for 3 balls per floor
BALL_OFFSETS = [(-1.5, 0.0), (0.75, 1.3), (0.75, -1.3)]


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def generate_floor_colors(n):
    """Generate n distinct colors using HSV rainbow."""
    colors = []
    for i in range(n):
        r, g, b = colorsys.hsv_to_rgb(i / n, 0.9, 1.0)
        colors.append([r, g, b])
    return colors


def discover_floor_levels(stage, building_path="/World/Building"):
    """Discover distinct floor surfaces by clustering floor element bboxes."""
    bbox_cache = UsdGeom.BBoxCache(
        Usd.TimeCode.Default(), [UsdGeom.Tokens.default_]
    )
    building_prim = stage.GetPrimAtPath(building_path)

    floor_data = []
    for _level, category_name, element in iter_building_elements(stage, building_prim):
        if category_name != "Floors":
            continue
        bbox = bbox_cache.ComputeWorldBound(element)
        box_range = bbox.ComputeAlignedRange()
        bb_min = box_range.GetMin()
        bb_max = box_range.GetMax()
        if bb_min == bb_max:
            continue
        dx = bb_max[0] - bb_min[0]
        dy = bb_max[1] - bb_min[1]
        if dx < 0.5 and dy < 0.5:
            continue
        floor_data.append({
            "z_top": bb_max[2],
            "x_center": (bb_min[0] + bb_max[0]) / 2.0,
            "y_center": (bb_min[1] + bb_max[1]) / 2.0,
            "area": dx * dy,
        })

    if not floor_data:
        return []

    floor_data.sort(key=lambda f: f["z_top"])
    clusters = []
    current = [floor_data[0]]
    for fd in floor_data[1:]:
        if fd["z_top"] - current[0]["z_top"] > 1.5:
            clusters.append(current)
            current = [fd]
        else:
            current.append(fd)
    clusters.append(current)

    levels = []
    for i, cluster in enumerate(clusters):
        total_area = sum(f["area"] for f in cluster)
        if total_area < 0.01:
            continue
        avg_x = sum(f["x_center"] * f["area"] for f in cluster) / total_area
        avg_y = sum(f["y_center"] * f["area"] for f in cluster) / total_area
        avg_z = sum(f["z_top"] * f["area"] for f in cluster) / total_area
        levels.append({
            "name": f"floor_{i + 1}",
            "z": avg_z,
            "center_x": avg_x,
            "center_y": avg_y,
            "count": len(cluster),
        })

    return levels


def add_ground_plane(stage, z):
    """Add a large collision plane at the given Z height."""
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


def hide_roofs(stage, building_prim):
    """Hide only roof elements so the camera can see inside from above."""
    hidden = 0
    for _level, category_name, element in iter_building_elements(stage, building_prim):
        if category_name == "Roofs":
            UsdGeom.Imageable(element).MakeInvisible()
            hidden += 1
    return hidden


def look_at_orientation(cam_pos, target):
    dx = target[0] - cam_pos[0]
    dy = target[1] - cam_pos[1]
    dz = target[2] - cam_pos[2]
    yaw = math.degrees(math.atan2(dy, dx))
    pitch = math.degrees(math.atan2(-dz, math.sqrt(dx**2 + dy**2)))
    return rot_utils.euler_angles_to_quats(
        np.array([0, pitch, yaw]), degrees=True
    )


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    global _report_file
    _report_file = open(_report_path, "w")

    log("=" * 65)
    log("Innovation Center Floor Showcase")
    log("  Roof off, balls drop onto every floor")
    log("=" * 65)

    # --- Phase 1: Scene Setup ---
    log("\n--- Phase 1: Scene Setup ---")

    world = World(stage_units_in_meters=1.0)
    stage = omni.usd.get_context().get_stage()

    physics_usd = os.path.join(
        _project_dir, "assets", "environments", "msoe_building",
        "innovation_center_physics.usd"
    )
    if not os.path.exists(physics_usd):
        log(f"ERROR: {physics_usd} not found.")
        log("Run setup_innovation_center_physics.py first.")
        _report_file.close()
        simulation_app.close()
        sys.exit(1)

    log(f"Loading: {physics_usd}")
    building_prim = stage.DefinePrim("/World/Building", "Xform")
    building_prim.GetReferences().AddReference(
        physics_usd, primPath=Sdf.Path("/World/Building")
    )
    log("Building loaded.")

    setup_physics_scene(stage)

    # Building bounding box
    building_prim = stage.GetPrimAtPath("/World/Building")
    bbox_cache = UsdGeom.BBoxCache(
        Usd.TimeCode.Default(), [UsdGeom.Tokens.default_]
    )
    bbox = bbox_cache.ComputeWorldBound(building_prim)
    box_range = bbox.ComputeAlignedRange()
    bb_min = box_range.GetMin()
    bb_max = box_range.GetMax()

    bldg_center = np.array([(bb_min[i] + bb_max[i]) / 2.0 for i in range(3)])
    bldg_extent = np.array([bb_max[i] - bb_min[i] for i in range(3)])
    ground_z = bb_min[2] - 0.5

    log(f"Building bbox: ({bb_min[0]:.1f}, {bb_min[1]:.1f}, {bb_min[2]:.1f}) to "
        f"({bb_max[0]:.1f}, {bb_max[1]:.1f}, {bb_max[2]:.1f})")
    log(f"Ground plane at Z={ground_z:.1f}")

    add_ground_plane(stage, ground_z)

    # Lighting
    distant = UsdLux.DistantLight.Define(stage, Sdf.Path("/World/Sun"))
    distant.CreateIntensityAttr(3000)
    distant.CreateAngleAttr(1.0)
    xform = UsdGeom.Xformable(distant.GetPrim())
    xform.AddRotateXYZOp().Set(Gf.Vec3f(-45, 30, 0))

    UsdLux.DomeLight.Define(stage, Sdf.Path("/World/DomeLight")).CreateIntensityAttr(500)

    # --- Phase 2: Hide Roofs ---
    log("\n--- Phase 2: Hide Roofs ---")
    hidden = hide_roofs(stage, building_prim)
    log(f"Hidden {hidden} roof elements")

    # --- Phase 3: Freeze Doors, Discover Floors, Spawn Balls ---
    log("\n--- Phase 3: Freeze Doors & Discover Floors ---")

    frozen = freeze_doors(stage)
    log(f"Froze {frozen} dynamic doors")

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

    # Generate rainbow colors for all floors
    colors = generate_floor_colors(len(floors))

    log(f"\n{'=' * 65}")
    log(f"SPAWNING BALLS ON ALL {len(floors)} FLOORS")
    log("=" * 65)

    all_balls = []
    for i, fl in enumerate(floors):
        drop_z = fl["z"] + 3.0
        color = colors[i]

        for j, (ox, oy) in enumerate(BALL_OFFSETS):
            name = f"ball_{fl['name']}_{j}"
            sphere = world.scene.add(DynamicSphere(
                prim_path=f"/World/{name}", name=name,
                position=np.array([fl["center_x"] + ox, fl["center_y"] + oy, drop_z]),
                radius=BALL_RADIUS,
                color=np.array(color),
            ))
            all_balls.append({
                "name": name, "obj": sphere,
                "floor_z": fl["z"], "floor_name": fl["name"],
            })

        log(f"  {fl['name']:12s}  drop_z={drop_z:.1f}  color=({color[0]:.2f}, {color[1]:.2f}, {color[2]:.2f})")

    log(f"\nTotal balls: {len(all_balls)}")

    # --- Phase 4: Camera Setup ---
    horiz_diag = math.sqrt(float(bldg_extent[0])**2 + float(bldg_extent[1])**2)
    radius = horiz_diag * 1.5
    cam_height = float(bldg_center[2]) + float(bldg_extent[2]) * 0.7

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
    log(f"\nCamera orbit radius: {radius:.1f}, height: ~{cam_height:.1f}")

    # --- Phase 5: Settle + Record ---
    log(f"\n--- Phase 5: Settle + Record ---")

    world.reset()
    camera.initialize()

    # Warmup
    log(f"Warming up ({WARMUP_FRAMES} frames)...")
    for _ in range(WARMUP_FRAMES):
        world.step(render=True)

    # Settle — let balls land
    log(f"Settling ({SETTLE_FRAMES} frames)...")
    for i in range(SETTLE_FRAMES):
        world.step(render=True)
        if i % 60 == 0:
            log(f"  Frame {i}/{SETTLE_FRAMES}")

    # Validate
    log(f"\n{'=' * 65}")
    log("POST-SETTLE VALIDATION")
    log("=" * 65)

    all_pass = True
    for ball in all_balls:
        pos, _ = ball["obj"].get_world_pose()
        z = pos[2]
        passed = z > (ball["floor_z"] - 1.0)
        if not passed:
            all_pass = False
        log(f"  {ball['name']:28s}  Z={z:7.2f}  "
            f"floor={ball['floor_z']:.2f}  "
            f"[{'PASS' if passed else 'FAIL'}]")

    log(f"\nFLOOR PHYSICS: {'PASS' if all_pass else 'FAIL'}")

    # Record orbital tour
    log(f"\n--- Recording Orbital Tour ---")
    log(f"  {RECORD_FRAMES} frames ({RECORD_FRAMES / FPS:.0f}s at {FPS}fps)")

    with VideoRecorder("environment/floor_showcase", fps=FPS) as recorder:
        for i in range(RECORD_FRAMES):
            angle = 2 * math.pi * i / RECORD_FRAMES
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

            if i % 100 == 0:
                log(f"  Frame {i}/{RECORD_FRAMES}  "
                    f"cam=({cam_x:.0f}, {cam_y:.0f}, {cam_z:.0f})")

    log(f"\n{'=' * 65}")
    log("Output:")
    log(f"  Video  -> videos/environment/floor_showcase.mp4")
    log(f"  Report -> {_report_path}")
    log("=" * 65)

    _report_file.close()
    simulation_app.close()


if __name__ == "__main__":
    main()
