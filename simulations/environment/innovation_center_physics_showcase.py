#!/usr/bin/env python3
"""
Innovation Center Physics Showcase — Exposed Floors + Door Animation

Loads the physics-ready building USD, hides all walls and ceilings to expose
the floor plates, drops colorful objects on multiple floors, and animates
doors swinging open and closed — all while an orbital camera records a 360°
tour of the building skeleton.

Run on Rosie:
    sbatch submit_isaac_job.sh simulations/environment/innovation_center_physics_showcase.py

Output:
    videos/environment/innovation_center_physics_showcase.mp4  (20s @ 30fps, 1920x1080)
    logs/innovation_center_physics_showcase.txt
"""

import os
import sys
import math

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
    setup_physics_scene, iter_building_elements, set_drive_target,
)

_report_path = os.path.join(_project_dir, "logs", "innovation_center_physics_showcase.txt")
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

OBJ_SIZE = 0.3  # meters
FLOOR_COLORS = [
    ([1.0, 0.15, 0.15], [1.0, 0.5, 0.0]),     # red / orange
    ([0.2, 0.3, 1.0],   [0.4, 0.7, 1.0]),      # blue / light blue
    ([0.1, 0.8, 0.2],   [0.5, 1.0, 0.2]),      # green / lime
    ([1.0, 0.4, 0.7],   [0.9, 0.0, 0.6]),      # pink / magenta
    ([1.0, 0.9, 0.1],   [0.85, 0.65, 0.1]),    # yellow / gold
    ([0.0, 0.9, 0.9],   [0.0, 0.6, 0.6]),      # cyan / teal
]

# Categories to hide (walls, ceilings, roofs, curtain wall elements, windows)
HIDE_CATEGORIES = {
    "Walls", "Wall_Sweeps",
    "Ceilings", "Roofs",
    "CurtainPanels", "Curtain_Panels",
    "CurtainWallMullions", "Curtain_Wall_Mullions",
    "Windows",
}

# Warmup + recording frame counts
WARMUP_FRAMES = 30
RECORD_FRAMES = 600
TOTAL_FRAMES = WARMUP_FRAMES + RECORD_FRAMES
FPS = 30

# Door animation keyframes (relative to recording start, i.e. after warmup)
DOOR_OPEN_START = 150   # frame when doors start opening (5s in)
DOOR_OPEN_END = 300     # fully open (10s in)
DOOR_HOLD_END = 420     # hold open until (14s in)
DOOR_CLOSE_END = 540    # fully closed (18s in)
DOOR_TARGET_ANGLE = 80.0  # degrees


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

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


def hide_walls_and_ceilings(stage, building_prim):
    """Hide all wall, ceiling, roof, curtain wall, and window elements."""
    hidden = 0
    for _level, category_name, element in iter_building_elements(stage, building_prim):
        if category_name is not None and category_name in HIDE_CATEGORIES:
            UsdGeom.Imageable(element).MakeInvisible()
            hidden += 1
    return hidden


def find_driven_doors(stage):
    """Find all revolute joints with angular DriveAPI on Door prims."""
    doors = []
    for prim in stage.Traverse():
        if not prim.IsA(UsdPhysics.RevoluteJoint):
            continue
        path_str = str(prim.GetPath())
        if "Door" not in path_str:
            continue
        drive = UsdPhysics.DriveAPI.Get(prim, "angular")
        if not drive or not drive.GetStiffnessAttr().HasValue():
            continue
        doors.append((path_str, drive))
    return doors


def compute_door_target(record_frame):
    """Compute door drive target angle for a given recording frame."""
    if record_frame < DOOR_OPEN_START:
        return 0.0
    elif record_frame < DOOR_OPEN_END:
        t = (record_frame - DOOR_OPEN_START) / float(DOOR_OPEN_END - DOOR_OPEN_START)
        return t * DOOR_TARGET_ANGLE
    elif record_frame < DOOR_HOLD_END:
        return DOOR_TARGET_ANGLE
    elif record_frame < DOOR_CLOSE_END:
        t = (record_frame - DOOR_HOLD_END) / float(DOOR_CLOSE_END - DOOR_HOLD_END)
        return DOOR_TARGET_ANGLE * (1.0 - t)
    else:
        return 0.0


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
    log("Innovation Center Physics Showcase")
    log("  Exposed floors + door animation + orbital camera")
    log("=" * 65)

    # --- Phase 1: Scene Setup ---
    log("\n--- Phase 1: Scene Setup ---")

    world = World(stage_units_in_meters=1.0)
    stage = omni.usd.get_context().get_stage()

    physics_usd = os.path.join(
        _project_dir, "assets", "environments", "msoe_building",
        "innovation_center_finalV1.usd"
    )
    if not os.path.exists(physics_usd):
        log(f"ERROR: {physics_usd} not found.")
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

    # --- Phase 2: Hide Walls & Ceilings ---
    log("\n--- Phase 2: Hide Walls & Ceilings ---")
    hidden = hide_walls_and_ceilings(stage, building_prim)
    log(f"Hidden {hidden} wall/ceiling/roof/window elements")

    # --- Phase 3: Discover Floors & Spawn Objects ---
    log("\n--- Phase 3: Discover Floors & Spawn Objects ---")
    floors = discover_floor_levels(stage)
    log(f"Found {len(floors)} floor levels:")
    for fl in floors:
        log(f"  {fl['name']:12s}  Z={fl['z']:.2f}m  "
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

    log(f"Total objects: {len(all_objects)}")

    # --- Phase 4: Discover Doors ---
    log("\n--- Phase 4: Discover Driven Doors ---")
    doors = find_driven_doors(stage)
    log(f"Found {len(doors)} doors with DriveAPI")

    # --- Camera Setup ---
    horiz_diag = math.sqrt(float(bldg_extent[0])**2 + float(bldg_extent[1])**2)
    radius = horiz_diag * 1.3
    cam_height = float(bldg_center[2]) + float(bldg_extent[2]) * 0.3

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
    log(f"Camera orbit radius: {radius:.1f}m, height: ~{cam_height:.1f}m")

    # --- Phase 5: Simulate & Record ---
    log(f"\n--- Phase 5: Simulate & Record ---")
    log(f"  Warmup: {WARMUP_FRAMES} frames")
    log(f"  Record: {RECORD_FRAMES} frames ({RECORD_FRAMES / FPS:.0f}s at {FPS}fps)")

    world.reset()
    camera.initialize()

    # Warmup (no recording)
    log("Warming up...")
    for _ in range(WARMUP_FRAMES):
        world.step(render=True)

    # Record
    log("Recording...")
    with VideoRecorder("environment/innovation_center_physics_showcase", fps=FPS) as recorder:
        for i in range(RECORD_FRAMES):
            # Door animation
            target_angle = compute_door_target(i)
            for _path, drive in doors:
                set_drive_target(drive, target_angle)

            # Orbital camera
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
                    f"door_target={target_angle:.0f}°  "
                    f"cam=({cam_x:.0f}, {cam_y:.0f}, {cam_z:.0f})")

    # --- Validation ---
    log(f"\n{'=' * 65}")
    log("POST-SETTLE VALIDATION")
    log("=" * 65)

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

    log(f"\nFLOOR PHYSICS: {'PASS' if all_pass else 'FAIL'}")
    log(f"DOORS ANIMATED: {len(doors)} doors driven to {DOOR_TARGET_ANGLE}° and back")

    log(f"\n{'=' * 65}")
    log("Output:")
    log(f"  Video  -> videos/environment/innovation_center_physics_showcase.mp4")
    log(f"  Report -> {_report_path}")
    log("=" * 65)

    _report_file.close()
    simulation_app.close()


if __name__ == "__main__":
    main()
