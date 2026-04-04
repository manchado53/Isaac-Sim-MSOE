#!/usr/bin/env python3
"""
MSOE Building USD Inspection & Video Tour

Loads the TestBuilding.usd (Revit export of an MSOE building), inspects the
scene hierarchy, computes the bounding box, and records an orbital camera tour.

Output:
  - logs/msoe_building_report.txt -> prim hierarchy, type summary, bounding box
  - videos/environment/msoe_building_tour.mp4 -> orbital tour video
"""

import os
import sys

os.environ["OMNI_KIT_ACCEPT_EULA"] = "YES"

from isaacsim import SimulationApp

simulation_app = SimulationApp({"headless": True})

import math
import traceback
import numpy as np
import isaacsim.core.utils.numpy.rotations as rot_utils

import omni.usd
from pxr import Sdf, Usd, UsdGeom, UsdLux, Gf
from isaacsim.core.api import World
from isaacsim.core.utils.stage import add_reference_to_stage
from isaacsim.sensors.camera import Camera

# Add project root to path for sim_utils import
_script_dir = os.path.dirname(os.path.abspath(__file__))
_project_dir = os.path.abspath(os.path.join(_script_dir, '..', '..'))
sys.path.insert(0, _project_dir)
from sim_utils.video_capture import VideoRecorder

# Report file for reliable output (Isaac Sim can swallow stdout)
_report_path = os.path.join(_project_dir, "logs", "msoe_building_report.txt")
os.makedirs(os.path.dirname(_report_path), exist_ok=True)
_report_file = None


def log(msg=""):
    """Print to both stdout (flushed) and the report file."""
    print(msg, flush=True)
    if _report_file:
        _report_file.write(msg + "\n")
        _report_file.flush()


# ---------------------------------------------------------------------------
# Phase 1: Load USD & Inspect Hierarchy
# ---------------------------------------------------------------------------

def print_prim_tree(prim, depth=0, max_depth=3):
    """Recursively print prim hierarchy up to max_depth."""
    if depth > max_depth:
        return
    indent = "  " * depth
    type_name = prim.GetTypeName() or "(untyped)"
    log(f"{indent}{prim.GetName()} [{type_name}]")
    children = prim.GetChildren()
    # Cap output at depth 1 to avoid flooding (building has thousands of top-level children)
    if depth == 0 and len(children) > 50:
        for child in children[:50]:
            print_prim_tree(child, depth + 1, max_depth)
        log(f"  ... and {len(children) - 50} more children (truncated)")
    else:
        for child in children:
            print_prim_tree(child, depth + 1, max_depth)


def inspect_hierarchy(stage, root_path="/World/Building"):
    """Inspect and summarize the USD prim hierarchy."""
    root = stage.GetPrimAtPath(root_path)
    if not root.IsValid():
        log(f"ERROR: No prim at {root_path}")
        return

    log("\n" + "=" * 60)
    log("PRIM HIERARCHY (max depth 3, first 50 children)")
    log("=" * 60)
    print_prim_tree(root, max_depth=3)

    # Count prims by type
    log("\nCounting prim types...")
    type_counts = {}
    total = 0
    for prim in Usd.PrimRange(root):
        total += 1
        t = prim.GetTypeName() or "(untyped)"
        type_counts[t] = type_counts.get(t, 0) + 1

    log(f"\n{'=' * 60}")
    log(f"PRIM TYPE SUMMARY  (total: {total})")
    log("=" * 60)
    for t, count in sorted(type_counts.items(), key=lambda x: -x[1]):
        log(f"  {t:30s} {count}")

    # Search for floor-related prims
    floor_keywords = ["floor", "ground", "level", "f1", "first", "base", "story"]
    floor_prims = []
    for prim in Usd.PrimRange(root):
        name_lower = prim.GetName().lower()
        if any(kw in name_lower for kw in floor_keywords):
            floor_prims.append(prim)

    log(f"\n{'=' * 60}")
    log(f"FLOOR-RELATED PRIMS  (found: {len(floor_prims)})")
    log("=" * 60)
    for p in floor_prims[:30]:
        log(f"  {p.GetPath()} [{p.GetTypeName()}]")
    if len(floor_prims) > 30:
        log(f"  ... and {len(floor_prims) - 30} more")


# ---------------------------------------------------------------------------
# Phase 2: Compute Bounding Box
# ---------------------------------------------------------------------------

def compute_bounding_box(stage, root_path="/World/Building"):
    """Compute world-space bounding box of the building."""
    root = stage.GetPrimAtPath(root_path)

    log(f"\n{'=' * 60}")
    log("BOUNDING BOX")
    log("=" * 60)

    bbox_cache = UsdGeom.BBoxCache(Usd.TimeCode.Default(), [UsdGeom.Tokens.default_])
    bbox = bbox_cache.ComputeWorldBound(root)
    box_range = bbox.ComputeAlignedRange()
    bb_min = box_range.GetMin()
    bb_max = box_range.GetMax()

    if bb_min == bb_max:
        # Degenerate result — fall back to scanning individual meshes
        log("  BBoxCache returned degenerate result, falling back to mesh scan...")
        all_min = Gf.Vec3d(float('inf'), float('inf'), float('inf'))
        all_max = Gf.Vec3d(float('-inf'), float('-inf'), float('-inf'))
        mesh_count = 0
        for prim in Usd.PrimRange(root):
            if prim.IsA(UsdGeom.Mesh):
                mesh_bbox = bbox_cache.ComputeWorldBound(prim)
                mesh_range = mesh_bbox.ComputeAlignedRange()
                m_min = mesh_range.GetMin()
                m_max = mesh_range.GetMax()
                if m_min != m_max:
                    all_min = Gf.Vec3d(min(all_min[0], m_min[0]),
                                       min(all_min[1], m_min[1]),
                                       min(all_min[2], m_min[2]))
                    all_max = Gf.Vec3d(max(all_max[0], m_max[0]),
                                       max(all_max[1], m_max[1]),
                                       max(all_max[2], m_max[2]))
                    mesh_count += 1
        if mesh_count > 0:
            bb_min, bb_max = all_min, all_max
            log(f"  Computed from {mesh_count} meshes")
        else:
            log("  WARNING: No valid mesh bounds found. Using defaults.")
            bb_min = Gf.Vec3d(-10, -10, 0)
            bb_max = Gf.Vec3d(10, 10, 10)

    center = (bb_min + bb_max) / 2.0
    extent = bb_max - bb_min

    log(f"  Min:    ({bb_min[0]:.2f}, {bb_min[1]:.2f}, {bb_min[2]:.2f})")
    log(f"  Max:    ({bb_max[0]:.2f}, {bb_max[1]:.2f}, {bb_max[2]:.2f})")
    log(f"  Center: ({center[0]:.2f}, {center[1]:.2f}, {center[2]:.2f})")
    log(f"  Extent: ({extent[0]:.2f}, {extent[1]:.2f}, {extent[2]:.2f})")

    return (
        np.array([bb_min[0], bb_min[1], bb_min[2]]),
        np.array([bb_max[0], bb_max[1], bb_max[2]]),
    )


# ---------------------------------------------------------------------------
# Phase 3: Orbital Camera Tour
# ---------------------------------------------------------------------------

def orbital_tour(world, bb_min, bb_max):
    """Record a 360-degree orbital tour around the building."""
    center = (bb_min + bb_max) / 2.0
    extent = bb_max - bb_min
    horiz_diag = math.sqrt(extent[0] ** 2 + extent[1] ** 2)
    radius = max(horiz_diag * 2.0, 5.0)
    cam_height_base = center[2] + extent[2] * 0.6

    log(f"\n{'=' * 60}")
    log("ORBITAL CAMERA TOUR")
    log("=" * 60)
    log(f"  Center: ({center[0]:.2f}, {center[1]:.2f}, {center[2]:.2f})")
    log(f"  Radius: {radius:.2f}")
    log(f"  Camera height: ~{cam_height_base:.2f}")

    # Add a ground plane for visual reference
    world.scene.add_default_ground_plane()

    # Add a debug marker at building center (bright red cube)
    from isaacsim.core.api.objects import DynamicCuboid
    world.scene.add(
        DynamicCuboid(
            prim_path="/World/DebugMarker",
            name="debug_marker",
            position=np.array([center[0], center[1], center[2]]),
            scale=np.array([1.0, 1.0, 1.0]),
            color=np.array([1.0, 0.0, 0.0]),
        )
    )

    # Create camera — euler convention matches jetbot/franka demos:
    # [0, pitch, yaw] where yaw = atan2(dy, dx) from camera toward target
    initial_yaw = math.degrees(math.atan2(
        center[1] - (center[1]),
        center[0] - (center[0] + radius)
    ))  # = 180 degrees
    initial_pitch = 20.0
    initial_pos = np.array([center[0] + radius, center[1], cam_height_base])
    initial_orient = rot_utils.euler_angles_to_quats(
        np.array([0, initial_pitch, initial_yaw]), degrees=True
    )

    camera = Camera(
        prim_path="/World/TourCamera",
        position=initial_pos,
        frequency=30,
        resolution=(1280, 720),
        orientation=initial_orient,
    )

    world.reset()
    camera.initialize()

    # Warm up rendering (more frames to ensure pipeline is ready)
    for _ in range(30):
        world.step(render=True)

    orbit_steps = 360
    log(f"  Recording {orbit_steps} frames...")

    with VideoRecorder("environment/msoe_building_tour", fps=15) as recorder:
        for i in range(orbit_steps):
            angle = 2 * math.pi * i / orbit_steps

            # Slight vertical oscillation
            height_offset = extent[2] * 0.1 * math.sin(2 * angle)
            cam_x = center[0] + radius * math.cos(angle)
            cam_y = center[1] + radius * math.sin(angle)
            cam_z = cam_height_base + height_offset

            cam_pos = np.array([cam_x, cam_y, cam_z])

            # Yaw: atan2(dy, dx) from camera toward center
            # This matches the jetbot/franka demo convention directly
            dx = center[0] - cam_x
            dy = center[1] - cam_y
            yaw = math.degrees(math.atan2(dy, dx))

            # Pitch: tilt down toward building center
            dz = center[2] - cam_z
            dist_horiz = math.sqrt(dx ** 2 + dy ** 2)
            pitch = math.degrees(math.atan2(-dz, dist_horiz))

            orient = rot_utils.euler_angles_to_quats(
                np.array([0, pitch, yaw]), degrees=True
            )

            # Update camera BEFORE render step (critical for async pipeline)
            camera.set_world_pose(position=cam_pos, orientation=orient)
            world.step(render=True)

            rgba = camera.get_rgba()

            # Log diagnostics for first captured frame
            if i == 5:
                if rgba is not None:
                    log(f"  Frame 5 RGBA shape: {rgba.shape}, "
                        f"min={rgba.min()}, max={rgba.max()}, "
                        f"mean={rgba.mean():.1f}")
                else:
                    log("  Frame 5 RGBA: None!")

            if rgba is not None and i > 0:
                recorder.capture_rgba(rgba)

            if i % 60 == 0:
                log(f"  Frame {i}/{orbit_steps} | pos=({cam_x:.1f}, {cam_y:.1f}, {cam_z:.1f}) yaw={yaw:.1f} pitch={pitch:.1f}")

    log("  Tour recording complete.")


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    global _report_file
    _report_file = open(_report_path, "w")

    log("=" * 60)
    log("MSOE Building USD Inspection & Video Tour")
    log("=" * 60)

    # Create world
    my_world = World(stage_units_in_meters=1.0)

    # Load building USD
    usd_path = os.path.join(
        _project_dir, "assets", "environments", "msoe_building", "TestBuilding.usd"
    )
    log(f"\nLoading building USD: {usd_path}")
    add_reference_to_stage(usd_path, prim_path="/World/Building")
    log("Building loaded.")

    # Add lighting
    stage = omni.usd.get_context().get_stage()
    distant_light = UsdLux.DistantLight.Define(stage, Sdf.Path("/World/DistantLight"))
    distant_light.CreateIntensityAttr(500)
    dome_light = UsdLux.DomeLight.Define(stage, Sdf.Path("/World/DomeLight"))
    dome_light.CreateIntensityAttr(200)

    # Phase 1: Inspect hierarchy
    try:
        inspect_hierarchy(stage)
    except Exception as e:
        log(f"ERROR in Phase 1 (hierarchy): {e}")
        log(traceback.format_exc())

    # Phase 2: Compute bounding box
    try:
        bb_min, bb_max = compute_bounding_box(stage)
    except Exception as e:
        log(f"ERROR in Phase 2 (bounding box): {e}")
        log(traceback.format_exc())
        # Fallback bounding box
        bb_min = np.array([-10, -10, 0])
        bb_max = np.array([10, 10, 10])

    # Phase 3: Orbital camera tour
    try:
        orbital_tour(my_world, bb_min, bb_max)
    except Exception as e:
        log(f"ERROR in Phase 3 (orbital tour): {e}")
        log(traceback.format_exc())

    log("\n" + "=" * 60)
    log("Done!")
    log(f"  Report -> {_report_path}")
    log("  Video  -> videos/environment/msoe_building_tour.mp4")
    log("=" * 60)

    _report_file.close()
    simulation_app.close()


if __name__ == "__main__":
    main()
