#!/usr/bin/env python3
"""
Innovation Center USD Inspection & Video Tour

Loads innovation_center_finalV1.usd, inspects the scene hierarchy, discovers
floor levels via Z-clustering, computes per-floor stats (bounding box, area,
element counts), and records an orbital camera tour.

Output:
  - logs/innovation_center_inspection.txt  -> hierarchy, floors, dimensions
  - videos/innovation_center_tour.mp4      -> orbital tour video
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
from pxr import Sdf, Usd, UsdGeom, UsdLux, Gf
from isaacsim.core.api import World
from isaacsim.core.utils.stage import add_reference_to_stage
from isaacsim.sensors.camera import Camera

# Add project root for sim_utils import
_script_dir = os.path.dirname(os.path.abspath(__file__))
_project_dir = os.path.abspath(os.path.join(_script_dir, '..', '..'))
sys.path.insert(0, _project_dir)
from sim_utils.video_capture import VideoRecorder
from sim_utils.building_physics import iter_building_elements

# Report file
_report_path = os.path.join(_project_dir, "logs", "innovation_center_inspection.txt")
os.makedirs(os.path.dirname(_report_path), exist_ok=True)
_report_file = None


def log(msg=""):
    """Print to both stdout (flushed) and the report file."""
    print(msg, flush=True)
    if _report_file:
        _report_file.write(msg + "\n")
        _report_file.flush()


# ---------------------------------------------------------------------------
# Phase 1: Inspect Hierarchy
# ---------------------------------------------------------------------------

def print_prim_tree(prim, depth=0, max_depth=3):
    """Recursively print prim hierarchy up to max_depth."""
    if depth > max_depth:
        return
    indent = "  " * depth
    type_name = prim.GetTypeName() or "(untyped)"
    log(f"{indent}{prim.GetName()} [{type_name}]")
    children = prim.GetChildren()
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
# Phase 2: Bounding Box
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
# Phase 3: Discover Floor Levels (Z-clustering)
# ---------------------------------------------------------------------------

def discover_floor_levels(stage, building_path="/World/Building"):
    """Discover distinct floor surfaces by clustering floor element bboxes.

    Returns list of dicts sorted by z:
        [{"name", "z", "center_x", "center_y", "x_min", "x_max",
          "y_min", "y_max", "total_area", "count"}, ...]
    """
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
            "x_min": bb_min[0],
            "x_max": bb_max[0],
            "y_min": bb_min[1],
            "y_max": bb_max[1],
            "area": dx * dy,
        })

    if not floor_data:
        return []

    # Cluster by Z height (1.5m tolerance)
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

    # Aggregate each cluster
    levels = []
    for i, cluster in enumerate(clusters):
        total_area = sum(f["area"] for f in cluster)
        if total_area < 0.01:
            continue
        avg_x = sum(f["x_center"] * f["area"] for f in cluster) / total_area
        avg_y = sum(f["y_center"] * f["area"] for f in cluster) / total_area
        avg_z = sum(f["z_top"] * f["area"] for f in cluster) / total_area
        x_min = min(f["x_min"] for f in cluster)
        x_max = max(f["x_max"] for f in cluster)
        y_min = min(f["y_min"] for f in cluster)
        y_max = max(f["y_max"] for f in cluster)
        levels.append({
            "name": f"floor_{i + 1}",
            "z": avg_z,
            "center_x": avg_x,
            "center_y": avg_y,
            "x_min": x_min,
            "x_max": x_max,
            "y_min": y_min,
            "y_max": y_max,
            "total_area": total_area,
            "count": len(cluster),
        })

    return levels


# ---------------------------------------------------------------------------
# Phase 4: Per-Floor Element Counts by Category
# ---------------------------------------------------------------------------

def count_elements_per_floor(stage, floors, building_path="/World/Building"):
    """Count elements per category for each discovered floor level.

    Assigns elements to the nearest floor by Z proximity.
    Returns dict: {floor_name: {category: count}}
    """
    bbox_cache = UsdGeom.BBoxCache(
        Usd.TimeCode.Default(), [UsdGeom.Tokens.default_]
    )
    building_prim = stage.GetPrimAtPath(building_path)

    floor_counts = {fl["name"]: {} for fl in floors}
    unassigned = 0

    for _level, category_name, element in iter_building_elements(stage, building_prim):
        bbox = bbox_cache.ComputeWorldBound(element)
        box_range = bbox.ComputeAlignedRange()
        bb_min = box_range.GetMin()
        bb_max = box_range.GetMax()
        if bb_min == bb_max:
            continue
        elem_z = (bb_min[2] + bb_max[2]) / 2.0

        # Find nearest floor
        best_floor = None
        best_dist = float('inf')
        for fl in floors:
            dist = abs(elem_z - fl["z"])
            if dist < best_dist:
                best_dist = dist
                best_floor = fl["name"]

        if best_floor is None:
            unassigned += 1
            continue

        cat = category_name or "uncategorized"
        floor_counts[best_floor][cat] = floor_counts[best_floor].get(cat, 0) + 1

    return floor_counts, unassigned


# ---------------------------------------------------------------------------
# Phase 5: Orbital Camera Tour
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

    world.scene.add_default_ground_plane()

    # Create camera
    initial_pos = np.array([center[0] + radius, center[1], cam_height_base])
    initial_yaw = 180.0
    initial_pitch = 20.0
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

    # Warm up rendering
    for _ in range(30):
        world.step(render=True)

    orbit_steps = 360
    log(f"  Recording {orbit_steps} frames...")

    with VideoRecorder("innovation_center_tour", fps=15) as recorder:
        for i in range(orbit_steps):
            angle = 2 * math.pi * i / orbit_steps

            height_offset = extent[2] * 0.1 * math.sin(2 * angle)
            cam_x = center[0] + radius * math.cos(angle)
            cam_y = center[1] + radius * math.sin(angle)
            cam_z = cam_height_base + height_offset

            cam_pos = np.array([cam_x, cam_y, cam_z])

            dx = center[0] - cam_x
            dy = center[1] - cam_y
            yaw = math.degrees(math.atan2(dy, dx))

            dz = center[2] - cam_z
            dist_horiz = math.sqrt(dx ** 2 + dy ** 2)
            pitch = math.degrees(math.atan2(-dz, dist_horiz))

            orient = rot_utils.euler_angles_to_quats(
                np.array([0, pitch, yaw]), degrees=True
            )

            camera.set_world_pose(position=cam_pos, orientation=orient)
            world.step(render=True)

            rgba = camera.get_rgba()

            if i == 5 and rgba is not None:
                log(f"  Frame 5 RGBA shape: {rgba.shape}, "
                    f"min={rgba.min()}, max={rgba.max()}, "
                    f"mean={rgba.mean():.1f}")

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
    log("INNOVATION CENTER USD INSPECTION")
    log("=" * 60)

    # Create world
    my_world = World(stage_units_in_meters=1.0)

    # Load building USD
    usd_path = os.path.join(
        _project_dir, "assets", "environments", "msoe_building",
        "innovation_center_finalV1.usd"
    )
    log(f"\nLoading building USD: {usd_path}")
    if not os.path.exists(usd_path):
        log(f"ERROR: USD file not found at {usd_path}")
        _report_file.close()
        simulation_app.close()
        sys.exit(1)

    add_reference_to_stage(usd_path, prim_path="/World/Building")
    log("Building loaded.")

    # Add lighting
    stage = omni.usd.get_context().get_stage()
    distant_light = UsdLux.DistantLight.Define(stage, Sdf.Path("/World/DistantLight"))
    distant_light.CreateIntensityAttr(500)
    dome_light = UsdLux.DomeLight.Define(stage, Sdf.Path("/World/DomeLight"))
    dome_light.CreateIntensityAttr(200)

    # Phase 1: Inspect hierarchy
    log("\n--- Phase 1: Hierarchy Inspection ---")
    try:
        inspect_hierarchy(stage)
    except Exception as e:
        log(f"ERROR in Phase 1 (hierarchy): {e}")
        import traceback
        log(traceback.format_exc())

    # Phase 2: Compute bounding box
    log("\n--- Phase 2: Bounding Box ---")
    try:
        bb_min, bb_max = compute_bounding_box(stage)
    except Exception as e:
        log(f"ERROR in Phase 2 (bounding box): {e}")
        import traceback
        log(traceback.format_exc())
        bb_min = np.array([-10, -10, 0])
        bb_max = np.array([10, 10, 10])

    # Phase 3: Discover floor levels
    log("\n--- Phase 3: Floor Level Discovery ---")
    floors = []
    try:
        floors = discover_floor_levels(stage)
        log(f"\n{'=' * 60}")
        log(f"FLOOR LEVELS  (found: {len(floors)})")
        log("=" * 60)
        for fl in floors:
            log(f"\n  {fl['name'].upper()}")
            log(f"    Z height:   {fl['z']:.2f} m")
            log(f"    XY center:  ({fl['center_x']:.2f}, {fl['center_y']:.2f})")
            log(f"    XY extent:  X=[{fl['x_min']:.2f}, {fl['x_max']:.2f}]  "
                f"Y=[{fl['y_min']:.2f}, {fl['y_max']:.2f}]")
            xy_width = fl['x_max'] - fl['x_min']
            xy_depth = fl['y_max'] - fl['y_min']
            log(f"    XY size:    {xy_width:.2f} x {xy_depth:.2f} m")
            log(f"    Floor area: {fl['total_area']:.1f} m²  ({fl['count']} elements)")

        if not floors:
            log("  WARNING: No floor levels discovered via Floors category.")
            log("  The building may use a different structure. Check hierarchy above.")
    except Exception as e:
        log(f"ERROR in Phase 3 (floor discovery): {e}")
        import traceback
        log(traceback.format_exc())

    # Phase 4: Per-floor element counts
    if floors:
        log("\n--- Phase 4: Per-Floor Element Counts ---")
        try:
            floor_counts, unassigned = count_elements_per_floor(stage, floors)
            log(f"\n{'=' * 60}")
            log("ELEMENTS BY FLOOR AND CATEGORY")
            log("=" * 60)
            for fl in floors:
                counts = floor_counts[fl["name"]]
                total = sum(counts.values())
                log(f"\n  {fl['name'].upper()} ({total} elements)")
                for cat, cnt in sorted(counts.items(), key=lambda x: -x[1]):
                    log(f"    {cat:30s} {cnt}")
            if unassigned > 0:
                log(f"\n  Unassigned elements: {unassigned}")
        except Exception as e:
            log(f"ERROR in Phase 4 (element counts): {e}")
            import traceback
            log(traceback.format_exc())

    # Phase 5: Orbital camera tour
    log("\n--- Phase 5: Orbital Camera Tour ---")
    try:
        orbital_tour(my_world, bb_min, bb_max)
    except Exception as e:
        log(f"ERROR in Phase 5 (orbital tour): {e}")
        import traceback
        log(traceback.format_exc())

    log("\n" + "=" * 60)
    log("INSPECTION COMPLETE")
    log(f"  Report -> {_report_path}")
    log("  Video  -> videos/innovation_center_tour.mp4")
    log("=" * 60)

    _report_file.close()
    simulation_app.close()


if __name__ == "__main__":
    main()
