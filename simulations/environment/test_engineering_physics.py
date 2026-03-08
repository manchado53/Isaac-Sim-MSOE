#!/usr/bin/env python3
"""
Test Innovation Center Physics — Floor Drop Test

Loads innovation_center_physics.usd, discovers floor surfaces dynamically,
drops colored cubes above each floor level, and verifies they rest on
surfaces rather than falling through.

Output:
  - videos/test_physics_innovation_center.mp4
  - PASS/FAIL per floor level based on final cube positions
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
from isaacsim.core.api.objects import DynamicCuboid
from isaacsim.sensors.camera import Camera

_script_dir = os.path.dirname(os.path.abspath(__file__))
_project_dir = os.path.abspath(os.path.join(_script_dir, '..', '..'))
sys.path.insert(0, _project_dir)

from sim_utils.video_capture import VideoRecorder
from sim_utils.building_physics import (
    setup_physics_scene, iter_building_elements,
)


CUBE_SIZE = 0.3  # meters
FLOOR_COLORS = [
    [1.0, 0.0, 0.0],   # red
    [0.0, 1.0, 0.0],   # green
    [0.0, 0.0, 1.0],   # blue
    [1.0, 1.0, 0.0],   # yellow
    [1.0, 0.0, 1.0],   # magenta
    [0.0, 1.0, 1.0],   # cyan
    [1.0, 0.5, 0.0],   # orange
    [0.5, 0.0, 1.0],   # purple
]


# ---------------------------------------------------------------------------
# Floor Discovery
# ---------------------------------------------------------------------------

def discover_floor_levels(stage, building_path="/World/Building"):
    """Discover distinct floor surfaces by clustering floor element bboxes.

    Returns list of dicts sorted by z:
        [{"name", "z", "center_x", "center_y", "count"}, ...]
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

    # Aggregate each cluster with area-weighted center
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
    print("=" * 60, flush=True)
    print("Test Innovation Center Physics — Floor Drop Test", flush=True)
    print("=" * 60, flush=True)

    # --- Load physics-enabled building ---
    world = World(stage_units_in_meters=1.0)
    stage = omni.usd.get_context().get_stage()

    physics_usd = os.path.join(
        _project_dir, "assets", "environments", "msoe_building",
        "innovation_center_physics.usd"
    )
    if not os.path.exists(physics_usd):
        print(f"ERROR: {physics_usd} not found.", flush=True)
        print("Run setup_innovation_center_physics.py first.", flush=True)
        simulation_app.close()
        sys.exit(1)

    print(f"Loading: {physics_usd}", flush=True)
    building_prim = stage.DefinePrim("/World/Building", "Xform")
    building_prim.GetReferences().AddReference(
        physics_usd, primPath=Sdf.Path("/World/Building")
    )
    print("Building loaded.", flush=True)

    setup_physics_scene(stage)

    # --- Building bbox for ground plane and camera ---
    building_prim = stage.GetPrimAtPath("/World/Building")
    bbox_cache = UsdGeom.BBoxCache(
        Usd.TimeCode.Default(), [UsdGeom.Tokens.default_]
    )
    bbox = bbox_cache.ComputeWorldBound(building_prim)
    box_range = bbox.ComputeAlignedRange()
    bb_min = box_range.GetMin()
    bb_max = box_range.GetMax()

    bldg_center = [(bb_min[i] + bb_max[i]) / 2.0 for i in range(3)]
    bldg_extent = [bb_max[i] - bb_min[i] for i in range(3)]
    ground_z = bb_min[2] - 0.5

    print(f"Building bbox: ({bb_min[0]:.1f}, {bb_min[1]:.1f}, {bb_min[2]:.1f}) to "
          f"({bb_max[0]:.1f}, {bb_max[1]:.1f}, {bb_max[2]:.1f})", flush=True)
    print(f"Ground plane at Z={ground_z:.1f}", flush=True)

    add_ground_plane(stage, ground_z)

    # --- Discover floor levels ---
    print("\nDiscovering floor levels...", flush=True)
    floors = discover_floor_levels(stage)
    print(f"Found {len(floors)} distinct floor levels:", flush=True)
    for fl in floors:
        print(f"  {fl['name']:12s}  Z={fl['z']:.2f}m  "
              f"center=({fl['center_x']:.1f}, {fl['center_y']:.1f})  "
              f"({fl['count']} elements)", flush=True)

    if not floors:
        print("ERROR: No floor levels found!", flush=True)
        simulation_app.close()
        sys.exit(1)

    # --- Lighting ---
    UsdLux.DistantLight.Define(stage, Sdf.Path("/World/DistantLight")).CreateIntensityAttr(500)
    UsdLux.DomeLight.Define(stage, Sdf.Path("/World/DomeLight")).CreateIntensityAttr(200)

    # --- Drop cubes (one per floor level) ---
    print(f"\n{'=' * 60}", flush=True)
    print("SETTING UP DROP TEST CUBES", flush=True)
    print("=" * 60, flush=True)

    all_cubes = []  # (floor_name, cube, floor_z)
    for i, fl in enumerate(floors):
        color = FLOOR_COLORS[i % len(FLOOR_COLORS)]
        drop_z = fl["z"] + 2.0

        cube = world.scene.add(
            DynamicCuboid(
                prim_path=f"/World/{fl['name']}",
                name=fl["name"],
                position=np.array([fl["center_x"], fl["center_y"], drop_z]),
                scale=np.array([CUBE_SIZE, CUBE_SIZE, CUBE_SIZE]),
                color=np.array(color),
            )
        )
        all_cubes.append((fl["name"], cube, fl["z"]))
        print(f"  {fl['name']:12s}  drop=({fl['center_x']:.1f}, {fl['center_y']:.1f}, {drop_z:.1f})  "
              f"floor_z={fl['z']:.2f}", flush=True)

    print(f"\nAdded {len(all_cubes)} test cubes", flush=True)

    # --- Overview camera ---
    horiz_diag = math.sqrt(bldg_extent[0]**2 + bldg_extent[1]**2)
    cam_distance = horiz_diag * 1.2
    target = bldg_center
    cam_pos = [
        bldg_center[0] + cam_distance * 0.7,
        bldg_center[1] - cam_distance * 0.5,
        bldg_center[2] + cam_distance * 0.4,
    ]
    orient = look_at_orientation(cam_pos, target)
    camera = Camera(
        prim_path="/World/TestCamera",
        position=np.array(cam_pos),
        frequency=30,
        resolution=(1280, 720),
        orientation=orient,
    )

    # --- Run simulation ---
    world.reset()
    camera.initialize()

    for _ in range(20):
        world.step(render=True)

    total_frames = 300
    print(f"\nRecording {total_frames} frames ({total_frames / 30:.0f}s at 30fps)...",
          flush=True)

    with VideoRecorder("test_physics_innovation_center", fps=30) as recorder:
        for i in range(total_frames):
            world.step(render=True)
            rgba = camera.get_rgba()
            if rgba is not None and i > 0:
                recorder.capture_rgba(rgba)
            if i % 60 == 0:
                print(f"  Frame {i}/{total_frames}", flush=True)

    # --- Check final positions ---
    print(f"\n{'=' * 60}", flush=True)
    print("FINAL CUBE POSITIONS", flush=True)
    print("=" * 60, flush=True)

    passed_count = 0
    failed_count = 0
    for floor_name, cube, floor_z in all_cubes:
        pos, _ = cube.get_world_pose()
        z = pos[2]
        passed = z > (floor_z - 1.0)
        status = "PASS" if passed else "FAIL"
        if passed:
            passed_count += 1
        else:
            failed_count += 1
        print(f"  {floor_name:12s}  final_z={z:.2f}  floor_z={floor_z:.2f}  [{status}]",
              flush=True)

    # --- Summary ---
    print(f"\n{'=' * 60}", flush=True)
    total = passed_count + failed_count
    if failed_count == 0:
        print(f"OVERALL: PASS -- All {total} cubes resting on surfaces", flush=True)
    else:
        print(f"OVERALL: FAIL -- {failed_count}/{total} cubes fell through", flush=True)
    print("=" * 60, flush=True)

    simulation_app.close()


if __name__ == "__main__":
    main()
