#!/usr/bin/env python3
"""
Pipeline Test: Floor Drop

Discovers floor levels dynamically, drops cubes on each floor, lets them
settle, and verifies cubes rest on solid surfaces. Generic — works on any
building USD processed by setup_building_physics.py.

Usage:
    sbatch scripts/slurm/submit_isaac_job.sh simulations/environment/test_floor_drop.py \
        --input assets/environments/msoe_building/MSOEBuildingTake2_physics.usd

Exit codes:
    0 = all cubes landed on floors (PASS)
    1 = one or more cubes fell through (FAIL)
"""

import argparse
import os
import sys
import time

parser = argparse.ArgumentParser(description="Floor drop test for physics USD.")
parser.add_argument("--input", required=True, help="Path to physics-ready USD file")
parser.add_argument("--report", default=None, help="Report log path")
args, _unknown = parser.parse_known_args()

os.environ["OMNI_KIT_ACCEPT_EULA"] = "YES"

from isaacsim import SimulationApp

simulation_app = SimulationApp({"headless": True})

import numpy as np
import omni.usd
from pxr import Usd, UsdGeom, UsdPhysics, Sdf, Gf
from isaacsim.core.api import World
from isaacsim.core.api.objects import DynamicCuboid

_script_dir = os.path.dirname(os.path.abspath(__file__))
_project_dir = os.path.abspath(os.path.join(_script_dir, '..', '..'))
sys.path.insert(0, _project_dir)

from sim_utils.building_physics import setup_physics_scene, discover_floor_levels

# --- Config ---
CUBE_SIZE = 0.3         # meters
DROP_HEIGHT = 2.0       # above floor center (stage units)
SETTLE_FRAMES = 600     # ~10s at 60Hz physics
PASS_THRESHOLD = -1.0   # cube must be above floor_z + threshold

# Resolve paths
input_usd = os.path.abspath(args.input)
stem = os.path.splitext(os.path.basename(input_usd))[0]

if args.report:
    report_path = os.path.abspath(args.report)
else:
    report_path = os.path.join(_project_dir, "logs", f"{stem}_floor_test.txt")

os.makedirs(os.path.dirname(report_path), exist_ok=True)

_report_file = None


def log(msg=""):
    print(msg, flush=True)
    if _report_file:
        _report_file.write(msg + "\n")
        _report_file.flush()


def freeze_doors(stage):
    """Make all RigidBodyAPI prims kinematic so doors don't interfere."""
    frozen = 0
    for prim in stage.Traverse():
        if prim.HasAPI(UsdPhysics.RigidBodyAPI):
            rb = UsdPhysics.RigidBodyAPI(prim)
            rb.CreateKinematicEnabledAttr(True)
            frozen += 1
    return frozen


def add_ground_plane(stage, z):
    """Large collision plane as safety net below building."""
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
    global _report_file
    _report_file = open(report_path, "w")
    t_start = time.time()

    log("=" * 60)
    log("FLOOR DROP TEST")
    log("=" * 60)
    log(f"Input: {input_usd}")

    if not os.path.exists(input_usd):
        log(f"ERROR: USD not found: {input_usd}")
        _report_file.close()
        simulation_app.close()
        sys.exit(1)

    # --- Scene setup ---
    world = World(stage_units_in_meters=1.0)
    stage = omni.usd.get_context().get_stage()

    log(f"Loading: {input_usd}")
    building_prim = stage.DefinePrim("/World/Building", "Xform")
    building_prim.GetReferences().AddReference(
        input_usd, primPath=Sdf.Path("/World/Building")
    )

    setup_physics_scene(stage)

    # Ground plane below building
    building_prim = stage.GetPrimAtPath("/World/Building")
    bbox_cache = UsdGeom.BBoxCache(Usd.TimeCode.Default(), [UsdGeom.Tokens.default_])
    bbox = bbox_cache.ComputeWorldBound(building_prim)
    bb_min = bbox.ComputeAlignedRange().GetMin()
    ground_z = bb_min[2] - 5.0
    add_ground_plane(stage, ground_z)

    frozen = freeze_doors(stage)
    log(f"Froze {frozen} dynamic bodies")

    # --- Discover floors ---
    log("\nDiscovering floor levels...")
    floors = discover_floor_levels(stage)
    log(f"Found {len(floors)} floor levels")

    if not floors:
        log("ERROR: No floor levels found!")
        _report_file.close()
        simulation_app.close()
        sys.exit(1)

    # Pick up to 8 evenly spaced floors for testing
    if len(floors) > 8:
        step = len(floors) / 8.0
        test_floors = [floors[int(i * step)] for i in range(8)]
    else:
        test_floors = floors

    # --- Spawn cubes ---
    log(f"\nSpawning cubes on {len(test_floors)} floors...")
    cubes = []  # list of (name, obj, floor_z, floor_name)
    colors = [
        [1.0, 0.15, 0.15], [0.2, 0.3, 1.0], [0.1, 0.8, 0.2],
        [1.0, 0.9, 0.1], [1.0, 0.4, 0.7], [0.0, 0.9, 0.9],
        [1.0, 0.5, 0.0], [0.5, 0.0, 1.0],
    ]

    for i, fl in enumerate(test_floors):
        name = f"test_cube_{fl['name']}"
        drop_z = fl["z"] + DROP_HEIGHT
        color = colors[i % len(colors)]

        cube = world.scene.add(DynamicCuboid(
            prim_path=f"/World/{name}",
            name=name,
            position=np.array([fl["center_x"], fl["center_y"], drop_z]),
            scale=np.array([CUBE_SIZE] * 3),
            color=np.array(color),
        ))
        cubes.append((name, cube, fl["z"], fl["name"]))
        log(f"  {fl['name']:12s}  drop=({fl['center_x']:.1f}, {fl['center_y']:.1f}, {drop_z:.1f})")

    # --- Simulate ---
    log(f"\nSimulating {SETTLE_FRAMES} frames...")
    world.reset()

    # Warmup
    for _ in range(30):
        world.step(render=False)

    for i in range(SETTLE_FRAMES):
        world.step(render=False)
        if i % 120 == 0:
            log(f"  Frame {i}/{SETTLE_FRAMES}")

    # --- Validate ---
    log(f"\n{'=' * 60}")
    log("RESULTS")
    log("=" * 60)

    all_pass = True
    passed_count = 0
    for name, cube, floor_z, floor_name in cubes:
        pos, _ = cube.get_world_pose()
        final_z = float(pos[2])
        passed = final_z > (floor_z + PASS_THRESHOLD)
        if passed:
            passed_count += 1
        else:
            all_pass = False
        status = "PASS" if passed else "FAIL"
        log(f"  {floor_name:12s}  final_z={final_z:7.2f}  floor_z={floor_z:7.2f}  [{status}]")

    log(f"\n  {passed_count}/{len(cubes)} cubes landed on floors")
    log(f"  OVERALL: {'PASS' if all_pass else 'FAIL'}")
    log(f"  Runtime: {time.time() - t_start:.1f}s")
    log(f"  Report:  {report_path}")

    _report_file.close()

    exit_code = 0 if all_pass else 1
    os._exit(exit_code)


if __name__ == "__main__":
    main()
