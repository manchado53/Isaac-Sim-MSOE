#!/usr/bin/env python3
"""
Pipeline Test: Door Swing

Finds all revolute-jointed doors with DriveAPI in a physics USD, drives them
to 80 degrees, measures rotation, and reports pass/fail. Generic — works on
any building USD processed by setup_building_physics.py.

Usage:
    sbatch submit_isaac_job.sh simulations/environment/test_door_swing_generic.py \
        --input assets/environments/msoe_building/MSOEBuildingTake2_physics.usd

Exit codes:
    0 = all doors rotated past threshold (PASS)
    1 = one or more doors stuck (FAIL)
"""

import argparse
import os
import sys
import math
import time

parser = argparse.ArgumentParser(description="Door swing test for physics USD.")
parser.add_argument("--input", required=True, help="Path to physics-ready USD file")
parser.add_argument("--report", default=None, help="Report log path")
args, _unknown = parser.parse_known_args()

os.environ["OMNI_KIT_ACCEPT_EULA"] = "YES"

from isaacsim import SimulationApp

simulation_app = SimulationApp({"headless": True})

import omni.usd
from pxr import Usd, UsdGeom, UsdPhysics, Sdf

_script_dir = os.path.dirname(os.path.abspath(__file__))
_project_dir = os.path.abspath(os.path.join(_script_dir, '..', '..'))
sys.path.insert(0, _project_dir)

from isaacsim.core.api import World
from sim_utils.building_physics import setup_physics_scene, set_drive_target

# --- Config ---
DRIVE_TARGET_ANGLE = 80.0   # degrees
DRIVE_FRAMES = 300           # ~5s at 60Hz physics
PASS_THRESHOLD = 5.0         # degrees — minimum rotation to count as working

# Resolve paths
input_usd = os.path.abspath(args.input)
stem = os.path.splitext(os.path.basename(input_usd))[0]

if args.report:
    report_path = os.path.abspath(args.report)
else:
    report_path = os.path.join(_project_dir, "logs", f"{stem}_door_test.txt")

os.makedirs(os.path.dirname(report_path), exist_ok=True)

_report_file = None


def log(msg=""):
    print(msg, flush=True)
    if _report_file:
        _report_file.write(msg + "\n")
        _report_file.flush()


def get_door_angle_from_orient(prim):
    """Read the door's Z-axis rotation from its local xformOp:orient.

    IMPORTANT: Do NOT use ComputeLocalToWorldTransform — parent scale
    from Revit corrupts quaternion extraction.
    """
    xformable = UsdGeom.Xformable(prim)
    for op in xformable.GetOrderedXformOps():
        if op.GetOpType() == UsdGeom.XformOp.TypeOrient:
            quat = op.Get()
            if quat is None:
                return 0.0
            w = float(quat.GetReal())
            img = quat.GetImaginary()
            x, y, z = float(img[0]), float(img[1]), float(img[2])
            siny_cosp = 2.0 * (w * z + x * y)
            cosy_cosp = 1.0 - 2.0 * (y * y + z * z)
            return math.degrees(math.atan2(siny_cosp, cosy_cosp))
    return 0.0


def find_driven_doors(stage):
    """Find all revolute joints with angular DriveAPI on Door prims.

    Returns list of (joint_path_str, joint_prim, drive, door_prim).
    """
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
        joint = UsdPhysics.RevoluteJoint(prim)
        targets = joint.GetBody1Rel().GetTargets()
        if not targets:
            continue
        door_prim = stage.GetPrimAtPath(targets[0])
        if not door_prim.IsValid():
            continue
        doors.append((path_str, prim, drive, door_prim))
    return doors


def door_short_name(path_str):
    """Extract short door identifier from path."""
    for part in path_str.split("/"):
        if part.startswith("Doors_") or part.startswith("Door_"):
            return part
    return path_str.split("/")[-2] if "/" in path_str else path_str


def main():
    global _report_file
    _report_file = open(report_path, "w")
    t_start = time.time()

    log("=" * 60)
    log("DOOR SWING TEST")
    log("=" * 60)
    log(f"Input:     {input_usd}")
    log(f"Target:    {DRIVE_TARGET_ANGLE} degrees")
    log(f"Threshold: {PASS_THRESHOLD} degrees")

    if not os.path.exists(input_usd):
        log(f"ERROR: USD not found: {input_usd}")
        _report_file.close()
        simulation_app.close()
        sys.exit(1)

    # --- Scene setup ---
    world = World(stage_units_in_meters=1.0)
    stage = omni.usd.get_context().get_stage()

    log(f"\nLoading: {input_usd}")
    building_prim = stage.DefinePrim("/World/Building", "Xform")
    building_prim.GetReferences().AddReference(
        input_usd, primPath=Sdf.Path("/World/Building")
    )

    setup_physics_scene(stage)

    # --- Find doors ---
    all_doors = find_driven_doors(stage)
    log(f"Found {len(all_doors)} driven doors")

    if not all_doors:
        log("\nWARNING: No driven doors found in this USD.")
        log("This could be normal if the building has no hinged doors.")
        log("OVERALL: PASS (no doors to test)")
        _report_file.close()
        os._exit(0)

    # Record initial angles
    initial_angles = {}
    for path_str, joint_prim, drive, door_prim in all_doors:
        initial_angles[path_str] = get_door_angle_from_orient(door_prim)

    # --- Simulate ---
    world.reset()

    # Warmup
    log(f"\nWarming up (30 frames)...")
    for _ in range(30):
        world.step(render=False)

    # Set all doors to target angle
    log(f"Driving {len(all_doors)} doors to {DRIVE_TARGET_ANGLE} degrees...")
    for path_str, joint_prim, drive, door_prim in all_doors:
        set_drive_target(drive, DRIVE_TARGET_ANGLE)

    # Simulate
    for i in range(DRIVE_FRAMES):
        world.step(render=False)
        if i % 60 == 0:
            log(f"  Frame {i}/{DRIVE_FRAMES}")

    # --- Validate ---
    log(f"\n{'=' * 60}")
    log("RESULTS")
    log("=" * 60)

    passed_count = 0
    failed_count = 0
    failed_doors = []

    for path_str, joint_prim, drive, door_prim in all_doors:
        name = door_short_name(path_str)
        initial = initial_angles.get(path_str, 0.0)
        final = get_door_angle_from_orient(door_prim)
        delta = abs(final - initial)
        passed = delta > PASS_THRESHOLD

        if passed:
            passed_count += 1
        else:
            failed_count += 1
            failed_doors.append(name)

        status = "PASS" if passed else "FAIL"
        log(f"  {name:<30s}  {initial:7.1f} -> {final:7.1f}  "
            f"delta={delta:5.1f}  [{status}]")

    all_pass = failed_count == 0

    log(f"\n  Passed: {passed_count}/{len(all_doors)}")
    log(f"  Failed: {failed_count}/{len(all_doors)}")
    if failed_doors:
        log(f"  Failed doors: {', '.join(failed_doors)}")
    log(f"  OVERALL: {'PASS' if all_pass else 'FAIL'}")
    log(f"  Runtime: {time.time() - t_start:.1f}s")
    log(f"  Report:  {report_path}")

    _report_file.close()

    exit_code = 0 if all_pass else 1
    os._exit(exit_code)


if __name__ == "__main__":
    main()
