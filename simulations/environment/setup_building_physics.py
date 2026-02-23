#!/usr/bin/env python3
"""
MSOE Building Physics Setup

Loads TestBuilding.usd (Revit export), applies physics collision, rigid bodies,
materials, and revolute joints for doors, then exports TestBuilding_physics.usd.

Design:
  - Walls, floors, roofs, stairs, glazing -> static colliders (triangle mesh)
  - Doors -> dynamic rigid bodies (convex hull) with revolute joints
  - MEP/HVAC/electrical/mullions -> skipped (no physics value)

Output:
  - assets/environments/msoe_building/TestBuilding_physics.usd
  - logs/building_physics_setup.txt  (categorization summary)
"""

import os
import sys

os.environ["OMNI_KIT_ACCEPT_EULA"] = "YES"

from isaacsim import SimulationApp

simulation_app = SimulationApp({"headless": True})

import omni.usd
from pxr import Usd, UsdGeom, Sdf
from isaacsim.core.api import World
from isaacsim.core.utils.stage import add_reference_to_stage

# Add project root for sim_utils import
_script_dir = os.path.dirname(os.path.abspath(__file__))
_project_dir = os.path.abspath(os.path.join(_script_dir, '..', '..'))
sys.path.insert(0, _project_dir)

from sim_utils.building_physics import (
    PrimCategory,
    categorize_prim,
    apply_static_collider,
    apply_dynamic_rigid_body,
    create_physics_materials,
    bind_physics_material,
    create_revolute_joint,
    estimate_door_hinge_offset,
    setup_physics_scene,
    CATEGORY_MATERIALS,
)

# Report file
_report_path = os.path.join(_project_dir, "logs", "building_physics_setup.txt")
os.makedirs(os.path.dirname(_report_path), exist_ok=True)
_report_file = None


def log(msg=""):
    print(msg, flush=True)
    if _report_file:
        _report_file.write(msg + "\n")
        _report_file.flush()


def main():
    global _report_file
    _report_file = open(_report_path, "w")

    log("=" * 60)
    log("MSOE Building Physics Setup")
    log("=" * 60)

    # --- Create world and load building ---
    world = World(stage_units_in_meters=1.0)
    stage = omni.usd.get_context().get_stage()

    building_usd = os.path.join(
        _project_dir, "assets", "environments", "msoe_building", "TestBuilding.usd"
    )
    log(f"\nLoading: {building_usd}")
    add_reference_to_stage(building_usd, prim_path="/World/Building")
    log("Building loaded.")

    # --- Setup physics scene and materials ---
    log("\nConfiguring physics scene...")
    setup_physics_scene(stage)
    material_paths = create_physics_materials(stage)
    log(f"Created {len(material_paths)} physics materials: {list(material_paths.keys())}")

    # --- Iterate over building children and apply physics ---
    building_prim = stage.GetPrimAtPath("/World/Building")
    children = building_prim.GetChildren()
    log(f"\nProcessing {len(children)} top-level children under /World/Building...")

    # Counters
    counts = {cat: 0 for cat in PrimCategory}
    applied_collider = 0
    applied_rigid_body = 0
    skipped_no_mesh = 0
    door_prims = []  # collect for joint creation

    for child in children:
        name = child.GetName()
        category = categorize_prim(name)
        counts[category] += 1

        if category == PrimCategory.SKIP:
            continue

        if category == PrimCategory.DYNAMIC_DOOR:
            if apply_dynamic_rigid_body(stage, child, mass=15.0):
                applied_rigid_body += 1
                door_prims.append(child)
            else:
                skipped_no_mesh += 1
                continue
        else:
            # All static categories use triangle mesh collision
            if apply_static_collider(stage, child, approx="none"):
                applied_collider += 1
            else:
                skipped_no_mesh += 1
                continue

        # Bind physics material
        mat_name = CATEGORY_MATERIALS.get(category)
        if mat_name and mat_name in material_paths:
            bind_physics_material(stage, child, material_paths[mat_name])

    # --- Create revolute joints for doors ---
    log(f"\nCreating revolute joints for {len(door_prims)} doors...")
    joints_created = 0
    for door in door_prims:
        door_path = str(door.GetPath())
        joint_path = door_path + "/HingeJoint"

        local_pos0, local_pos1 = estimate_door_hinge_offset(stage, door)

        create_revolute_joint(
            stage,
            joint_path=joint_path,
            body0_path="",  # world frame (static building)
            body1_path=door_path,
            axis="Z",
            lower_limit=-100.0,
            upper_limit=100.0,
            local_pos0=local_pos0,
            local_pos1=local_pos1,
        )
        joints_created += 1

    # --- Summary ---
    log(f"\n{'=' * 60}")
    log("CATEGORIZATION SUMMARY")
    log("=" * 60)
    for cat in PrimCategory:
        log(f"  {cat.name:25s} {counts[cat]:5d}")
    log(f"  {'TOTAL':25s} {sum(counts.values()):5d}")

    log(f"\n{'=' * 60}")
    log("PHYSICS APPLICATION SUMMARY")
    log("=" * 60)
    log(f"  Static colliders applied:    {applied_collider}")
    log(f"  Dynamic rigid bodies applied: {applied_rigid_body}")
    log(f"  Revolute joints created:      {joints_created}")
    log(f"  Skipped (no mesh child):      {skipped_no_mesh}")

    # --- Export physics USD ---
    output_usd = os.path.join(
        _project_dir, "assets", "environments", "msoe_building", "TestBuilding_physics.usd"
    )
    log(f"\nExporting: {output_usd}")
    stage.Export(output_usd)
    log("Export complete.")

    file_size_mb = os.path.getsize(output_usd) / (1024 * 1024)
    log(f"File size: {file_size_mb:.1f} MB")

    log(f"\n{'=' * 60}")
    log("Done!")
    log(f"  Report  -> {_report_path}")
    log(f"  Physics -> {output_usd}")
    log("=" * 60)

    _report_file.close()
    simulation_app.close()


if __name__ == "__main__":
    main()
