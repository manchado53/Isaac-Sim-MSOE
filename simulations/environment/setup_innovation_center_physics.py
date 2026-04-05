#!/usr/bin/env python3
"""
MSOE Innovation Center Physics Setup

Loads the Revit-exported Innovation Center (MSOEBuildingTake2) USD, applies physics
collisions, materials, and joints, then exports innovation_center_physics.usd.

Design:
  - Walls, floors, ceilings, columns, framing -> static colliders (triangle mesh)
  - Stairs, railings -> static colliders (triangle mesh)
  - Glazing, windows, curtain panels -> static colliders (triangle mesh)
  - Doors -> dynamic rigid bodies (convex hull) with revolute joints
  - MEP/HVAC/electrical/plumbing/lighting -> skipped (no physics value)

Building characteristics:
  - 17 floor levels with hierarchical category folders (Doors, Walls, etc.)
  - 77,485 meshes hidden behind USD instancing (7,106 instanceable elements)
  - De-instancing + CollisionAPI confirmed working
  - Stage units in feet (metersPerUnit=0.3048), Isaac Sim handles conversion

SLURM resource requirements:
  #SBATCH --time=04:00:00
  #SBATCH --mem=32G

Output:
  - assets/environments/msoe_building/innovation_center_physics.usd
  - logs/innovation_center_physics_setup.txt  (categorization summary)
"""

import os
import sys
import time

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
    DoorMechanism,
    categorize_prim,
    classify_door_mechanism,
    apply_static_collider,
    apply_dynamic_rigid_body,
    create_physics_materials,
    bind_physics_material,
    create_revolute_joint,
    estimate_door_hinge_offset,
    setup_door_drive,
    setup_physics_scene,
    add_door_collision_filters,
    CATEGORY_MATERIALS,
    DOOR_DRIVE_PARAMS,
    iter_building_elements,
)

# Report file
_report_path = os.path.join(_project_dir, "logs", "innovation_center_physics_setup.txt")
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
    t_start = time.time()

    log("=" * 70)
    log("MSOE INNOVATION CENTER PHYSICS SETUP")
    log("=" * 70)

    # --- Create world and load building ---
    world = World(stage_units_in_meters=1.0)
    stage = omni.usd.get_context().get_stage()

    building_usd = os.path.join(
        _project_dir, "assets", "environments", "msoe_building",
        "MSOEBuildingTake2.usd"
    )
    log(f"\nLoading: {building_usd}")
    add_reference_to_stage(building_usd, prim_path="/World/Building")
    log("Building loaded.")

    # --- Setup physics scene and materials ---
    log("\nConfiguring physics scene...")
    setup_physics_scene(stage)
    material_paths = create_physics_materials(stage)
    log(f"Created {len(material_paths)} physics materials: {list(material_paths.keys())}")

    # --- Iterate over building elements and apply physics ---
    building_prim = stage.GetPrimAtPath("/World/Building")
    log(f"\nProcessing all levels under /World/Building...")
    log(f"Using iter_building_elements() with deep_search=True for instanced meshes\n")

    # Counters — global
    counts = {cat: 0 for cat in PrimCategory}
    applied_collider = 0
    applied_rigid_body = 0
    skipped_no_mesh = 0
    door_prims = []

    # Per-level counters
    level_counts = {}  # level_name -> {category: count}
    level_colliders = {}  # level_name -> int
    current_level = None

    for level_name, category_name, element in iter_building_elements(stage, building_prim):
        # Track per-level progress
        if level_name != current_level:
            if current_level is not None:
                _log_level_summary(current_level, level_counts.get(current_level, {}),
                                   level_colliders.get(current_level, 0))
            current_level = level_name
            log(f"\n--- Level: {level_name} ---")

        # Initialize per-level counters
        if level_name not in level_counts:
            level_counts[level_name] = {cat: 0 for cat in PrimCategory}
            level_colliders[level_name] = 0

        # Categorize using category_hint (folder name) for hierarchical building
        element_name = element.GetName()
        category = categorize_prim(element_name, category_hint=category_name)
        counts[category] += 1
        level_counts[level_name][category] += 1

        if category == PrimCategory.SKIP:
            continue

        if category == PrimCategory.DYNAMIC_DOOR:
            mechanism = classify_door_mechanism(element)
            # Only set up hinged doors (single and double)
            if mechanism not in (DoorMechanism.HINGED_SINGLE, DoorMechanism.HINGED_DOUBLE):
                counts[PrimCategory.SKIP] += 1
                level_counts[level_name][PrimCategory.SKIP] = level_counts[level_name].get(PrimCategory.SKIP, 0) + 1
                log(f"  Skipping non-hinged door ({mechanism.name}): {element.GetName()}")
                continue
            door_mass = 30.0 if mechanism == DoorMechanism.HINGED_DOUBLE else 15.0
            if apply_dynamic_rigid_body(stage, element, mass=door_mass, deep_search=True):
                applied_rigid_body += 1
                door_prims.append((level_name, element, mechanism))
                # Bind Wood physics material to door
                mat_name = CATEGORY_MATERIALS.get(category)
                if mat_name and mat_name in material_paths:
                    bind_physics_material(stage, element, material_paths[mat_name])
            else:
                skipped_no_mesh += 1
            continue

        # Static categories: structural, glazing, stairs
        if apply_static_collider(stage, element, approx="none", deep_search=True):
            applied_collider += 1
            level_colliders[level_name] = level_colliders.get(level_name, 0) + 1
        else:
            skipped_no_mesh += 1
            continue

        # Bind physics material based on category
        mat_name = CATEGORY_MATERIALS.get(category)
        if mat_name and mat_name in material_paths:
            bind_physics_material(stage, element, material_paths[mat_name])

    # Log final level
    if current_level is not None:
        _log_level_summary(current_level, level_counts.get(current_level, {}),
                           level_colliders.get(current_level, 0))

    # --- Create revolute joints and drives for doors ---
    log(f"\nCreating revolute joints for {len(door_prims)} doors...")
    joints_created = 0
    joints_skipped = 0
    drives_created = 0
    drives_skipped = 0
    for level_name, door, mechanism in door_prims:
        door_path = str(door.GetPath())
        joint_path = door_path + "/HingeJoint"

        local_pos0, local_pos1 = estimate_door_hinge_offset(stage, door)
        if local_pos0 is None:
            log(f"  WARNING: Could not estimate hinge for {door_path} — skipping joint")
            joints_skipped += 1
            continue

        # Double doors get reduced angular limits since both leaves swing as unit
        if mechanism == DoorMechanism.HINGED_DOUBLE:
            lower_limit, upper_limit = -70.0, 70.0
        else:
            lower_limit, upper_limit = -100.0, 100.0

        joint = create_revolute_joint(
            stage,
            joint_path=joint_path,
            body0_path="",          # world frame (static building)
            body1_path=door_path,
            axis="Z",               # vertical hinge axis
            lower_limit=lower_limit,
            upper_limit=upper_limit,
            local_pos0=local_pos0,
            local_pos1=local_pos1,
        )
        joints_created += 1
        log(f"  Joint ({mechanism.name}): {door_path}")

        # Apply DriveAPI for hinged doors
        drive_params = DOOR_DRIVE_PARAMS.get(mechanism)
        if drive_params is not None:
            stiffness, damping, max_force = drive_params
            setup_door_drive(joint, stiffness=stiffness, damping=damping,
                             max_force=max_force, target_position=0.0)
            drives_created += 1
        else:
            drives_skipped += 1
            log(f"  Drive skipped ({mechanism.name}): {door_path}")

    log(f"  Revolute joints created: {joints_created}")
    if joints_skipped > 0:
        log(f"  Joints skipped (no bbox): {joints_skipped}")
    log(f"  Drives created:           {drives_created}")
    log(f"  Drives skipped:           {drives_skipped}")

    # --- Collision filtering for doors ---
    # Inflate each door's bbox by 1.0m to catch nearby mullions/panels
    # that the door sweeps into during the first degrees of rotation.
    log(f"\nApplying collision filters for {len(door_prims)} doors (margin=1.0m)...")
    door_prim_list = [door for _, door, _ in door_prims]
    filtered = add_door_collision_filters(stage, door_prim_list, margin=1.0)
    log(f"  Filtered {filtered} door-static collision pairs")

    # --- Summary ---
    t_physics = time.time()
    log(f"\n{'=' * 70}")
    log("CATEGORIZATION SUMMARY (all levels)")
    log("=" * 70)
    for cat in PrimCategory:
        log(f"  {cat.name:25s} {counts[cat]:6d}")
    log(f"  {'TOTAL':25s} {sum(counts.values()):6d}")

    log(f"\n{'=' * 70}")
    log("PER-LEVEL SUMMARY")
    log("=" * 70)
    log(f"  {'Level':<40s} {'Structural':>10s} {'Glazing':>10s} {'Stair':>10s} {'Door':>10s} {'Skip':>10s} {'Colliders':>10s}")
    log(f"  {'-' * 100}")
    for lname in sorted(level_counts.keys()):
        lc = level_counts[lname]
        log(f"  {lname:<40s} "
            f"{lc[PrimCategory.STATIC_STRUCTURAL]:10d} "
            f"{lc[PrimCategory.STATIC_GLAZING]:10d} "
            f"{lc[PrimCategory.STATIC_STAIR]:10d} "
            f"{lc[PrimCategory.DYNAMIC_DOOR]:10d} "
            f"{lc[PrimCategory.SKIP]:10d} "
            f"{level_colliders.get(lname, 0):10d}")

    log(f"\n{'=' * 70}")
    log("PHYSICS APPLICATION SUMMARY")
    log("=" * 70)
    log(f"  Static colliders applied:     {applied_collider}")
    log(f"  Dynamic rigid bodies applied:  {applied_rigid_body}")
    log(f"  Revolute joints created:       {joints_created}")
    log(f"  Drives created (hinged):       {drives_created}")
    log(f"  Drives skipped (non-hinged):   {drives_skipped}")
    log(f"  Skipped (no mesh found):       {skipped_no_mesh}")

    # --- Export physics USD ---
    output_usd = os.path.join(
        _project_dir, "assets", "environments", "msoe_building",
        "innovation_center_physics.usd"
    )
    os.makedirs(os.path.dirname(output_usd), exist_ok=True)

    log(f"\nExporting: {output_usd}")
    stage.Export(output_usd)
    log("Export complete.")

    file_size_mb = os.path.getsize(output_usd) / (1024 * 1024)
    log(f"File size: {file_size_mb:.1f} MB")

    t_end = time.time()
    log(f"\n{'=' * 70}")
    log("TIMING")
    log("=" * 70)
    log(f"  Physics application: {t_physics - t_start:.1f}s")
    log(f"  Total runtime:       {t_end - t_start:.1f}s")

    log(f"\n{'=' * 70}")
    log("Done!")
    log(f"  Report  -> {_report_path}")
    log(f"  Physics -> {output_usd}")
    log("=" * 70)

    _report_file.close()
    simulation_app.close()


def _log_level_summary(level_name, level_cat_counts, collider_count):
    """Log a brief summary line after finishing a level."""
    total = sum(level_cat_counts.values())
    static = (level_cat_counts.get(PrimCategory.STATIC_STRUCTURAL, 0)
              + level_cat_counts.get(PrimCategory.STATIC_GLAZING, 0)
              + level_cat_counts.get(PrimCategory.STATIC_STAIR, 0))
    doors = level_cat_counts.get(PrimCategory.DYNAMIC_DOOR, 0)
    skipped = level_cat_counts.get(PrimCategory.SKIP, 0)
    log(f"  -> {level_name}: {total} elements "
        f"({static} static, {doors} doors, {skipped} skipped, "
        f"{collider_count} colliders applied)")


if __name__ == "__main__":
    main()
