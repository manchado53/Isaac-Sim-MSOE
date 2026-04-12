#!/usr/bin/env python3
"""
Generic Building Physics Setup

Loads a Revit-exported USD, applies physics collisions, materials, and joints,
then exports a physics-ready USD.

Usage:
    sbatch submit_isaac_job.sh simulations/environment/setup_building_physics.py \
        --input assets/environments/msoe_building/MSOEBuildingTake2.usd

    sbatch submit_isaac_job.sh simulations/environment/setup_building_physics.py \
        --input assets/environments/msoe_building/MyBuilding.usd \
        --output assets/environments/msoe_building/MyBuilding_physics.usd \
        --margin 1.5

SLURM resource requirements:
    #SBATCH --time=04:00:00
    #SBATCH --mem=32G
"""

import argparse
import json
import os
import sys
import time

# Parse CLI args BEFORE SimulationApp (which takes ~40s and needs GPU).
# Use parse_known_args to avoid conflicts with Omniverse Kit's argv injection.
parser = argparse.ArgumentParser(description="Apply physics to a Revit-exported USD building.")
parser.add_argument("--input", required=True, help="Path to raw building USD file")
parser.add_argument("--output", default=None, help="Output physics USD path (default: <input_stem>_physics.usd)")
parser.add_argument("--margin", type=float, default=1.0, help="Door collision filter margin in meters (default: 1.0)")
parser.add_argument("--use-pairwise-filter", action="store_true",
                    help="Use legacy pairwise collision filtering instead of collision groups")
parser.add_argument("--report", default=None, help="Report log path (default: logs/<stem>_physics_setup.txt)")
args, _unknown = parser.parse_known_args()

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
    setup_collision_groups,
    create_placeholder_door_mesh,
    fix_absolute_asset_paths,
    CATEGORY_MATERIALS,
    DOOR_DRIVE_PARAMS,
    iter_building_elements,
)

# Resolve paths
input_usd = os.path.abspath(args.input)
stem = os.path.splitext(os.path.basename(input_usd))[0]

if args.output:
    output_usd = os.path.abspath(args.output)
else:
    output_usd = os.path.join(os.path.dirname(input_usd), f"{stem}_physics.usd")

if args.report:
    report_path = os.path.abspath(args.report)
else:
    report_path = os.path.join(_project_dir, "logs", f"{stem}_physics_setup.txt")

os.makedirs(os.path.dirname(report_path), exist_ok=True)
os.makedirs(os.path.dirname(output_usd), exist_ok=True)

_report_file = None


def log(msg=""):
    print(msg, flush=True)
    if _report_file:
        _report_file.write(msg + "\n")
        _report_file.flush()


def _log_level_summary(level_name, level_cat_counts, collider_count):
    total = sum(level_cat_counts.values())
    static = (level_cat_counts.get(PrimCategory.STATIC_STRUCTURAL, 0)
              + level_cat_counts.get(PrimCategory.STATIC_GLAZING, 0)
              + level_cat_counts.get(PrimCategory.STATIC_STAIR, 0))
    doors = level_cat_counts.get(PrimCategory.DYNAMIC_DOOR, 0)
    skipped = level_cat_counts.get(PrimCategory.SKIP, 0)
    log(f"  -> {level_name}: {total} elements "
        f"({static} static, {doors} doors, {skipped} skipped, "
        f"{collider_count} colliders applied)")


def main():
    global _report_file
    _report_file = open(report_path, "w")
    t_start = time.time()

    log("=" * 70)
    log("BUILDING PHYSICS SETUP")
    log("=" * 70)
    log(f"Input:  {input_usd}")
    log(f"Output: {output_usd}")
    log(f"Margin: {args.margin}m")

    if not os.path.exists(input_usd):
        log(f"\nERROR: Input USD not found: {input_usd}")
        _report_file.close()
        simulation_app.close()
        sys.exit(1)

    # --- Create world and load building ---
    world = World(stage_units_in_meters=1.0)
    stage = omni.usd.get_context().get_stage()

    log(f"\nLoading: {input_usd}")
    add_reference_to_stage(input_usd, prim_path="/World/Building")
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

    counts = {cat: 0 for cat in PrimCategory}
    applied_collider = 0
    applied_rigid_body = 0
    skipped_no_mesh = 0
    door_prims = []

    level_counts = {}
    level_colliders = {}
    current_level = None

    for level_name, category_name, element in iter_building_elements(stage, building_prim):
        if level_name != current_level:
            if current_level is not None:
                _log_level_summary(current_level, level_counts.get(current_level, {}),
                                   level_colliders.get(current_level, 0))
            current_level = level_name
            log(f"\n--- Level: {level_name} ---")

        if level_name not in level_counts:
            level_counts[level_name] = {cat: 0 for cat in PrimCategory}
            level_colliders[level_name] = 0

        element_name = element.GetName()
        category = categorize_prim(element_name, category_hint=category_name)
        counts[category] += 1
        level_counts[level_name][category] += 1

        if category == PrimCategory.SKIP:
            continue

        if category == PrimCategory.DYNAMIC_DOOR:
            # Skip doors with geometry that prevents any rotation
            if element.GetName() in ("Doors_13182085", "Doors_13606466"):
                counts[PrimCategory.SKIP] += 1
                level_counts[level_name][PrimCategory.SKIP] = level_counts[level_name].get(PrimCategory.SKIP, 0) + 1
                log(f"  Skipping jammed door: {element.GetName()}")
                continue
            mechanism = classify_door_mechanism(element)
            if mechanism != DoorMechanism.HINGED_SINGLE:
                counts[PrimCategory.SKIP] += 1
                level_counts[level_name][PrimCategory.SKIP] = level_counts[level_name].get(PrimCategory.SKIP, 0) + 1
                if mechanism == DoorMechanism.HINGED_DOUBLE:
                    log(f"  Skipping double door: {element.GetName()}")
                else:
                    log(f"  Skipping non-hinged door ({mechanism.name}): {element.GetName()}")
                continue
            door_mass = 15.0
            if apply_dynamic_rigid_body(stage, element, mass=door_mass, deep_search=True):
                applied_rigid_body += 1
                door_prims.append((level_name, element, mechanism))
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

        hinge_result = estimate_door_hinge_offset(stage, door)
        local_pos0, local_pos1 = hinge_result[0], hinge_result[1]
        local_rot0 = hinge_result[2] if len(hinge_result) > 2 else None
        local_rot1 = hinge_result[3] if len(hinge_result) > 3 else None
        if local_pos0 is None:
            log(f"  WARNING: Could not estimate hinge for {door_path} — skipping joint")
            joints_skipped += 1
            continue

        if mechanism == DoorMechanism.HINGED_DOUBLE:
            lower_limit, upper_limit = -70.0, 70.0
        else:
            lower_limit, upper_limit = -100.0, 100.0

        joint = create_revolute_joint(
            stage,
            joint_path=joint_path,
            body0_path="",
            body1_path=door_path,
            axis="Z",
            lower_limit=lower_limit,
            upper_limit=upper_limit,
            local_pos0=local_pos0,
            local_pos1=local_pos1,
            local_rot0=local_rot0,
            local_rot1=local_rot1,
        )
        joints_created += 1
        log(f"  Joint ({mechanism.name}): {door_path}")

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
    # Use pairwise margin filtering — single doors in tight Revit frames need
    # per-door collision disabling with nearby walls/mullions. Collision groups
    # alone are too coarse (they disable ALL door-static pairs globally).
    door_prim_list = [door for _, door, _ in door_prims]
    log(f"\nApplying collision filters for {len(door_prims)} doors (margin={args.margin}m)...")
    filtered = add_door_collision_filters(stage, door_prim_list, margin=args.margin)
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

    # --- Write manifest JSON (element counts for preservation verification) ---
    mesh_count = sum(1 for p in stage.Traverse() if p.IsA(UsdGeom.Mesh))

    building_prim_for_bbox = stage.GetPrimAtPath("/World/Building")
    bbox_cache = UsdGeom.BBoxCache(Usd.TimeCode.Default(), [UsdGeom.Tokens.default_])
    bbox = bbox_cache.ComputeWorldBound(building_prim_for_bbox)
    box_range = bbox.ComputeAlignedRange()
    bb_min = box_range.GetMin()
    bb_max = box_range.GetMax()

    manifest = {
        "source_usd": os.path.basename(input_usd),
        "total_elements": sum(counts.values()),
        "mesh_count": mesh_count,
        "bounding_box": {
            "min": [float(bb_min[i]) for i in range(3)],
            "max": [float(bb_max[i]) for i in range(3)],
        },
        "categories": {cat.name: count for cat, count in counts.items()},
        "per_level": {
            lname: {cat.name: lc[cat] for cat in PrimCategory}
            for lname, lc in level_counts.items()
        },
        "physics_applied": {
            "static_colliders": applied_collider,
            "dynamic_rigid_bodies": applied_rigid_body,
            "revolute_joints": joints_created,
            "drives": drives_created,
            "collision_filter_pairs": filtered,
            "skipped_no_mesh": skipped_no_mesh,
        },
    }

    manifest_path = os.path.splitext(output_usd)[0] + "_manifest.json"
    with open(manifest_path, "w") as mf:
        json.dump(manifest, mf, indent=2)
    log(f"\nManifest: {manifest_path}")

    # --- Fix absolute asset paths for portability (pre-export pass) ---
    fixed_paths = fix_absolute_asset_paths(stage)
    if fixed_paths:
        log(f"\nFixed {fixed_paths} absolute asset paths -> relative (pre-export)")

    # --- Export physics USD ---
    log(f"\nExporting: {output_usd}")
    stage.Export(output_usd)
    log("Export complete.")

    # --- Post-export pass: stage.Export() can re-absolutize relative MDL paths ---
    # Re-open the exported file and fix any absolute paths that were baked in.
    log("\nPost-export portability check...")
    exported_stage = Usd.Stage.Open(output_usd)
    post_fixed = fix_absolute_asset_paths(exported_stage)
    if post_fixed:
        exported_stage.GetRootLayer().Save()
        log(f"Fixed {post_fixed} absolute asset paths -> relative (post-export)")
    else:
        log("No absolute paths found in exported file — already portable.")
    del exported_stage

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
    log(f"  Report   -> {report_path}")
    log(f"  Physics  -> {output_usd}")
    log(f"  Manifest -> {manifest_path}")
    log("=" * 70)

    _report_file.close()
    simulation_app.close()


if __name__ == "__main__":
    main()
