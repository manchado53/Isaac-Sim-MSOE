#!/usr/bin/env python3
"""
Generate 2D occupancy maps from InnovationCenter.usd using multi-origin scanning.

Scans from multiple origin points across the building to eliminate unknown
regions that a single-origin scan can't reach (areas behind walls).

Run with:
  sbatch submit_isaac_job.sh floorplan_vision/generate_occupancy_map.py
  OR (interactive):
  srun --partition=teaching --gres=gpu:1 --time=2:00:00 --pty bash
  ./run_isaac_interactive.sh floorplan_vision/generate_occupancy_map.py
"""

import os
import sys
import json
import argparse
import numpy as np

os.environ["OMNI_KIT_ACCEPT_EULA"] = "YES"

from isaacsim import SimulationApp
simulation_app = SimulationApp({"headless": True})

import omni
import omni.kit.app
import omni.timeline
from pxr import Gf, Usd, UsdGeom, UsdPhysics, PhysxSchema, Sdf

# ---------------------------------------------------------------------------
# Config
# ---------------------------------------------------------------------------
PROJECT_DIR = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.abspath(os.path.join(PROJECT_DIR, "..")))
from sim_utils.building_physics import (  # noqa: E402
    PrimCategory,
    apply_static_collider,
    categorize_prim,
    iter_building_elements,
)
# Write straight into the building-centric layout so downstream consumers
# (auto_graph_generator.py, Nav2 map_server, plot_graph_with_axes.py) read
# from a single canonical location.
OUTPUT_DIR = os.path.abspath(os.path.join(
    PROJECT_DIR, "..", "assets", "environments", "msoe_building", "maps"
))
USD_PATH = os.path.abspath(os.path.join(
    PROJECT_DIR, "..", "assets", "environments", "msoe_building",
    "MSOEBuildingV4_physics.usd"
))

CELL_SIZE = 0.05  # 5 cm per cell — V4 stage units are meters
OCCUPIED = 1
FREE = 0
UNKNOWN = 2

# Multi-origin grid: how many origins to place along each axis
GRID_ORIGINS_X = 5  # origins spread across X
GRID_ORIGINS_Y = 5  # origins spread across Y
MAX_RESCAN_ROUNDS = 6  # additional rounds targeting remaining unknown regions
RESCAN_ORIGINS_PER_ROUND = 50  # origins per rescan round (placed inside unknown regions)

os.makedirs(OUTPUT_DIR, exist_ok=True)


def scan_from_origin(generator, origin, min_bound, max_bound, cell_size):
    """Run a single occupancy map scan from the given origin."""
    generator.update_settings(cell_size, OCCUPIED, FREE, UNKNOWN)
    generator.set_transform(origin, min_bound, max_bound)
    simulation_app.update()
    generator.generate2d()
    simulation_app.update()

    buffer = generator.get_buffer()
    dims = generator.get_dimensions()

    if len(buffer) == 0 or dims[0] == 0 or dims[1] == 0:
        return None, dims

    grid = np.array(buffer, dtype=np.float32).reshape(dims[1], dims[0])
    return grid, dims


def merge_grids(base_grid, new_grid):
    """Merge a new scan into the base grid.
    Rules:
    - OCCUPIED in either scan -> OCCUPIED (walls are walls)
    - FREE in new scan overrides UNKNOWN in base (we found open space)
    - UNKNOWN doesn't override anything
    Handles grids with slightly different dimensions (due to FP rounding).
    """
    # Crop both grids to the minimum common size
    min_h = min(base_grid.shape[0], new_grid.shape[0])
    min_w = min(base_grid.shape[1], new_grid.shape[1])
    bg = base_grid[:min_h, :min_w]
    ng = new_grid[:min_h, :min_w]

    # Where base is unknown but new found something, use new
    unknown_mask = (bg == UNKNOWN)
    bg[unknown_mask & (ng == FREE)] = FREE
    bg[unknown_mask & (ng == OCCUPIED)] = OCCUPIED

    # Where new found occupied and base was free, mark occupied
    # (wall detected from a different angle)
    bg[(ng == OCCUPIED)] = OCCUPIED

    return base_grid


def find_unknown_origins(grid, world_min_x, world_min_y, cell_size, origin_z, num_origins=10):
    """Find origin points inside unknown regions for targeted rescanning."""
    unknown_coords = np.argwhere(grid == UNKNOWN)
    if len(unknown_coords) == 0:
        return []

    # Cluster unknown regions and pick spread-out points
    # Simple approach: divide unknown area into a grid and pick center of each cell
    origins = []
    n = min(num_origins, len(unknown_coords))

    # Pick evenly spaced points from the unknown coordinates
    indices = np.linspace(0, len(unknown_coords) - 1, n, dtype=int)
    for idx in indices:
        r, c = unknown_coords[idx]
        # Convert grid coords back to world coords (absolute)
        world_x = world_min_x + c * cell_size + cell_size / 2
        world_y = world_min_y + r * cell_size + cell_size / 2
        origins.append((world_x, world_y, origin_z))

    return origins


def save_floor(grid, dims, floor_name, meta_base):
    """Save a floor's occupancy grid as npy, png, and json metadata."""
    n_occupied = int(np.sum(grid == OCCUPIED))
    n_free = int(np.sum(grid == FREE))
    n_unknown = int(np.sum(grid == UNKNOWN))
    total = grid.size

    print(f"  Final: Occupied {n_occupied} ({100*n_occupied/total:.1f}%), "
          f"Free {n_free} ({100*n_free/total:.1f}%), "
          f"Unknown {n_unknown} ({100*n_unknown/total:.1f}%)")

    # Isaac Sim's occupancy map plugin produces a buffer where col 0 is at
    # world x_max and row 0 is at world y_min (see
    # isaacsim.asset.gen.omap.utils::compute_coordinates). We need to end up
    # with col 0 = x_min AND row 0 = y_max, which is the ROS map_server
    # convention: origin is the world pose of the BOTTOM-LEFT pixel, and
    # map_io.cpp internally reverses rows on load (map_io.cpp:326 "ROS
    # expects origin at bottom-left"). Fix both axes here, before the .npy
    # and .png writes, so everything downstream sees a consistent map.
    grid = np.fliplr(grid)   # col 0 = x_max -> col 0 = x_min
    grid = np.flipud(grid)   # row 0 = y_min -> row 0 = y_max

    # Save numpy array (post-flip)
    np.save(os.path.join(OUTPUT_DIR, f"{floor_name}_occupancy.npy"), grid)

    # Save PNG (built from the flipped grid)
    img_array = np.zeros((dims[1], dims[0], 3), dtype=np.uint8)
    img_array[grid == OCCUPIED] = [0, 0, 0]
    img_array[grid == FREE] = [255, 255, 255]
    img_array[grid == UNKNOWN] = [128, 128, 128]

    try:
        from PIL import Image
        Image.fromarray(img_array).save(
            os.path.join(OUTPUT_DIR, f"{floor_name}_occupancy.png"))
        print(f"  Saved: {floor_name}_occupancy.npy + .png")
    except ImportError:
        try:
            import matplotlib
            matplotlib.use('Agg')
            import matplotlib.pyplot as plt
            plt.imsave(os.path.join(OUTPUT_DIR, f"{floor_name}_occupancy.png"), img_array)
            print(f"  Saved: {floor_name}_occupancy.npy + .png (matplotlib)")
        except ImportError:
            np.save(os.path.join(OUTPUT_DIR, f"{floor_name}_occupancy_rgb.npy"), img_array)
            print(f"  Saved: {floor_name}_occupancy.npy + _rgb.npy")

    # Save metadata
    meta = {**meta_base,
            "occupied_count": n_occupied,
            "free_count": n_free,
            "unknown_count": n_unknown}
    with open(os.path.join(OUTPUT_DIR, f"{floor_name}_meta.json"), "w") as f:
        json.dump(meta, f, indent=2)


# ---------------------------------------------------------------------------
# Glass + door preparation (pre-physics)
# ---------------------------------------------------------------------------
# The _omap.Generator raycasts PhysX collision shapes, NOT visual materials,
# so OmniGlass alpha/opacity is irrelevant to whether glass blocks rays. Some
# glass prims in the Revit-exported USD are nested Xforms whose meshes aren't
# reached by a simple `prim.IsA(UsdGeom.Mesh)` traversal, so we run a
# dedicated pass using apply_static_collider(..., deep_search=True,
# all_meshes=True) to guarantee every glass panel contributes a collider.

GLASS_NAME_SUBSTRINGS = (
    "GL_",
    "Glass",
    "Curtain",
    "System_Panel_Glazed",
    "Glazing",
    "Storefront",
    "Sidelite",
)


def enumerate_operable_door_paths(stage):
    """Return set of prim paths belonging to operable (hinged) door leaves.

    Resolves each HingeJoint's body1 relationship to find the door prim and
    includes every descendant path. This is more reliable than a path-
    substring or fixed-depth parent walk — doors may live under any folder
    structure (Revit exports vary between Innovation Center / Engineering
    building layouts), and the joint's body1 pointer is authoritative.
    """
    operable_paths = set()
    door_roots = []
    for prim in stage.Traverse():
        if prim.GetName() != "HingeJoint":
            continue
        joint = UsdPhysics.RevoluteJoint.Get(stage, prim.GetPath())
        if not joint:
            continue
        body1_targets = joint.GetBody1Rel().GetTargets() if joint.GetBody1Rel() else []
        for target in body1_targets:
            door_prim = stage.GetPrimAtPath(target)
            if not door_prim or not door_prim.IsValid():
                continue
            door_roots.append(door_prim)
            for desc in Usd.PrimRange(door_prim):
                operable_paths.add(str(desc.GetPath()))
    return operable_paths, door_roots


def disable_operable_door_collision(stage, door_roots):
    """Turn off physics:collisionEnabled on every operable-door descendant.

    MSOEBuildingV4_physics.usd pre-bakes CollisionAPI on door leaves via
    sim_utils.building_physics.apply_dynamic_rigid_body. Without this pass,
    those colliders ride along as the doors rotate under DriveAPI control
    during the brief physics ticks before the first scan, leaving
    swing-arc blobs in the occupancy map. Disabling collision is enough —
    we don't need to remove the schema or touch the rigid body.

    Also flips each door's RigidBodyAPI to kinematic so even if residual
    collision lingers on some descendant we missed, the body itself can't
    move under forces.
    """
    collision_disabled = 0
    kinematic_set = 0
    for root in door_roots:
        # Kinematic flip — prevents any physics-driven motion of the door
        if root.HasAPI(UsdPhysics.RigidBodyAPI):
            rb = UsdPhysics.RigidBodyAPI(root)
            kin_attr = rb.GetKinematicEnabledAttr()
            if not kin_attr:
                kin_attr = rb.CreateKinematicEnabledAttr(True)
            else:
                kin_attr.Set(True)
            kinematic_set += 1
        # Disable collision on every descendant (door leaf + frame + glass)
        for desc in Usd.PrimRange(root):
            if desc.HasAPI(UsdPhysics.CollisionAPI):
                col = UsdPhysics.CollisionAPI(desc)
                enabled_attr = col.GetCollisionEnabledAttr()
                if not enabled_attr:
                    enabled_attr = col.CreateCollisionEnabledAttr(False)
                else:
                    enabled_attr.Set(False)
                collision_disabled += 1
    print(f"  Disabled collision on {collision_disabled} operable-door descendants, "
          f"flipped {kinematic_set} doors to kinematic")


def force_glass_colliders(stage, building_path="/World/Building",
                          operable_door_paths=None):
    """Apply a static collider to every glass panel in the building.

    Runs two passes:
      1. Category pass: iter_building_elements -> STATIC_GLAZING -> apply_static_collider
      2. Name pass: stage.Traverse() matching GLASS_NAME_SUBSTRINGS on the prim path,
         catching glazed elements exported under non-glass categories.
    """
    operable_door_paths = operable_door_paths or set()
    building_prim = stage.GetPrimAtPath(building_path)
    cat_hits = 0
    cat_applied = 0
    cat_door_skipped = 0
    if building_prim and building_prim.IsValid():
        for _level, category_name, element in iter_building_elements(stage, building_prim):
            if str(element.GetPath()) in operable_door_paths:
                cat_door_skipped += 1
                continue
            category = categorize_prim(element.GetName(), category_hint=category_name)
            if category is not PrimCategory.STATIC_GLAZING:
                continue
            cat_hits += 1
            if apply_static_collider(
                stage, element, approx="none", deep_search=True, all_meshes=True
            ):
                cat_applied += 1

    name_hits = 0
    name_applied = 0
    name_door_skipped = 0
    seen_paths = set()
    for prim in stage.Traverse():
        path_str = str(prim.GetPath())
        if path_str in seen_paths:
            continue
        # Skip anything that belongs to an operable door (identified by
        # HingeJoint -> body1 ancestry). Door leaves often carry glass
        # panels (door windows, curtain-wall doors). If we let the glass
        # pass apply collision to them, the collision rotates with the
        # door at scan time and leaves swing-arc artifacts in the map.
        if path_str in operable_door_paths:
            if any(sub in path_str for sub in GLASS_NAME_SUBSTRINGS):
                name_door_skipped += 1
            continue
        if not any(sub in path_str for sub in GLASS_NAME_SUBSTRINGS):
            continue
        if prim.IsA(UsdGeom.Mesh) or prim.IsA(UsdGeom.Xform):
            name_hits += 1
            if apply_static_collider(
                stage, prim, approx="none", deep_search=True, all_meshes=True
            ):
                name_applied += 1
                seen_paths.add(path_str)

    print(
        f"  Glass colliders forced: {cat_hits + name_hits} elements "
        f"(category pass {cat_applied}/{cat_hits} [door-skipped {cat_door_skipped}], "
        f"name pass {name_applied}/{name_hits} [door-skipped {name_door_skipped}])"
    )


def close_all_doors(stage):
    """Ensure every operable door starts at 0 degrees before physics ticks.

    Zeros:
      - UsdPhysics.DriveAPI angular targetPosition on every HingeJoint prim
      - PhysxSchema.JointStateAPI angular position if present
      - xformOp:orient / xformOp:rotateZ on door-leaf Xforms under /Doors/
        whose ancestor Xform carries an operable HingeJoint (same detection
        as the raycast-skip logic)
    """
    doors_closed = 0
    drives_reset = 0
    states_reset = 0
    xforms_reset = 0

    for prim in stage.Traverse():
        if prim.GetName() != "HingeJoint":
            continue
        doors_closed += 1

        drive = UsdPhysics.DriveAPI.Get(prim, "angular")
        if not drive or not drive.GetTargetPositionAttr():
            drive = UsdPhysics.DriveAPI.Apply(prim, "angular")
        target_attr = drive.GetTargetPositionAttr()
        if not target_attr:
            target_attr = drive.CreateTargetPositionAttr(0.0)
        else:
            target_attr.Set(0.0)
        drives_reset += 1

        try:
            state = PhysxSchema.JointStateAPI.Get(prim, "angular")
            if not state or not state.GetPositionAttr():
                state = PhysxSchema.JointStateAPI.Apply(prim, "angular")
            pos_attr = state.GetPositionAttr()
            if not pos_attr:
                pos_attr = state.CreatePositionAttr(0.0)
            else:
                pos_attr.Set(0.0)
            vel_attr = state.GetVelocityAttr() or state.CreateVelocityAttr(0.0)
            vel_attr.Set(0.0)
            states_reset += 1
        except Exception:
            pass

    identity_quat = Gf.Quatf(1.0, 0.0, 0.0, 0.0)
    identity_quatd = Gf.Quatd(1.0, 0.0, 0.0, 0.0)
    for prim in stage.Traverse():
        path_str = str(prim.GetPath())
        if "/Doors/" not in path_str:
            continue
        if not prim.IsA(UsdGeom.Xform):
            continue
        p = prim
        has_hinge = False
        for _ in range(6):
            p = p.GetParent()
            if not p or not p.IsValid():
                break
            if any(c.GetName() == "HingeJoint" for c in p.GetChildren()):
                has_hinge = True
                break
        if not has_hinge:
            continue

        xformable = UsdGeom.Xformable(prim)
        reset_any = False
        for op in xformable.GetOrderedXformOps():
            name = op.GetOpName()
            try:
                if name.endswith("xformOp:orient"):
                    if op.GetPrecision() == UsdGeom.XformOp.PrecisionDouble:
                        op.Set(identity_quatd)
                    else:
                        op.Set(identity_quat)
                    reset_any = True
                elif name.endswith("xformOp:rotateZ") or name.endswith("xformOp:rotateY") or name.endswith("xformOp:rotateX"):
                    op.Set(0.0)
                    reset_any = True
                elif name.endswith("xformOp:rotateXYZ") or name.endswith("xformOp:rotateZYX") or name.endswith("xformOp:rotateXZY") or name.endswith("xformOp:rotateYXZ") or name.endswith("xformOp:rotateYZX") or name.endswith("xformOp:rotateZXY"):
                    op.Set(Gf.Vec3f(0.0, 0.0, 0.0))
                    reset_any = True
            except Exception:
                continue
        if reset_any:
            xforms_reset += 1

    print(
        f"  Closed {doors_closed} operable doors "
        f"({drives_reset} drive targets reset, {states_reset} joint states reset, "
        f"{xforms_reset} xforms reset)"
    )


def main():
    print("=" * 60)
    print("Multi-Origin Occupancy Map Generator")
    print("=" * 60)
    sys.stdout.flush()

    # ------------------------------------------------------------------
    # Step 1: Load USD stage
    # ------------------------------------------------------------------
    print(f"\nLoading USD: {USD_PATH}")
    if not os.path.exists(USD_PATH):
        print(f"ERROR: USD file not found at {USD_PATH}")
        simulation_app.close()
        sys.exit(1)

    omni.usd.get_context().open_stage(USD_PATH)
    for i in range(3):
        simulation_app.update()

    stage = omni.usd.get_context().get_stage()
    if not stage:
        print("ERROR: Failed to open stage")
        simulation_app.close()
        sys.exit(1)

    up_axis = UsdGeom.GetStageUpAxis(stage)
    meters_per_unit = UsdGeom.GetStageMetersPerUnit(stage)
    prim_count = len(list(stage.Traverse()))
    print(f"  Stage loaded: {prim_count} prims, up={up_axis}, units={'cm' if meters_per_unit == 0.01 else 'm'}")
    sys.stdout.flush()

    # ------------------------------------------------------------------
    # Step 2a: Enumerate operable door prim paths (HingeJoint -> body1)
    # ------------------------------------------------------------------
    # Must run BEFORE the collision pass so we know which meshes to skip.
    # Joint body1 pointer is authoritative and works regardless of folder
    # structure (Innovation Center / Engineering building use different
    # Revit hierarchies — a fixed-depth parent walk misses some doors).
    print("\nEnumerating operable-door prim paths...")
    operable_door_paths, door_roots = enumerate_operable_door_paths(stage)
    print(f"  {len(door_roots)} HingeJoint -> body1 roots, "
          f"{len(operable_door_paths)} descendant paths marked as operable")
    sys.stdout.flush()

    # Disable pre-baked collision + flip to kinematic so rays pass through
    # doorways even if setup_building_physics.py already applied colliders
    # to door leaves, and so doors can't drift under physics.
    print("\nDisabling collision on operable doors...")
    disable_operable_door_collision(stage, door_roots)
    sys.stdout.flush()

    # ------------------------------------------------------------------
    # Step 2: Apply collision (only to architectural meshes, skip HVAC/steel)
    # ------------------------------------------------------------------
    print("\nApplying collision to architectural mesh prims...")
    # Skip non-architectural geometry that creates noise in occupancy maps
    # MSOEBuildingTake2.usd has no HVAC/structural steel meshes, but skip
    # Room volume meshes (Revit room labels) as they may interfere
    SKIP_PREFIXES = (
        "Room_",  # Revit room volume/label meshes
        "Model_Text",
    )
    mesh_count = 0
    skipped = 0
    door_skipped = 0
    door_kept = 0
    for prim in stage.Traverse():
        if prim.IsA(UsdGeom.Mesh):
            path_str = str(prim.GetPath())
            name = prim.GetName()
            # Door handling: skip OPERABLE doors (identified via
            # HingeJoint body1 descent) so raycasts pass through real
            # doorways. Non-hinged "doors" (sliding glass walls, sidelight
            # panels, double curtain-wall panels) are glass walls for
            # navigation purposes and must block raycasts.
            if path_str in operable_door_paths:
                door_skipped += 1
                continue
            if "/Doors/" in path_str:
                door_kept += 1
            if any(name.startswith(prefix) for prefix in SKIP_PREFIXES):
                skipped += 1
                continue
            mesh_count += 1
            if not prim.HasAPI(UsdPhysics.CollisionAPI):
                UsdPhysics.CollisionAPI.Apply(prim)
    print(f"  Meshes with collision: {mesh_count} (skipped {skipped} non-architectural, "
          f"{door_skipped} operable doors, kept {door_kept} non-hinged door meshes)")
    sys.stdout.flush()

    # ------------------------------------------------------------------
    # Step 2b: Force colliders on every glass prim (deep + name-based)
    # ------------------------------------------------------------------
    # The per-Mesh loop above misses glass wrapped in nested Xforms or hidden
    # inside instanced prototypes. Without this pass, curtain panels and
    # glazed partitions leave phantom corridors in the occupancy map.
    print("\nForcing colliders on glass panels...")
    force_glass_colliders(stage, operable_door_paths=operable_door_paths)
    sys.stdout.flush()

    # ------------------------------------------------------------------
    # Step 2c: Auto-close all operable doors
    # ------------------------------------------------------------------
    # timeline.play() + the two update ticks below would otherwise let any
    # door authored with a nonzero drive target or xform rotation swing open,
    # corrupting the raycast. Deterministically zero every HingeJoint drive
    # target, joint state, and door-leaf xform rotation.
    print("\nAuto-closing operable doors...")
    close_all_doors(stage)
    sys.stdout.flush()

    # ------------------------------------------------------------------
    # Step 2.5: Compute bounding box BEFORE starting physics.
    # ------------------------------------------------------------------
    # With dynamic rigid-body doors in the physics USD (129 of them), even two
    # update ticks after timeline.play() can send a badly-hinged door flying to
    # infinity, corrupting any world bbox sampled afterward. Capture the static
    # layout bounds first.
    print("\nComputing bounding box (pre-physics)...")
    bbox_cache_pre = UsdGeom.BBoxCache(Usd.TimeCode.Default(), ["default", "render"])
    world_prim_pre = stage.GetPrimAtPath("/World/Building") or stage.GetPrimAtPath("/World") or stage.GetPseudoRoot()
    bbox_pre = bbox_cache_pre.ComputeWorldBound(world_prim_pre)
    _bbox_pre_range = bbox_pre.ComputeAlignedRange()
    _pre_bb_min = _bbox_pre_range.GetMin()
    _pre_bb_max = _bbox_pre_range.GetMax()
    print(f"  Pre-physics Min: ({_pre_bb_min[0]:.2f}, {_pre_bb_min[1]:.2f}, {_pre_bb_min[2]:.2f})")
    print(f"  Pre-physics Max: ({_pre_bb_max[0]:.2f}, {_pre_bb_max[1]:.2f}, {_pre_bb_max[2]:.2f})")
    sys.stdout.flush()

    # ------------------------------------------------------------------
    # Step 3: Physics setup
    # ------------------------------------------------------------------
    print("\nSetting up physics...")
    physics_scene_path = "/World/physicsScene"
    if not stage.GetPrimAtPath(physics_scene_path):
        UsdPhysics.Scene.Define(stage, Sdf.Path(physics_scene_path))

    physx_scene = PhysxSchema.PhysxSceneAPI.Get(stage, physics_scene_path)
    if not physx_scene:
        physx_scene = PhysxSchema.PhysxSceneAPI.Apply(stage.GetPrimAtPath(physics_scene_path))
    physx_scene.CreateEnableGPUDynamicsAttr(False)
    physx_scene.CreateBroadphaseTypeAttr("MBP")

    simulation_app.update()
    simulation_app.update()

    timeline = omni.timeline.get_timeline_interface()
    timeline.play()
    simulation_app.update()
    simulation_app.update()
    print("  Physics ready, timeline playing")
    sys.stdout.flush()

    # ------------------------------------------------------------------
    # Step 4: Enable occupancy map extension
    # ------------------------------------------------------------------
    print("\nEnabling occupancy map extension...")
    ext_manager = omni.kit.app.get_app().get_extension_manager()
    ext_manager.set_extension_enabled_immediate("isaacsim.asset.gen.omap", True)
    simulation_app.update()
    from isaacsim.asset.gen.omap.bindings import _omap
    print("  Extension loaded")
    sys.stdout.flush()

    # ------------------------------------------------------------------
    # Step 5: Bounding box
    # ------------------------------------------------------------------
    print("\nComputing bounding box...")
    # Use the pre-physics bbox captured above — dynamic bodies may have drifted
    # during the physics ticks, so re-sampling now would be unreliable.
    bb_min = _pre_bb_min
    bb_max = _pre_bb_max

    print(f"  Min: ({bb_min[0]:.1f}, {bb_min[1]:.1f}, {bb_min[2]:.1f})")
    print(f"  Max: ({bb_max[0]:.1f}, {bb_max[1]:.1f}, {bb_max[2]:.1f})")
    print(f"  Size: ({bb_max[0]-bb_min[0]:.1f} x {bb_max[1]-bb_min[1]:.1f} x {bb_max[2]-bb_min[2]:.1f}) cm")

    bbox_info = {
        "min": [float(bb_min[0]), float(bb_min[1]), float(bb_min[2])],
        "max": [float(bb_max[0]), float(bb_max[1]), float(bb_max[2])],
        "meters_per_unit": meters_per_unit,
        "up_axis": up_axis,
        "cell_size": CELL_SIZE,
    }
    with open(os.path.join(OUTPUT_DIR, "bbox_info.json"), "w") as f:
        json.dump(bbox_info, f, indent=2)
    sys.stdout.flush()

    # ------------------------------------------------------------------
    # Step 6: Floor slices (based on floor_analysis.json from find_floor_levels.py)
    # ------------------------------------------------------------------
    # Floor slab Z levels from MSOEBuildingTake2.usd geometry analysis:
    #   Ground/basement slab:  Z ≈ 870 cm  (range 856-884)
    #   Floor 1 slab:          Z ≈ 1351 cm (range 1313-1387, large area 41M cm²)
    #   Floor 1 ceiling:       Z ≈ 1662 cm (range 1600-1701)
    #   Floor 2 slab:          Z ≈ 1797 cm (range 1732-1867)
    #   Floor 2 ceiling:       Z ≈ 2131 cm (range 2113-2139, large area 73M cm²)
    #   Floor 3 slab:          Z ≈ 2550 cm (range 2525-2566, large area 77M cm²)
    #   Roof structure:        Z ≈ 2982 cm (range 2936-3012)
    #   Top roof:              Z ≈ 3572 cm (range 3548-3597)
    # Walls: Z range 838-3597, clusters at Z-min ~860-870 and ~1200-1230
    #
    # Scan at robot height (~50-150cm above each floor slab)
    # Keep Z ranges NARROW to avoid hitting ceiling/slab geometry
    # V4 stage units are METERS (meters_per_unit=1.0).
    # Z ranges derived from find_floor_levels analysis of MSOEBuildingV4_physics.usd:
    #   Walls span Z 8.4–36.0 m, Doors 13.4–29.3 m, Ceiling 11.1–30.7 m
    # Scan at robot walking height (~0.5–1.5 m above each floor slab).
    all_floor_slices = [
        ("floor_1",      14.0, 16.0),   # floor 1 slab ~13.5 m, scan 0.5–2.5 m above
        ("floor_2",      19.5, 20.5),   # floor 2 slab ~19 m, walking height
        ("floor_3",      23.5, 24.5),   # floor 3 slab ~23 m, walking height
        ("floor_4",      27.5, 29.0),   # floor 4 slab ~27 m, walking height
    ]

    # Parse floor filter from OMAP_FLOOR env var or --floor command line arg
    # Usage: OMAP_FLOOR=floor_2 sbatch submit_isaac_job.sh ...
    #   OR:  ./python.sh script.py --floor floor_2
    floor_filter = os.environ.get("OMAP_FLOOR", None)
    if not floor_filter:
        for i, arg in enumerate(sys.argv):
            if arg == "--floor" and i + 1 < len(sys.argv):
                floor_filter = sys.argv[i + 1]
                break

    if floor_filter:
        floor_slices = [(n, zlo, zhi) for n, zlo, zhi in all_floor_slices
                        if n == floor_filter]
        if not floor_slices:
            valid = [n for n, _, _ in all_floor_slices]
            print(f"  ERROR: Unknown floor '{floor_filter}'. Valid: {valid}")
            simulation_app.close()
            sys.exit(1)
        print(f"\n  Processing single floor: {floor_filter}")
    else:
        floor_slices = all_floor_slices

    print(f"\n  Floor slices (from geometry analysis):")
    for name, z_lo, z_hi in floor_slices:
        print(f"  {name}: Z [{z_lo:.0f}, {z_hi:.0f}] cm")
    sys.stdout.flush()

    # ------------------------------------------------------------------
    # Step 7: Multi-origin occupancy map generation
    # ------------------------------------------------------------------
    print("\n" + "=" * 60)
    print("Generating multi-origin occupancy maps...")
    print("=" * 60)

    physx = omni.physx.get_physx_interface()
    stage_id = omni.usd.get_context().get_stage_id()

    padding = 1.0  # 1 m padding around building bounds (V4 units are meters)
    # World-space bounds (absolute) for the scan area
    world_min_x = bb_min[0] - padding
    world_min_y = bb_min[1] - padding
    world_max_x = bb_max[0] + padding
    world_max_y = bb_max[1] + padding

    # Pre-compute the expected grid dimensions from global bounds
    expected_width = int(np.ceil((world_max_x - world_min_x) / CELL_SIZE))
    expected_height = int(np.ceil((world_max_y - world_min_y) / CELL_SIZE))
    print(f"\n  Expected grid size: {expected_width} x {expected_height} "
          f"(from global XY bounds)")

    # Pre-compute the grid of origin points spread across the building
    origin_xs = np.linspace(bb_min[0] + 50, bb_max[0] - 50, GRID_ORIGINS_X)
    origin_ys = np.linspace(bb_min[1] + 50, bb_max[1] - 50, GRID_ORIGINS_Y)

    for floor_name, z_min, z_max in floor_slices:
        print(f"\n{'='*40}")
        print(f"  {floor_name} (Z: {z_min:.0f} to {z_max:.0f} cm)")
        print(f"{'='*40}")
        sys.stdout.flush()

        origin_z = (z_min + z_max) / 2.0

        # --- Pre-allocate fixed-size grid from global bounds ---
        base_grid = np.full((expected_height, expected_width), UNKNOWN,
                            dtype=np.float32)
        scan_count = 0

        print(f"\n  Round 1: Scanning from {GRID_ORIGINS_X}x{GRID_ORIGINS_Y} = "
              f"{GRID_ORIGINS_X * GRID_ORIGINS_Y} grid origins...")
        sys.stdout.flush()

        for ox in origin_xs:
            for oy in origin_ys:
                origin = (float(ox), float(oy), origin_z)
                # IMPORTANT: min_bound and max_bound are RELATIVE to origin
                # so we compute offsets to always cover the same world region
                rel_min = (world_min_x - float(ox),
                           world_min_y - float(oy),
                           z_min - origin_z)
                rel_max = (world_max_x - float(ox),
                           world_max_y - float(oy),
                           z_max - origin_z)
                generator = _omap.Generator(physx, stage_id)
                grid, dims = scan_from_origin(
                    generator, origin, rel_min, rel_max, CELL_SIZE)

                if grid is None:
                    continue

                scan_count += 1
                base_grid = merge_grids(base_grid, grid)

        if scan_count == 0:
            print(f"  WARNING: No valid scans for {floor_name}")
            continue

        n_unknown = int(np.sum(base_grid == UNKNOWN))
        total = base_grid.size
        print(f"  After {scan_count} scans: "
              f"{100*n_unknown/total:.1f}% unknown remaining")
        sys.stdout.flush()

        # --- Rounds 2+: Target remaining unknown regions ---
        prev_unknown_pct = 100 * n_unknown / total
        for round_num in range(MAX_RESCAN_ROUNDS):
            n_unknown = int(np.sum(base_grid == UNKNOWN))
            if n_unknown == 0:
                print(f"  No unknown regions left, skipping further rounds")
                break

            unknown_pct = 100 * n_unknown / total
            if unknown_pct < 0.5:
                print(f"  Only {unknown_pct:.1f}% unknown, good enough")
                break

            # Stagnation detection: if unknown % didn't improve by at least 0.1%
            # from the previous round, stop — these are likely unreachable cells
            if round_num > 0 and (prev_unknown_pct - unknown_pct) < 0.1:
                print(f"  Stagnation detected: {unknown_pct:.1f}% unknown "
                      f"(was {prev_unknown_pct:.1f}%), stopping rescans")
                break

            # Find origins inside unknown regions
            extra_origins = find_unknown_origins(
                base_grid, world_min_x, world_min_y, CELL_SIZE, origin_z,
                num_origins=RESCAN_ORIGINS_PER_ROUND)

            if not extra_origins:
                break

            print(f"\n  Round {round_num + 2}: Scanning from {len(extra_origins)} "
                  f"origins in unknown regions ({unknown_pct:.1f}% unknown)...")
            sys.stdout.flush()

            for origin in extra_origins:
                ox, oy, oz = origin
                rel_min = (world_min_x - ox, world_min_y - oy, z_min - oz)
                rel_max = (world_max_x - ox, world_max_y - oy, z_max - oz)
                generator = _omap.Generator(physx, stage_id)
                grid, _ = scan_from_origin(
                    generator, origin, rel_min, rel_max, CELL_SIZE)

                if grid is not None:
                    base_grid = merge_grids(base_grid, grid)
                    scan_count += 1

            prev_unknown_pct = unknown_pct
            n_unknown = int(np.sum(base_grid == UNKNOWN))
            print(f"  After {scan_count} total scans: "
                  f"{100*n_unknown/total:.1f}% unknown remaining")
            sys.stdout.flush()

        # --- Save final merged result ---
        dims = (base_grid.shape[1], base_grid.shape[0])  # (width, height)
        meta_base = {
            "floor_name": floor_name,
            "grid_width": dims[0],
            "grid_height": dims[1],
            "cell_size_cm": CELL_SIZE,
            "z_min": z_min,
            "z_max": z_max,
            "xy_min": [world_min_x, world_min_y],
            "xy_max": [world_max_x, world_max_y],
            "total_scans": scan_count,
        }
        save_floor(base_grid, dims, floor_name, meta_base)
        sys.stdout.flush()

    # ------------------------------------------------------------------
    # Done
    # ------------------------------------------------------------------
    timeline.stop()

    print("\n" + "=" * 60)
    print("Multi-origin occupancy map generation complete!")
    print(f"Output: {OUTPUT_DIR}")
    for f in sorted(os.listdir(OUTPUT_DIR)):
        size_kb = os.path.getsize(os.path.join(OUTPUT_DIR, f)) / 1024
        print(f"  {f} ({size_kb:.1f} KB)")
    print("=" * 60)
    sys.stdout.flush()

    simulation_app.close()


if __name__ == "__main__":
    main()
