#!/usr/bin/env python3
"""
Generate 2D occupancy maps from TestBuilding.usd using multi-origin scanning.

Scans from multiple origin points across the building to eliminate unknown
regions that a single-origin scan can't reach (areas behind walls).

Run with:
  sbatch submit_isaac_job.sh test_building/generate_occupancy_map.py
  OR (interactive):
  srun --partition=teaching --gres=gpu:1 --time=2:00:00 --pty bash
  ./run_isaac_interactive.sh test_building/generate_occupancy_map.py
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
from pxr import Usd, UsdGeom, UsdPhysics, PhysxSchema, Sdf

# ---------------------------------------------------------------------------
# Config
# ---------------------------------------------------------------------------
PROJECT_DIR = os.path.dirname(os.path.abspath(__file__))
OUTPUT_DIR = os.path.join(PROJECT_DIR, "output")
USD_PATH = os.path.join(os.path.dirname(PROJECT_DIR),
                        "assets", "environments", "msoe_building", "MSOEBuildingTake2.usd")

CELL_SIZE = 3.0  # 3 cm per cell (finer detail)
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

    # Save numpy array
    np.save(os.path.join(OUTPUT_DIR, f"{floor_name}_occupancy.npy"), grid)

    # Save PNG
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
    for prim in stage.Traverse():
        if prim.IsA(UsdGeom.Mesh):
            path_str = str(prim.GetPath())
            name = prim.GetName()
            # Skip door geometry so raycasts pass through doorways into rooms
            if "/Doors/" in path_str:
                door_skipped += 1
                continue
            if any(name.startswith(prefix) for prefix in SKIP_PREFIXES):
                skipped += 1
                continue
            mesh_count += 1
            if not prim.HasAPI(UsdPhysics.CollisionAPI):
                UsdPhysics.CollisionAPI.Apply(prim)
    print(f"  Meshes with collision: {mesh_count} (skipped {skipped} non-architectural, "
          f"{door_skipped} door prims)")
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
    bbox_cache = UsdGeom.BBoxCache(Usd.TimeCode.Default(), ["default", "render"])
    world_prim = stage.GetPrimAtPath("/World")
    if not world_prim:
        world_prim = stage.GetPseudoRoot()

    bbox = bbox_cache.ComputeWorldBound(world_prim)
    bbox_range = bbox.ComputeAlignedRange()
    bb_min = bbox_range.GetMin()
    bb_max = bbox_range.GetMax()

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
    all_floor_slices = [
        ("basement",      900.0, 1100.0),   # above ground slab (884), walking height, below floor 1 slab (1313)
        ("floor_1",      1400.0, 1600.0),   # above floor 1 slab (1387), below ceiling (1600)
        ("floor_2",      1950.0, 2050.0),   # walking height above floor 2 slab (1867), below ceiling (2113)
        ("floor_3",      2350.0, 2450.0),   # ECBE rooms (Z 2250-2570), walking height
        ("floor_4",      2750.0, 2900.0),   # CAECM rooms (Z 2677-2997), walking height above slab
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

    padding = 100.0
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
