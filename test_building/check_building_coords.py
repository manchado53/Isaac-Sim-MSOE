#!/usr/bin/env python3
"""
Check building USD position/scale against occupancy map metadata.
Run on a GPU node: sbatch submit_isaac_job.sh test_building/check_building_coords.py
"""
import os, sys, json
os.environ["OMNI_KIT_ACCEPT_EULA"] = "YES"

from isaacsim import SimulationApp
simulation_app = SimulationApp({"headless": True})

from pxr import Usd, UsdGeom, Gf

# Paths
SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
USD_PATH = os.path.join(SCRIPT_DIR, "..", "assets", "environments", "msoe_building", "MSOEBuildingTake2.usd")
META_PATH = os.path.join(SCRIPT_DIR, "output", "floor_4_meta.json")
RESULT_PATH = os.path.join(SCRIPT_DIR, "coord_check_results.txt")

# Write all output to file since Isaac Sim swallows stdout
out_lines = []
def log(msg=""):
    out_lines.append(msg)
    sys.stderr.write(msg + "\n")

# Load occupancy map metadata
with open(META_PATH) as f:
    meta = json.load(f)

log("=" * 70)
log("OCCUPANCY MAP METADATA (from scan)")
log("=" * 70)
log(f"  xy_min: [{meta['xy_min'][0]:.4f}, {meta['xy_min'][1]:.4f}]")
log(f"  xy_max: [{meta['xy_max'][0]:.4f}, {meta['xy_max'][1]:.4f}]")
log(f"  grid_size: {meta['grid_width']} x {meta['grid_height']}")
log(f"  cell_size_cm: {meta['cell_size_cm']}")
log(f"  z_min: {meta.get('z_min', 'N/A')}, z_max: {meta.get('z_max', 'N/A')}")
log()

# Open USD stage
stage = Usd.Stage.Open(USD_PATH)
if not stage:
    log("ERROR: Could not open USD file:", USD_PATH)
    simulation_app.close()
    sys.exit(1)

log("=" * 70)
log("USD FILE ANALYSIS")
log("=" * 70)
log(f"  File: {USD_PATH}")
log()

# Check root prims and their transforms
log("--- Root Prims and Transforms ---")
root = stage.GetPseudoRoot()
for prim in root.GetChildren():
    log(f"\n  Prim: {prim.GetPath()}")
    log(f"  Type: {prim.GetTypeName()}")

    # Check for xformable (has transform)
    xformable = UsdGeom.Xformable(prim)
    if xformable:
        xform_ops = xformable.GetOrderedXformOps()
        if xform_ops:
            log(f"  Xform ops ({len(xform_ops)}):")
            for op in xform_ops:
                val = op.Get()
                log(f"    {op.GetOpType()}: {val}")

        # Get local transform matrix
        local_xform = xformable.ComputeLocalToWorldTransform(Usd.TimeCode.Default())
        translate = local_xform.ExtractTranslation()
        log(f"  World Translation: ({translate[0]:.4f}, {translate[1]:.4f}, {translate[2]:.4f})")

        # Extract scale from matrix
        import math
        row0 = [local_xform[0][0], local_xform[0][1], local_xform[0][2]]
        row1 = [local_xform[1][0], local_xform[1][1], local_xform[1][2]]
        row2 = [local_xform[2][0], local_xform[2][1], local_xform[2][2]]
        sx = math.sqrt(sum(v*v for v in row0))
        sy = math.sqrt(sum(v*v for v in row1))
        sz = math.sqrt(sum(v*v for v in row2))
        log(f"  Scale: ({sx:.6f}, {sy:.6f}, {sz:.6f})")

    # Check children for transforms too
    for child in prim.GetChildren():
        child_xform = UsdGeom.Xformable(child)
        if child_xform:
            ops = child_xform.GetOrderedXformOps()
            if ops:
                log(f"\n  Child: {child.GetPath()}")
                log(f"  Type: {child.GetTypeName()}")
                for op in ops:
                    val = op.Get()
                    log(f"    {op.GetOpType()}: {val}")

# Compute world bounding box
log("\n--- World Bounding Box ---")
bbox_cache = UsdGeom.BBoxCache(Usd.TimeCode.Default(), [UsdGeom.Tokens.default_])
root_prim = stage.GetPseudoRoot()

# Try to get bounding box of the whole stage
total_bbox = Gf.BBox3d()
for prim in root_prim.GetChildren():
    try:
        prim_bbox = bbox_cache.ComputeWorldBound(prim)
        total_bbox = Gf.BBox3d.Combine(total_bbox, prim_bbox)
    except Exception as e:
        log(f"  Warning: Could not compute bbox for {prim.GetPath()}: {e}")

bbox_range = total_bbox.ComputeAlignedRange()
bb_min = bbox_range.GetMin()
bb_max = bbox_range.GetMax()

log(f"  USD BBox Min: ({bb_min[0]:.4f}, {bb_min[1]:.4f}, {bb_min[2]:.4f})")
log(f"  USD BBox Max: ({bb_max[0]:.4f}, {bb_max[1]:.4f}, {bb_max[2]:.4f})")
log(f"  USD Size: ({bb_max[0]-bb_min[0]:.2f} x {bb_max[1]-bb_min[1]:.2f} x {bb_max[2]-bb_min[2]:.2f})")

# Compare with occupancy metadata
log("\n" + "=" * 70)
log("COMPARISON: USD vs Occupancy Map")
log("=" * 70)

# The occupancy map adds 100cm padding
padding = 100.0
occ_building_min_x = meta['xy_min'][0] + padding
occ_building_min_y = meta['xy_min'][1] + padding
occ_building_max_x = meta['xy_max'][0] - padding
occ_building_max_y = meta['xy_max'][1] - padding

log(f"  Occupancy (minus 100cm padding):")
log(f"    X: [{occ_building_min_x:.2f}, {occ_building_max_x:.2f}]")
log(f"    Y: [{occ_building_min_y:.2f}, {occ_building_max_y:.2f}]")
log(f"  Current USD bounding box:")
log(f"    X: [{bb_min[0]:.2f}, {bb_max[0]:.2f}]")
log(f"    Y: [{bb_min[1]:.2f}, {bb_max[1]:.2f}]")
log(f"    Z: [{bb_min[2]:.2f}, {bb_max[2]:.2f}]")

offset_x = bb_min[0] - occ_building_min_x
offset_y = bb_min[1] - occ_building_min_y
log(f"\n  OFFSET (USD - Occupancy):")
log(f"    X offset: {offset_x:.2f} cm")
log(f"    Y offset: {offset_y:.2f} cm")

if abs(offset_x) > 10 or abs(offset_y) > 10:
    log(f"\n  *** MISMATCH DETECTED ***")
    log(f"  The building appears to have moved since the occupancy map was generated.")
    log(f"  To fix node coordinates, add ({offset_x:.2f}, {offset_y:.2f}) cm to all world_xy_cm values,")
    log(f"  or re-run the occupancy map generator.")
else:
    log(f"\n  Coordinates match within 10cm tolerance. No repositioning detected.")

# Check scale
occ_size_x = occ_building_max_x - occ_building_min_x
occ_size_y = occ_building_max_y - occ_building_min_y
usd_size_x = bb_max[0] - bb_min[0]
usd_size_y = bb_max[1] - bb_min[1]

scale_x = usd_size_x / occ_size_x if occ_size_x > 0 else 0
scale_y = usd_size_y / occ_size_y if occ_size_y > 0 else 0

log(f"\n  SCALE CHECK:")
log(f"    Occupancy size: {occ_size_x:.2f} x {occ_size_y:.2f} cm")
log(f"    USD size:       {usd_size_x:.2f} x {usd_size_y:.2f} cm")
log(f"    Scale ratio X:  {scale_x:.6f}")
log(f"    Scale ratio Y:  {scale_y:.6f}")

if abs(scale_x - 1.0) > 0.01 or abs(scale_y - 1.0) > 0.01:
    log(f"\n  *** SCALE MISMATCH DETECTED ***")
    log(f"  The building appears to have been rescaled since the occupancy map was generated.")
else:
    log(f"\n  Scale matches within 1% tolerance. No rescaling detected.")

log("\n" + "=" * 70)
log("DONE")
log("=" * 70)

# Write results to file
with open(RESULT_PATH, "w") as f:
    f.write("\n".join(out_lines))
log(f"\nResults written to: {RESULT_PATH}")

simulation_app.close()
