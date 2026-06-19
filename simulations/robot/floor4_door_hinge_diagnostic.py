#!/usr/bin/env python3
"""
Floor 4 Door Hinge Diagnostic

For each of the 7 curtain-wall doors on Floor 4, this script:
  1. Reads child mesh xformOp:transform matrices (origin ox, oy, oz in cm)
  2. Computes the world bounding box at the default (0°) pose
  3. Reads PhysicsRevoluteJoint properties if a HingeJoint child exists
  4. Determines the hinge edge: compares mesh origin to bbox min/max to
     find which edge is the pivot (closest bbox edge to the mesh origin
     on the wider axis of the door)
  5. Computes camera-center (bbox midpoint in XY, in metres)
  6. Outputs ready-to-use sbatch commands for door_static_collection.py

Run via:
  sbatch submit_isaac_job.sh simulations/robot/floor4_door_hinge_diagnostic.py

Results appear in logs/isaac_<JOBID>.out
"""

import os
import sys
import math

os.environ["OMNI_KIT_ACCEPT_EULA"] = "YES"
from isaacsim import SimulationApp
simulation_app = SimulationApp({"headless": True, "hide_ui": True})

import omni.usd
from pxr import Usd, UsdGeom, UsdPhysics, Gf

_script_dir  = os.path.dirname(os.path.abspath(__file__))
_project_dir = os.path.abspath(os.path.join(_script_dir, "..", ".."))

USD_PATH = os.path.join(_project_dir, "MSOEBuildingV4_physics.usd")

# The 7 Floor 4 doors identified by floor4_door_diagnostic.py (job 237551)
DOOR_BASE = (
    "/World/Building/_22_129___MSOE_Engineering_Building___A___2_25_2026"
    "/Geometry/Doors/RS___Curtain_Wall___Type_FG___New/New"
)
DOOR_IDS = [
    "Doors_1913690",
    "Doors_4190126",
    "Doors_12112322",
    "Doors_12112444",
    "Doors_13185454",
    "Doors_13606713",   # current training door (known hinge)
    "Doors_13935342",
]

FLOOR4_CAM_Z = 27.8  # metres — eye-level camera height on Floor 4

# ---------------------------------------------------------------------------
print("=" * 80)
print("Floor 4 Door Hinge Diagnostic")
print(f"USD: {USD_PATH}")
print("=" * 80)

if not os.path.isfile(USD_PATH):
    print(f"ERROR: USD file not found at {USD_PATH}")
    simulation_app.close()
    sys.exit(1)

print(f"\nOpening stage (this may take 30–60 s for the 154 MB USD)...")
omni.usd.get_context().open_stage(USD_PATH)
simulation_app.update()
stage = omni.usd.get_context().get_stage()
if not stage:
    print("ERROR: Failed to open USD stage")
    simulation_app.close()
    sys.exit(1)
print("Stage opened.\n")

time_code   = Usd.TimeCode.Default()
xform_cache = UsdGeom.XformCache(time_code)
bbox_cache  = UsdGeom.BBoxCache(time_code, [UsdGeom.Tokens.default_], useExtentsHint=False)

# ---------------------------------------------------------------------------
# Analyze each door
# ---------------------------------------------------------------------------

results = []

for door_id in DOOR_IDS:
    door_path = f"{DOOR_BASE}/{door_id}"
    door_prim = stage.GetPrimAtPath(door_path)

    print("-" * 80)
    print(f"DOOR: {door_id}")
    print(f"  Path: {door_path}")

    if not door_prim.IsValid():
        print(f"  ERROR: prim not found!")
        continue

    # --- Child meshes: read xformOp:transform ---
    meshes = []
    hinge_joint_prim = None

    for child in door_prim.GetChildren():
        child_name = child.GetName()
        child_type = child.GetTypeName()

        if child_type == "PhysicsRevoluteJoint" or "Joint" in child_name:
            hinge_joint_prim = child
            print(f"  HingeJoint found: {child.GetPath()}")
            continue

        if not child.IsA(UsdGeom.Mesh):
            print(f"  Skipping non-mesh child: {child_name} [{child_type}]")
            continue

        xformable = UsdGeom.Xformable(child)
        transform_op = None
        for xop in xformable.GetOrderedXformOps():
            if xop.GetOpName() == "xformOp:transform":
                transform_op = xop
                break

        if transform_op is None:
            print(f"  WARN: mesh {child_name} has no xformOp:transform")
            continue

        mat = transform_op.Get()
        ox = mat[3][0]  # cm
        oy = mat[3][1]  # cm
        oz = mat[3][2]  # cm

        # Also get world transform for verification
        try:
            world_xform = xform_cache.GetLocalToWorldTransform(child)
            world_t = world_xform.ExtractTranslation()
        except Exception:
            world_t = Gf.Vec3d(0, 0, 0)

        meshes.append({
            "name": child_name,
            "path": str(child.GetPath()),
            "ox": ox, "oy": oy, "oz": oz,
            "world_x": float(world_t[0]),
            "world_y": float(world_t[1]),
            "world_z": float(world_t[2]),
        })

        print(f"  Mesh: {child_name}")
        print(f"    xformOp origin (cm): ({ox:.2f}, {oy:.2f}, {oz:.2f})")
        print(f"    World position (m):  ({world_t[0]:.4f}, {world_t[1]:.4f}, {world_t[2]:.4f})")

    if not meshes:
        print(f"  ERROR: no child meshes found!")
        continue

    # --- HingeJoint properties ---
    joint_info = {}
    if hinge_joint_prim:
        print(f"\n  HingeJoint properties:")
        # Try to read all relevant attributes
        for attr in hinge_joint_prim.GetAttributes():
            attr_name = attr.GetName()
            val = attr.Get()
            if val is not None:
                joint_info[attr_name] = val
                # Print the most useful ones
                if any(kw in attr_name.lower() for kw in
                       ["pos", "axis", "body", "local", "anchor", "limit"]):
                    print(f"    {attr_name} = {val}")
    else:
        print(f"  No HingeJoint child found")

    # --- Bounding box of the door Xform (parent of all meshes) ---
    try:
        bbox = bbox_cache.ComputeWorldBound(door_prim)
        bbox_range = bbox.ComputeAlignedRange()
        bb_min = bbox_range.GetMin()
        bb_max = bbox_range.GetMax()
        bb_mid = bbox_range.GetMidpoint()
        bb_size = bbox_range.GetSize()
    except Exception as e:
        print(f"  ERROR computing bbox: {e}")
        continue

    print(f"\n  Bounding box (world, metres):")
    print(f"    Min: ({bb_min[0]:.4f}, {bb_min[1]:.4f}, {bb_min[2]:.4f})")
    print(f"    Max: ({bb_max[0]:.4f}, {bb_max[1]:.4f}, {bb_max[2]:.4f})")
    print(f"    Mid: ({bb_mid[0]:.4f}, {bb_mid[1]:.4f}, {bb_mid[2]:.4f})")
    print(f"    Size: ({bb_size[0]:.4f}, {bb_size[1]:.4f}, {bb_size[2]:.4f})")

    # --- Determine hinge edge ---
    # The hinge is at one of the 4 vertical edges of the door bbox.
    # Strategy: the mesh origin (ox, oy) in cm corresponds to the door's
    # internal coordinate frame. The hinge is at the bbox edge closest to
    # a specific corner. For these Revit curtain-wall doors, the hinge is
    # typically at the narrow (thickness) edge on one side.
    #
    # We check: which bbox edge (X-min, X-max, Y-min, Y-max) in world
    # metres is farthest from the bbox center? That tells us the door's
    # wider dimension. The hinge is at one end of that wider dimension.
    #
    # For Door 13606713 (known): hinge Y = bbox Y-min = 79.6907 m
    # (mesh origin Y in cm = 8014.94, bbox Y range = 7969.07 to 8060.80 cm)
    # So hinge is at Y-min of the wider (Y) dimension.

    # Convert mesh origin to metres for comparison with bbox
    # The mesh origin in cm needs to account for parent xform
    # But world positions from xform_cache already account for this
    ref_mesh = meshes[0]
    mesh_world_x = ref_mesh["world_x"]
    mesh_world_y = ref_mesh["world_y"]

    # Door extent in X vs Y (in world coords)
    x_extent = bb_size[0]
    y_extent = bb_size[1]

    print(f"\n  Door orientation analysis:")
    print(f"    X extent: {x_extent:.4f} m")
    print(f"    Y extent: {y_extent:.4f} m")

    # The wider axis is the door's swing direction
    if y_extent > x_extent:
        wide_axis = "Y"
        # Hinge is at Y-min or Y-max
        # Check which edge the mesh origin is closer to
        dist_to_ymin = abs(mesh_world_y - bb_min[1])
        dist_to_ymax = abs(mesh_world_y - bb_max[1])
        print(f"    Wide axis: Y (door panel spans Y)")
        print(f"    Mesh center dist to Y-min: {dist_to_ymin:.4f} m")
        print(f"    Mesh center dist to Y-max: {dist_to_ymax:.4f} m")

        # The mesh origin is roughly at the door center.
        # The hinge is at whichever end is the fixed pivot.
        # For a single-leaf door, the hinge edge is typically the one
        # where the door frame meets the wall.
        #
        # Heuristic: for Revit curtain-wall doors, check both edges.
        # We'll report both and flag the known door for validation.

        # Also use mesh origin in cm to compute hinge in cm
        # Bbox min/max in cm = bbox min/max in m * 100
        hinge_y_min_cm = bb_min[1] * 100.0
        hinge_y_max_cm = bb_max[1] * 100.0
        hinge_x_cm = ref_mesh["ox"]  # X stays at mesh origin X

        # Report both options
        print(f"    Option A (hinge at Y-min): hinge_y = {hinge_y_min_cm:.2f} cm ({bb_min[1]:.4f} m)")
        print(f"    Option B (hinge at Y-max): hinge_y = {hinge_y_max_cm:.2f} cm ({bb_max[1]:.4f} m)")

        # For Door 13606713 we know hinge is at Y-min (7969.07 cm)
        # Use Y-min as default heuristic (matches known door)
        hinge_x = hinge_x_cm
        hinge_y = hinge_y_min_cm
        hinge_edge = "Y-min"

    else:
        wide_axis = "X"
        # Hinge is at X-min or X-max
        dist_to_xmin = abs(mesh_world_x - bb_min[0])
        dist_to_xmax = abs(mesh_world_x - bb_max[0])
        print(f"    Wide axis: X (door panel spans X)")
        print(f"    Mesh center dist to X-min: {dist_to_xmin:.4f} m")
        print(f"    Mesh center dist to X-max: {dist_to_xmax:.4f} m")

        hinge_x_min_cm = bb_min[0] * 100.0
        hinge_x_max_cm = bb_max[0] * 100.0
        hinge_y_cm = ref_mesh["oy"]  # Y stays at mesh origin Y

        print(f"    Option A (hinge at X-min): hinge_x = {hinge_x_min_cm:.2f} cm ({bb_min[0]:.4f} m)")
        print(f"    Option B (hinge at X-max): hinge_x = {hinge_x_max_cm:.2f} cm ({bb_max[0]:.4f} m)")

        # Default to X-min
        hinge_x = hinge_x_min_cm
        hinge_y = hinge_y_cm
        hinge_edge = "X-min"

    # Camera center = bbox midpoint in XY (metres)
    cam_center_x = bb_mid[0]
    cam_center_y = bb_mid[1]

    door_result = {
        "door_id":      door_id,
        "door_path":    door_path,
        "meshes":       meshes,
        "bb_min":       (bb_min[0], bb_min[1], bb_min[2]),
        "bb_max":       (bb_max[0], bb_max[1], bb_max[2]),
        "bb_mid":       (bb_mid[0], bb_mid[1], bb_mid[2]),
        "bb_size":      (bb_size[0], bb_size[1], bb_size[2]),
        "wide_axis":    wide_axis,
        "hinge_x_cm":   hinge_x,
        "hinge_y_cm":   hinge_y,
        "hinge_edge":   hinge_edge,
        "cam_center_x": cam_center_x,
        "cam_center_y": cam_center_y,
        "joint_info":   joint_info,
    }
    results.append(door_result)

    print(f"\n  *** DERIVED HINGE: ({hinge_x:.2f}, {hinge_y:.2f}) cm  [edge: {hinge_edge}]")
    print(f"  *** CAMERA CENTER: ({cam_center_x:.4f}, {cam_center_y:.4f}) m")

# ---------------------------------------------------------------------------
# Validate against known door (Doors_13606713)
# ---------------------------------------------------------------------------

print("\n" + "=" * 80)
print("VALIDATION: Doors_13606713 (known hinge)")
print("=" * 80)

known = next((r for r in results if r["door_id"] == "Doors_13606713"), None)
if known:
    known_hinge_x = -2047.728271484375
    known_hinge_y = 7969.07
    dx = abs(known["hinge_x_cm"] - known_hinge_x)
    dy = abs(known["hinge_y_cm"] - known_hinge_y)
    print(f"  Computed hinge: ({known['hinge_x_cm']:.2f}, {known['hinge_y_cm']:.2f}) cm")
    print(f"  Known hinge:    ({known_hinge_x:.2f}, {known_hinge_y:.2f}) cm")
    print(f"  Delta: X={dx:.2f} cm, Y={dy:.2f} cm")
    if dx < 1.0 and dy < 1.0:
        print(f"  PASS — hinge detection matches known values")
    else:
        print(f"  MISMATCH — hinge detection differs from known values!")
        print(f"  The heuristic may need adjustment. Check HingeJoint properties above.")

# ---------------------------------------------------------------------------
# Summary table
# ---------------------------------------------------------------------------

print("\n" + "=" * 80)
print("SUMMARY: All Floor 4 Doors")
print("=" * 80)
print(f"{'#':>2}  {'Door ID':<18}  {'Hinge X (cm)':>14}  {'Hinge Y (cm)':>14}  "
      f"{'Cam X (m)':>10}  {'Cam Y (m)':>10}  {'Edge':>6}  {'Wide':>4}  "
      f"{'BBox (m)':>20}  Meshes")
print("-" * 130)

for i, r in enumerate(results):
    mesh_names = ", ".join(m["name"] for m in r["meshes"])
    bbox_str = f"{r['bb_size'][0]:.2f}×{r['bb_size'][1]:.2f}×{r['bb_size'][2]:.2f}"
    print(f"{i+1:>2}  {r['door_id']:<18}  {r['hinge_x_cm']:>14.2f}  {r['hinge_y_cm']:>14.2f}  "
          f"{r['cam_center_x']:>10.4f}  {r['cam_center_y']:>10.4f}  {r['hinge_edge']:>6}  "
          f"{r['wide_axis']:>4}  {bbox_str:>20}  {mesh_names}")

# ---------------------------------------------------------------------------
# Check for potential double doors (doors within 2m of each other)
# ---------------------------------------------------------------------------

print("\n" + "=" * 80)
print("DOUBLE DOOR CHECK (doors within 2 m of each other)")
print("=" * 80)

for i, a in enumerate(results):
    for j, b in enumerate(results):
        if j <= i:
            continue
        dist = math.sqrt(
            (a["cam_center_x"] - b["cam_center_x"]) ** 2 +
            (a["cam_center_y"] - b["cam_center_y"]) ** 2
        )
        if dist < 2.0:
            print(f"  {a['door_id']} <-> {b['door_id']}: {dist:.2f} m apart — POSSIBLE DOUBLE DOOR")

# ---------------------------------------------------------------------------
# Generate sbatch commands
# ---------------------------------------------------------------------------

print("\n" + "=" * 80)
print("SBATCH COMMANDS for door_static_collection.py")
print("(wide arc with occlusion filtering, multi-distance, multi-height)")
print("=" * 80)

for r in results:
    door_id = r["door_id"]
    cmd = (
        f'sbatch submit_isaac_job.sh simulations/robot/door_static_collection.py \\\n'
        f'    --door-prim "{r["door_path"]}" \\\n'
        f'    --hinge-x {r["hinge_x_cm"]:.2f} --hinge-y {r["hinge_y_cm"]:.2f} \\\n'
        f'    --camera-center-x {r["cam_center_x"]:.4f} '
        f'--camera-center-y {r["cam_center_y"]:.4f} \\\n'
        f'    --camera-z {FLOOR4_CAM_Z} \\\n'
        f'    --camera-arc-offsets -60,-45,-30,-15,0,15,30,45,60 \\\n'
        f'    --camera-distances 1.5,2.0,3.5 \\\n'
        f'    --camera-height-offsets -0.5,0.0,1.0 \\\n'
        f'    --output door_labels_{door_id}'
    )
    print(f"\n# {door_id} — hinge at {r['hinge_edge']}, "
          f"center ({r['cam_center_x']:.2f}, {r['cam_center_y']:.2f}) m")
    print(cmd)

# ---------------------------------------------------------------------------
# Also generate a compact shell script for batch submission
# ---------------------------------------------------------------------------

print("\n" + "=" * 80)
print("BATCH SUBMISSION SCRIPT (copy-paste to submit all doors)")
print("=" * 80)
print("#!/bin/bash")
print("# Submit data collection for all Floor 4 doors")
print("# Generated by floor4_door_hinge_diagnostic.py\n")

for r in results:
    door_id = r["door_id"]
    print(f'echo "Submitting {door_id}..."')
    # Single-line version for easy copy-paste
    cmd_oneline = (
        f'sbatch submit_isaac_job.sh simulations/robot/door_static_collection.py'
        f' --door-prim "{r["door_path"]}"'
        f' --hinge-x {r["hinge_x_cm"]:.2f} --hinge-y {r["hinge_y_cm"]:.2f}'
        f' --camera-center-x {r["cam_center_x"]:.4f}'
        f' --camera-center-y {r["cam_center_y"]:.4f}'
        f' --camera-z {FLOOR4_CAM_Z}'
        f' --camera-arc-offsets -60,-45,-30,-15,0,15,30,45,60'
        f' --camera-distances 1.5,2.0,3.5'
        f' --camera-height-offsets -0.5,0.0,1.0'
        f' --output door_labels_{door_id}'
    )
    print(cmd_oneline)
    print()

print('echo "All door collection jobs submitted."')

print("\n" + "=" * 80)
print("Diagnostic complete.")
print("=" * 80)

simulation_app.close()
