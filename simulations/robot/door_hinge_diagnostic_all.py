#!/usr/bin/env python3
"""
Multi-Floor Door Hinge Diagnostic

Extends floor4_door_hinge_diagnostic.py to cover the 9 doors in the
generalization experiment (7 training + 2 held-out test) spanning floors
2-5 and two Revit curtain-wall door families (Type_FG___New and
Type_F___NEW). For each door this script:

  1. Reads each child mesh's xformOp:transform matrix (origin in cm)
  2. Computes the world bounding box at the default (0 deg) pose
  3. Reports both hinge-edge candidates (Y-min/Y-max or X-min/X-max) so
     the collection script can be run against whichever matches
     DOOR_CONFIG["flip"] in door_snapshot.py
  4. Emits ready-to-use sbatch lines for door_static_collection.py, one
     per door, with the bbox-midpoint-derived camera center + camera_z

Run via:
  sbatch --partition=teaching submit_isaac_job.sh simulations/robot/door_hinge_diagnostic_all.py

Results land in logs/isaac_<JOBID>.out.
"""

import os
import sys
import math

os.environ["OMNI_KIT_ACCEPT_EULA"] = "YES"
from isaacsim import SimulationApp
simulation_app = SimulationApp({"headless": True, "hide_ui": True})

import omni.usd
from pxr import Usd, UsdGeom, Gf

_script_dir  = os.path.dirname(os.path.abspath(__file__))
_project_dir = os.path.abspath(os.path.join(_script_dir, "..", ".."))

USD_PATH = os.path.join(_project_dir, "MSOEBuildingV4_physics.usd")

DOOR_BASE = (
    "/World/Building/_22_129___MSOE_Engineering_Building___A___2_25_2026"
    "/Geometry/Doors"
)

# Target doors from virtual-beaming-lightning.md. Full prim paths are from
# door_catalog.json. Split: (type_family, door_id). The 7 training doors are
# marked train, the 2 held-out generalization test doors are marked test.
TARGET_DOORS = [
    # (role, door_id, type_family_path)
    ("train", "Doors_9817744",  "RS___Curtain_Wall___Type_FG___New/New"),
    ("train", "Doors_12066864", "RS___Curtain_Wall___Type_FG___New/New"),
    ("train", "Doors_3135827",  "RS___Curtain_Wall___Type_FG___New/New"),
    ("train", "Doors_1868517",  "RS___Curtain_Wall___Type_FG___New/New"),
    ("train", "Doors_10383956", "RS___Curtain_Wall___Type_FG___New/New"),
    ("train", "Doors_13182085", "RS___Curtain_Wall___Type_FG___New/New"),
    ("train", "Doors_13606713", "RS___Curtain_Wall___Type_FG___New/New"),  # known hinge — validation
    ("test",  "Doors_13607620", "RS___Curtain_Wall___Type_F___NEW/New"),
    ("test",  "Doors_4190126",  "RS___Curtain_Wall___Type_FG___New/New"),
]

# flip-swing setting per door, pulled from door_snapshot.py:216-230 DOOR_CONFIG
DOOR_FLIP = {
    "Doors_9817744":  True,
    "Doors_12066864": False,
    "Doors_13607620": True,
    "Doors_3135827":  False,
    "Doors_1868517":  True,
    "Doors_10383956": True,
    "Doors_13182085": False,
    "Doors_4190126":  False,
    "Doors_13606713": False,
}

# ---------------------------------------------------------------------------
print("=" * 80)
print("Multi-Floor Door Hinge Diagnostic — 9 doors (7 train + 2 test)")
print(f"USD: {USD_PATH}")
print("=" * 80)

if not os.path.isfile(USD_PATH):
    print(f"ERROR: USD file not found at {USD_PATH}")
    simulation_app.close()
    sys.exit(1)

print(f"\nOpening stage (this may take 30-60 s for the 160 MB USD)...")
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

for role, door_id, type_path in TARGET_DOORS:
    door_path = f"{DOOR_BASE}/{type_path}/{door_id}"
    door_prim = stage.GetPrimAtPath(door_path)

    print("-" * 80)
    print(f"DOOR: {door_id}  [{role}]")
    print(f"  Path: {door_path}")

    if not door_prim.IsValid():
        print(f"  ERROR: prim not found!")
        continue

    # --- Child meshes: read xformOp:transform ---
    meshes = []
    for child in door_prim.GetChildren():
        child_name = child.GetName()
        if not child.IsA(UsdGeom.Mesh):
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
        ox = mat[3][0]
        oy = mat[3][1]
        oz = mat[3][2]

        try:
            world_xform = xform_cache.GetLocalToWorldTransform(child)
            world_t = world_xform.ExtractTranslation()
        except Exception:
            world_t = Gf.Vec3d(0, 0, 0)

        meshes.append({
            "name":    child_name,
            "ox": ox, "oy": oy, "oz": oz,
            "world_x": float(world_t[0]),
            "world_y": float(world_t[1]),
            "world_z": float(world_t[2]),
        })
        print(f"  Mesh: {child_name}  xform_origin_cm=({ox:.2f}, {oy:.2f}, {oz:.2f})  "
              f"world_m=({world_t[0]:.3f}, {world_t[1]:.3f}, {world_t[2]:.3f})")

    if not meshes:
        print(f"  ERROR: no child meshes found!")
        continue

    # --- Bbox of door xform (parent of all meshes) ---
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

    print(f"\n  Bbox (world m): "
          f"min=({bb_min[0]:.3f},{bb_min[1]:.3f},{bb_min[2]:.3f}) "
          f"max=({bb_max[0]:.3f},{bb_max[1]:.3f},{bb_max[2]:.3f})  "
          f"size=({bb_size[0]:.3f},{bb_size[1]:.3f},{bb_size[2]:.3f})")

    x_extent = bb_size[0]
    y_extent = bb_size[1]

    # Wide axis = door swing span. Hinge is at one end of the wide axis.
    if y_extent > x_extent:
        wide_axis = "Y"
        hinge_y_min_cm = bb_min[1] * 100.0
        hinge_y_max_cm = bb_max[1] * 100.0
        hinge_x_cm = meshes[0]["ox"]

        # Default to Y-min (matches Doors_13606713's known hinge)
        hinge_x = hinge_x_cm
        hinge_y = hinge_y_min_cm
        hinge_edge = "Y-min"
        alt_hinge_y = hinge_y_max_cm
        alt_edge = "Y-max"
    else:
        wide_axis = "X"
        hinge_x_min_cm = bb_min[0] * 100.0
        hinge_x_max_cm = bb_max[0] * 100.0
        hinge_y_cm = meshes[0]["oy"]

        hinge_x = hinge_x_min_cm
        hinge_y = hinge_y_cm
        hinge_edge = "X-min"
        alt_hinge_y = hinge_y_cm
        alt_edge = "X-max"

    cam_center_x = float(bb_mid[0])
    cam_center_y = float(bb_mid[1])
    cam_z        = float(bb_mid[2])   # eye-level = door vertical midpoint

    print(f"  Wide axis: {wide_axis}")
    if wide_axis == "Y":
        print(f"  Primary hinge (Y-min): ({hinge_x:.2f}, {hinge_y:.2f}) cm")
        print(f"  Alt hinge (Y-max):     ({hinge_x:.2f}, {alt_hinge_y:.2f}) cm")
    else:
        alt_hinge_x = bb_max[0] * 100.0
        print(f"  Primary hinge (X-min): ({hinge_x:.2f}, {hinge_y:.2f}) cm")
        print(f"  Alt hinge (X-max):     ({alt_hinge_x:.2f}, {hinge_y:.2f}) cm")

    results.append({
        "role":         role,
        "door_id":      door_id,
        "door_path":    door_path,
        "wide_axis":    wide_axis,
        "hinge_x_cm":   hinge_x,
        "hinge_y_cm":   hinge_y,
        "hinge_edge":   hinge_edge,
        "cam_center_x": cam_center_x,
        "cam_center_y": cam_center_y,
        "cam_z":        cam_z,
        "bb_size":      (float(bb_size[0]), float(bb_size[1]), float(bb_size[2])),
    })

# ---------------------------------------------------------------------------
# Validation: Doors_13606713 (known hinge)
# ---------------------------------------------------------------------------

print("\n" + "=" * 80)
print("VALIDATION: Doors_13606713 (known hinge from working collection)")
print("=" * 80)
known = next((r for r in results if r["door_id"] == "Doors_13606713"), None)
if known:
    known_x, known_y = -2047.728271484375, 7969.07
    dx = abs(known["hinge_x_cm"] - known_x)
    dy = abs(known["hinge_y_cm"] - known_y)
    status = "PASS" if dx < 1.0 and dy < 1.0 else "MISMATCH"
    print(f"  Computed: ({known['hinge_x_cm']:.2f}, {known['hinge_y_cm']:.2f}) cm")
    print(f"  Known:    ({known_x:.2f}, {known_y:.2f}) cm")
    print(f"  Delta: X={dx:.2f} cm, Y={dy:.2f} cm  [{status}]")
    if status == "MISMATCH":
        print("  => Primary edge heuristic is wrong for this building family.")
        print("     Use the 'Alt hinge' values from the per-door block above.")

# ---------------------------------------------------------------------------
# Summary table
# ---------------------------------------------------------------------------

print("\n" + "=" * 80)
print("SUMMARY — all 9 doors")
print("=" * 80)
print(f"{'Role':<6}  {'Door ID':<18}  {'Hinge X (cm)':>13}  {'Hinge Y (cm)':>13}  "
      f"{'Cam X (m)':>10}  {'Cam Y (m)':>10}  {'Cam Z (m)':>10}  {'Edge':>6}  "
      f"{'Flip':>5}")
print("-" * 110)
for r in results:
    flip = DOOR_FLIP.get(r["door_id"], False)
    print(f"{r['role']:<6}  {r['door_id']:<18}  "
          f"{r['hinge_x_cm']:>13.2f}  {r['hinge_y_cm']:>13.2f}  "
          f"{r['cam_center_x']:>10.4f}  {r['cam_center_y']:>10.4f}  "
          f"{r['cam_z']:>10.4f}  {r['hinge_edge']:>6}  {str(flip):>5}")

# ---------------------------------------------------------------------------
# sbatch commands
# ---------------------------------------------------------------------------

print("\n" + "=" * 80)
print("SBATCH COMMANDS — copy/paste to run")
print("  train doors: full grid (lighting x material x angle)")
print("  test doors:  --baseline-only (bright_fluorescent + original only)")
print("=" * 80)

for r in results:
    door_id = r["door_id"]
    flip = DOOR_FLIP.get(door_id, False)
    flip_flag = " --flip-swing" if flip else ""
    baseline_flag = " --baseline-only" if r["role"] == "test" else ""
    output_name = (
        f"door_labels_{door_id}" if r["role"] == "train"
        else f"door_gen_test_{door_id}"
    )

    cmd = (
        f'sbatch --partition=teaching submit_isaac_job.sh '
        f'simulations/robot/door_static_collection.py \\\n'
        f'    --door-prim "{r["door_path"]}" \\\n'
        f'    --hinge-x {r["hinge_x_cm"]:.2f} --hinge-y {r["hinge_y_cm"]:.2f} \\\n'
        f'    --camera-center-x {r["cam_center_x"]:.4f} '
        f'--camera-center-y {r["cam_center_y"]:.4f} \\\n'
        f'    --camera-z {r["cam_z"]:.2f} \\\n'
        f'    --grid-mode{baseline_flag}{flip_flag} \\\n'
        f'    --output {output_name}'
    )
    print(f"\n# {door_id}  [{r['role']}]  hinge_edge={r['hinge_edge']}  flip={flip}")
    print(cmd)

print("\n" + "=" * 80)
print("Diagnostic complete.")
print("=" * 80)

simulation_app.close()
