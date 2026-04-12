#!/usr/bin/env python3
"""
Door Snapshot — All Floors

Auto-discovers every door in the building USD by traversing the prim
hierarchy for Xform prims under /Geometry/Doors/ that have child Mesh
prims with xformOp:transform. Places a camera 2 m in front of each
door (both east and west sides), renders one frame, and saves a JPEG.

Lighting: low-intensity DomeLight (ambient only) + per-door ceiling
RectLight grid on both sides of the door.

Output: data/door_snapshots_all/<DoorID>_<side>.jpg

Run via:
  sbatch submit_isaac_job.sh simulations/robot/door_snapshot.py
"""

import os
import sys
import numpy as np

os.environ["OMNI_KIT_ACCEPT_EULA"] = "YES"
from isaacsim import SimulationApp
simulation_app = SimulationApp({"headless": True})

import omni.usd
from pxr import Usd, UsdLux, UsdGeom, Gf, Sdf

import isaacsim.core.utils.numpy.rotations as rot_utils
from isaacsim.core.api import World
from isaacsim.sensors.camera import Camera

_script_dir  = os.path.dirname(os.path.abspath(__file__))
_project_dir = os.path.abspath(os.path.join(_script_dir, "..", ".."))

CAMERA_DIST    = 2.5     # metres from door centre
CAM_RESOLUTION = (640, 480)

# ---------------------------------------------------------------------------
# Scene setup
# ---------------------------------------------------------------------------

building_usd = os.path.join(_project_dir, "MSOEBuildingV4_physics.usd")
if not os.path.isfile(building_usd):
    print(f"ERROR: building USD not found: {building_usd}")
    simulation_app.close()
    sys.exit(1)

world = World(stage_units_in_meters=1.0)
stage = omni.usd.get_context().get_stage()
stage.GetRootLayer().subLayerPaths.insert(0, building_usd)

# Ambient-only DomeLight
dome = UsdLux.DomeLight.Define(stage, Sdf.Path("/World/DomeLight"))
dome.CreateIntensityAttr(5000)

world.reset()
simulation_app.update()

print("[DoorSnapshot] Warming up renderer (20 steps) ...")
for _ in range(20):
    world.step(render=True)

# ---------------------------------------------------------------------------
# Auto-discover all doors in the building
# ---------------------------------------------------------------------------

print("\n[DoorSnapshot] Scanning for door prims ...")

bbox_cache = UsdGeom.BBoxCache(Usd.TimeCode.Default(), [UsdGeom.Tokens.default_],
                                useExtentsHint=False)

doors = []  # list of dicts with door info

for prim in stage.TraverseAll():
    # We want Xform prims that are direct children under a /Doors/ category
    # and that have child Mesh prims (the actual door panels).
    # Pattern: .../Geometry/Doors/<door_type>/<category>/<Doors_XXXXXXXX>
    path_str = prim.GetPath().pathString
    prim_name = prim.GetName()

    # Must be under /Doors/ in the hierarchy
    if "/Doors/" not in path_str:
        continue

    # Must start with "Doors_" (the instance prim, not the mesh children)
    if not prim_name.startswith("Doors_"):
        continue

    # Must be an Xform (parent of mesh children), not a Mesh itself
    if prim.GetTypeName() not in ("Xform", ""):
        continue

    # Must have at least one child Mesh with xformOp:transform
    has_mesh = False
    for child in prim.GetChildren():
        if child.IsA(UsdGeom.Mesh):
            xformable = UsdGeom.Xformable(child)
            for xop in xformable.GetOrderedXformOps():
                if xop.GetOpName() == "xformOp:transform":
                    has_mesh = True
                    break
        if has_mesh:
            break

    if not has_mesh:
        continue

    # Compute bounding box
    try:
        bbox = bbox_cache.ComputeWorldBound(prim)
        bbox_range = bbox.ComputeAlignedRange()
        if bbox_range.IsEmpty():
            continue
        bb_mid = bbox_range.GetMidpoint()
        bb_size = bbox_range.GetSize()
    except Exception:
        continue

    # Skip degenerate doors (bbox too small to be a real door)
    max_dim = max(bb_size[0], bb_size[1], bb_size[2])
    if max_dim < 0.5:  # less than 0.5m in any dimension = not a door
        continue

    cx = float(bb_mid[0])
    cy = float(bb_mid[1])
    cz = float(bb_mid[2])  # vertical midpoint of door
    height = float(bb_size[2])

    # Camera Z = door vertical midpoint (eye level relative to door)
    cam_z = cz

    # Determine which axis the door face is perpendicular to
    # (thin axis = face normal direction)
    x_extent = float(bb_size[0])
    y_extent = float(bb_size[1])

    if x_extent < y_extent:
        # Door is thin in X → face perpendicular to X → camera approaches along X
        approach_axis = "X"
    else:
        # Door is thin in Y → face perpendicular to Y → camera approaches along Y
        approach_axis = "Y"

    # Count mesh children
    mesh_count = sum(1 for c in prim.GetChildren() if c.IsA(UsdGeom.Mesh))

    # Get the door type from the path (the grandparent category)
    path_parts = path_str.split("/")
    door_type = "unknown"
    for j, part in enumerate(path_parts):
        if part == "Doors" and j + 2 < len(path_parts):
            door_type = path_parts[j + 1]  # e.g. "RS___Curtain_Wall___Type_FG___New"
            break

    # Approximate floor from Z
    floor_z = round(cz / 4) * 4  # bucket by ~4m intervals

    doors.append({
        "id":            prim_name,
        "path":          path_str,
        "cx":            cx,
        "cy":            cy,
        "cz":            cz,
        "cam_z":         cam_z,
        "height":        height,
        "x_extent":      x_extent,
        "y_extent":      y_extent,
        "approach_axis": approach_axis,
        "mesh_count":    mesh_count,
        "door_type":     door_type,
        "floor_z":       floor_z,
    })

# Sort by Z level then X position
doors.sort(key=lambda d: (d["floor_z"], d["cx"]))

print(f"[DoorSnapshot] Found {len(doors)} door prims across all floors")

# ---------------------------------------------------------------------------
# Filter out double doors — any two doors within 1.5m of each other
# (on the same approximate floor) are a double-door pair; remove both.
# ---------------------------------------------------------------------------

import math

double_door_ids = set()
for i, a in enumerate(doors):
    for j, b in enumerate(doors):
        if j <= i:
            continue
        # Only compare doors on the same floor
        if a["floor_z"] != b["floor_z"]:
            continue
        dist = math.sqrt((a["cx"] - b["cx"]) ** 2 + (a["cy"] - b["cy"]) ** 2)
        if dist < 1.5:
            double_door_ids.add(a["id"])
            double_door_ids.add(b["id"])
            print(f"  [Double door] {a['id']} <-> {b['id']}: {dist:.2f}m apart — removing both")

doors_before = len(doors)
doors = [d for d in doors if d["id"] not in double_door_ids]
print(f"[DoorSnapshot] Removed {doors_before - len(doors)} double-door leaves, "
      f"{len(doors)} single doors remain\n")

# ---------------------------------------------------------------------------
# Keep only the top-10 manually selected doors (best lighting + diversity)
# ---------------------------------------------------------------------------

# ---------------------------------------------------------------------------
# Final 9 doors — per-door config for allowed sides and swing direction.
# Maps door ID → {"sides": set of allowed side labels, "flip": bool}
# ---------------------------------------------------------------------------

DOOR_CONFIG = {
    # Floor 2 (Z≈15)
    "Doors_9817744":  {"sides": {"north"},  "flip": True},
    "Doors_12066864": {"sides": {"north"},  "flip": False},
    "Doors_13607620": {"sides": {"north"},  "flip": True},
    # Floor 3 (Z≈19)
    "Doors_3135827":  {"sides": {"south"},  "flip": False},
    "Doors_1868517":  {"sides": {"north"},  "flip": True},
    "Doors_10383956": {"sides": {"west"},   "flip": True},
    # Floor 4 (Z≈24)
    "Doors_13182085": {"sides": {"east"},   "flip": False},
    # Floor 5 (Z≈28)
    "Doors_4190126":  {"sides": {"west"},   "flip": False},
    "Doors_13606713": {"sides": {"east"},   "flip": False},
}

SELECTED_DOOR_IDS = set(DOOR_CONFIG.keys())

doors_before = len(doors)
doors = [d for d in doors if d["id"] in SELECTED_DOOR_IDS]
print(f"[DoorSnapshot] Filtered to {len(SELECTED_DOOR_IDS)} selected doors: "
      f"{doors_before} → {len(doors)}\n")

# Print summary table
print(f"{'#':>3}  {'Door ID':<20}  {'Z':>6}  {'X':>8}  {'Y':>8}  "
      f"{'H(m)':>5}  {'Axis':>4}  {'Meshes':>6}  Type")
print("-" * 110)
for i, d in enumerate(doors):
    type_short = d["door_type"][:40]
    print(f"{i+1:>3}  {d['id']:<20}  {d['cz']:>6.1f}  {d['cx']:>8.2f}  {d['cy']:>8.2f}  "
          f"{d['height']:>5.2f}  {d['approach_axis']:>4}  {d['mesh_count']:>6}  {type_short}")

# Print floor distribution
from collections import Counter
floor_counts = Counter(d["floor_z"] for d in doors)
print(f"\nFloor distribution (single doors only):")
for z in sorted(floor_counts):
    print(f"  Z≈{z:5.0f} m : {floor_counts[z]} doors")

# ---------------------------------------------------------------------------
# Output directory
# ---------------------------------------------------------------------------

out_dir = os.path.join(_project_dir, "data", "door_snapshots_top10_finalV3")
os.makedirs(out_dir, exist_ok=True)

try:
    from PIL import Image as _PIL
    _pil_ok = True
except ImportError:
    _pil_ok = False
    print("[WARN] PIL not available — will save raw .npy instead of JPEG")

# ---------------------------------------------------------------------------
# Create per-door lighting + camera, render, save
# ---------------------------------------------------------------------------

print(f"\n[DoorSnapshot] Capturing {len(doors)} doors × 2 sides ...")
print("=" * 70)

light_idx = 0

for i, d in enumerate(doors):
    door_id = d["id"]
    cx, cy, cam_z = d["cx"], d["cy"], d["cam_z"]

    print(f"\n[{i+1}/{len(doors)}] {door_id}  Z={d['cz']:.1f}m  "
          f"({d['approach_axis']}-approach, {d['mesh_count']} meshes)")

    # --- Per-door ceiling lighting grid ---
    # 5×3 grid: dense along approach axis (±1m, ±2.5m) and ±2m along door span.
    # Placing lights closer together prevents structural columns / dark wall
    # panels from blocking light in narrow corridors (fixes Doors_10383956,
    # Doors_13606713 which had split-lighting with the old ±3m grid).
    if d["approach_axis"] == "X":
        # Camera approaches along X, door spans Y
        approach_offsets = [-2.5, -1.0, 0.0, 1.0, 2.5]
        span_offsets     = [-2.0, 0.0, 2.0]
        for x_off in approach_offsets:
            for y_off in span_offsets:
                lp = f"/World/HLight_{light_idx}"
                lt = UsdLux.RectLight.Define(stage, Sdf.Path(lp))
                lt.CreateIntensityAttr(50000)
                lt.CreateWidthAttr(1.5)
                lt.CreateHeightAttr(0.8)
                lx = UsdGeom.Xformable(lt.GetPrim())
                lx.AddTranslateOp().Set(Gf.Vec3d(cx + x_off, cy + y_off, cam_z + 1.2))
                lx.AddRotateXOp().Set(-90.0)
                light_idx += 1
    else:
        # Camera approaches along Y, door spans X
        span_offsets     = [-2.0, 0.0, 2.0]
        approach_offsets = [-2.5, -1.0, 0.0, 1.0, 2.5]
        for x_off in span_offsets:
            for y_off in approach_offsets:
                lp = f"/World/HLight_{light_idx}"
                lt = UsdLux.RectLight.Define(stage, Sdf.Path(lp))
                lt.CreateIntensityAttr(50000)
                lt.CreateWidthAttr(1.5)
                lt.CreateHeightAttr(0.8)
                lx = UsdGeom.Xformable(lt.GetPrim())
                lx.AddTranslateOp().Set(Gf.Vec3d(cx + x_off, cy + y_off, cam_z + 1.2))
                lx.AddRotateXOp().Set(-90.0)
                light_idx += 1

    simulation_app.update()

    # --- Camera on each side ---
    if d["approach_axis"] == "X":
        sides = [
            ("east", cx + CAMERA_DIST, cy, 180.0),
            ("west", cx - CAMERA_DIST, cy, 0.0),
        ]
    else:
        sides = [
            ("north", cx, cy + CAMERA_DIST, 270.0),
            ("south", cx, cy - CAMERA_DIST, 90.0),
        ]

    for side_label, cam_x, cam_y, yaw_deg in sides:
        # Skip sides not in this door's allowed set
        cfg = DOOR_CONFIG.get(door_id, {})
        allowed_sides = cfg.get("sides")
        if allowed_sides is not None and side_label not in allowed_sides:
            print(f"  {side_label}: skipped (not in allowed sides)")
            continue

        cam_pos = np.array([cam_x, cam_y, cam_z])
        cam_ori = rot_utils.euler_angles_to_quats(
            np.array([0.0, 0.0, yaw_deg]), degrees=True
        )

        prim_path = f"/World/SnapCam_{i}_{side_label}"

        cam = Camera(
            prim_path=prim_path,
            position=cam_pos,
            frequency=30,
            resolution=CAM_RESOLUTION,
            orientation=cam_ori,
        )

        simulation_app.update()
        cam.initialize()

        usd_cam = UsdGeom.Camera(stage.GetPrimAtPath(prim_path))
        usd_cam.GetFocalLengthAttr().Set(5.0)
        usd_cam.GetClippingRangeAttr().Set((0.01, 10000.0))

        for _ in range(15):
            world.step(render=True)

        rgba = cam.get_rgba()
        if rgba is None:
            print(f"  {side_label}: camera returned None — skipping")
            continue

        rgb = rgba[:, :, :3].astype(np.uint8)
        # Include floor Z in filename for easy sorting
        floor_label = f"z{d['cz']:.0f}"
        filename = f"{floor_label}_{door_id}_{side_label}.jpg"
        filepath = os.path.join(out_dir, filename)

        if _pil_ok:
            _PIL.fromarray(rgb).save(filepath, "JPEG", quality=92)
        else:
            np.save(filepath.replace(".jpg", ".npy"), rgb)

        print(f"  {side_label}: saved {filename}")

print(f"\n{'=' * 70}")
print(f"[DoorSnapshot] Done — {len(doors)} doors × 2 sides")
print(f"Total lights: {light_idx}")
print(f"Output: {out_dir}/")
print(f"{'=' * 70}")

world.stop()
simulation_app.close()
