#!/usr/bin/env python3
"""
Floor 4 Door Diagnostic

Opens the building USD and scans all mesh prims in the Floor 4 Z range
(26.0 – 30.0 m). Filters for Revit door component naming patterns
("door", "panel", "leaf", "swing") and prints each prim's:
  - USD prim path
  - World XYZ translation
  - Bounding box size (W × D × H in meters)

Also prints a summary of ALL Floor 4 mesh prims for orientation.

Run via:
  sbatch submit_isaac_job.sh simulations/robot/floor4_door_diagnostic.py

Results appear in logs/isaac_<JOBID>.out
"""

import os
import sys

# ---------------------------------------------------------------------------
# pxr is a kit-framework extension — SimulationApp must be initialized first
# to load the shared libraries that make `pxr` importable.
# ---------------------------------------------------------------------------
os.environ["OMNI_KIT_ACCEPT_EULA"] = "YES"
from isaacsim import SimulationApp
simulation_app = SimulationApp({"headless": True, "hide_ui": True})

import omni.usd
from pxr import Usd, UsdGeom, Gf

_script_dir  = os.path.dirname(os.path.abspath(__file__))
_project_dir = os.path.abspath(os.path.join(_script_dir, "..", ".."))

USD_PATH = os.path.join(_project_dir, "innovation_center_finalV1.usd")

FLOOR4_Z_MIN = 26.0
FLOOR4_Z_MAX = 30.0

# Revit exports hinged doors under a "Doors" family category in the USD path.
# Previous run showed "panel" caused 314 false positives (curtain wall panels,
# metal facade cladding). Now we search by:
#   1. PATH_DOOR_COMPONENTS — exact USD path segment matches (Revit family names)
#   2. NAME_DOOR_KEYWORDS   — prim name contains these (leaf/panel inside a Doors family)
# We also do an ALL-FLOORS search for any prim path containing "/Doors/" to find
# which floors have actual door geometry.
PATH_DOOR_COMPONENTS = ["/doors/", "/door_", "_door/", "door-"]  # path-level matches
NAME_DOOR_KEYWORDS   = ["leaf", "swing", "flush"]                # prim-name only

# ---------------------------------------------------------------------------
print("=" * 70)
print(f"Floor 4 Door Diagnostic")
print(f"USD: {USD_PATH}")
print(f"Floor 4 Z range: {FLOOR4_Z_MIN} – {FLOOR4_Z_MAX} m")
print("=" * 70)
print("Strategy: search for /Geometry/Doors/ path segment (Revit family category)")
print("          + prim name keywords: leaf, swing, flush")
print("          + all-floors scan to find which floors have door geometry")
print("=" * 70)

if not os.path.isfile(USD_PATH):
    print(f"ERROR: USD file not found at {USD_PATH}")
    simulation_app.close()
    sys.exit(1)

# Open via omni.usd context so the kit stage is fully initialized
print(f"\nOpening stage via omni.usd (this may take 30–60 s for the 154 MB USD)...")
omni.usd.get_context().open_stage(USD_PATH)
simulation_app.update()   # let kit process the open request
stage = omni.usd.get_context().get_stage()
if not stage:
    print("ERROR: Failed to open USD stage")
    simulation_app.close()
    sys.exit(1)
print("Stage opened.\n")

time_code  = Usd.TimeCode.Default()
xform_cache = UsdGeom.XformCache(time_code)
bbox_cache  = UsdGeom.BBoxCache(time_code, [UsdGeom.Tokens.default_], True)

# ---------------------------------------------------------------------------
# Traverse all mesh prims
# ---------------------------------------------------------------------------

floor4_meshes  = []   # all mesh prims in Z range
door_prims     = []   # strict door matches (path component OR name keyword)
all_door_prims = []   # any prim on ANY floor with /Doors/ in path

total_prims = 0

print("Traversing entire stage for Mesh prims (all floors + Floor 4 detail)...")

for prim in stage.TraverseAll():
    if prim.GetTypeName() != "Mesh":
        continue

    total_prims += 1
    path_str   = prim.GetPath().pathString
    path_lower = path_str.lower()
    name_lower = prim.GetName().lower()

    # All-floors door scan: any prim whose path contains /doors/ (Revit family)
    if "/doors/" in path_lower:
        try:
            world_xform = xform_cache.GetLocalToWorldTransform(prim)
            translation = world_xform.ExtractTranslation()
            try:
                bbox = bbox_cache.ComputeWorldBound(prim)
                size = bbox.GetRange().GetSize()
                bbox_size = (size[0], size[1], size[2])
            except Exception:
                bbox_size = (0.0, 0.0, 0.0)
            all_door_prims.append({
                "path": path_str, "name": prim.GetName(),
                "x": float(translation[0]), "y": float(translation[1]),
                "z": float(translation[2]), "bbox": bbox_size,
            })
        except Exception:
            pass

    # Floor 4 only from here
    try:
        world_xform = xform_cache.GetLocalToWorldTransform(prim)
    except Exception:
        continue

    translation = world_xform.ExtractTranslation()
    z = translation[2]

    if not (FLOOR4_Z_MIN <= z <= FLOOR4_Z_MAX):
        continue

    try:
        bbox = bbox_cache.ComputeWorldBound(prim)
        size = bbox.GetRange().GetSize()
        bbox_size = (size[0], size[1], size[2])
    except Exception:
        bbox_size = (0.0, 0.0, 0.0)

    # Strict door match: path has /doors/ segment OR name has door-specific keyword
    is_door = (
        "/doors/" in path_lower
        or any(kw in name_lower for kw in NAME_DOOR_KEYWORDS)
        or any(kw in path_lower for kw in PATH_DOOR_COMPONENTS)
    )

    entry = {
        "prim":    prim,
        "path":    path_str,
        "name":    prim.GetName(),
        "x":       float(translation[0]),
        "y":       float(translation[1]),
        "z":       float(translation[2]),
        "bbox":    bbox_size,
        "is_door": is_door,
    }

    floor4_meshes.append(entry)
    if is_door:
        door_prims.append(entry)

# ---------------------------------------------------------------------------
# Report 0: all-floors door scan (/Geometry/Doors/ path component)
# ---------------------------------------------------------------------------

all_door_prims.sort(key=lambda e: (round(e["z"], 1), e["x"]))

print(f"\nTotal mesh prims in USD: {total_prims}")
print(f"\n--- ALL-FLOORS DOOR SCAN (/Doors/ path segment, all {len(all_door_prims)} prims) ---")
if not all_door_prims:
    print("  NO prims found with '/Doors/' in their USD path on any floor.")
    print("  The Revit export may not include door geometry, or doors are")
    print("  named differently (e.g. 'Specialty Equipment', 'Generic Model').")
    # Print a sample of unique geometry category names to help identify the pattern
    categories = set()
    for p in stage.TraverseAll():
        path = p.GetPath().pathString
        parts = path.split("/")
        if len(parts) > 5 and parts[3] == "Building":
            # geometry category is typically parts[5]: Walls, Floors, Doors, etc.
            if len(parts) > 6:
                categories.add(parts[6])
    print(f"\n  USD Geometry categories found in /World/Building/.../Geometry/:")
    for cat in sorted(categories)[:40]:
        print(f"    {cat}")
else:
    # Group by approximate Z to show floor distribution
    from collections import defaultdict
    by_floor = defaultdict(list)
    for e in all_door_prims:
        floor_z = round(e["z"] / 4) * 4  # bucket by 4m intervals
        by_floor[floor_z].append(e)

    print(f"\n  Distribution by approximate Z level:")
    for z_level in sorted(by_floor.keys()):
        print(f"    Z≈{z_level:5.1f} m : {len(by_floor[z_level]):4d} door prims")

    print(f"\n  First 30 door prims (all floors):")
    print(f"  {'#':>4}  {'Z':>6}  {'X':>8}  {'Y':>8}  Path")
    print("  " + "-" * 100)
    for i, e in enumerate(all_door_prims[:30]):
        print(f"  {i+1:>4}  {e['z']:>6.2f}  {e['x']:>8.2f}  {e['y']:>8.2f}  {e['path']}")

# ---------------------------------------------------------------------------
# Report: all Floor 4 meshes (first 50, sorted by Z then X)
# ---------------------------------------------------------------------------

floor4_meshes.sort(key=lambda e: (round(e["z"], 1), e["x"]))

print(f"\nMesh prims on Floor 4 (Z {FLOOR4_Z_MIN}–{FLOOR4_Z_MAX} m): {len(floor4_meshes)}")
print(f"  Of which are strict door matches: {len(door_prims)}")

print("\n--- FLOOR 4: ALL MESH PRIMS (first 50, sorted by Z then X) ---")
print(f"{'#':>4}  {'Z':>6}  {'X':>8}  {'Y':>8}  {'BBox W×D×H (m)':>22}  Path")
print("-" * 100)

for i, e in enumerate(floor4_meshes[:50]):
    bbox_str = f"{e['bbox'][0]:.2f}×{e['bbox'][1]:.2f}×{e['bbox'][2]:.2f}"
    door_tag = " [DOOR]" if e["is_door"] else ""
    print(f"{i+1:>4}  {e['z']:>6.2f}  {e['x']:>8.2f}  {e['y']:>8.2f}  "
          f"{bbox_str:>22}  {e['path']}{door_tag}")

if len(floor4_meshes) > 50:
    print(f"  ... ({len(floor4_meshes) - 50} more prims not shown)")

# ---------------------------------------------------------------------------
# Report: door prims (all of them)
# ---------------------------------------------------------------------------

print(f"\n--- DOOR PRIMS ON FLOOR 4 (keyword match, all {len(door_prims)}) ---")
if not door_prims:
    print("  No door-keyword prims found in Z range 26.0–30.0.")
    print("  Try expanding the Z range or checking the USD naming conventions.")
    print("  Hint: print some non-door prim names above to see Revit naming patterns.")
    # Print a sample of all Floor 4 prim names for inspection
    print("\n  Sample of ALL Floor 4 prim names (first 30):")
    for e in floor4_meshes[:30]:
        print(f"    {e['name']!r:40s}  {e['path']}")
else:
    print(f"\n{'#':>4}  {'Path':<70}  {'X':>8}  {'Y':>8}  {'Z':>6}  BBox W×D×H")
    print("-" * 120)
    for i, e in enumerate(door_prims):
        bbox_str = f"{e['bbox'][0]:.2f}×{e['bbox'][1]:.2f}×{e['bbox'][2]:.2f}"
        print(f"{i+1:>4}  {e['path']:<70}  {e['x']:>8.2f}  {e['y']:>8.2f}  "
              f"{e['z']:>6.2f}  {bbox_str}")

# ---------------------------------------------------------------------------
# Copy-pasteable DOOR_PRIMS list for nova_carter_door_collection.py
# ---------------------------------------------------------------------------

if door_prims:
    print(f"\n--- DOOR_PRIMS list for nova_carter_door_collection.py ---")
    print("DOOR_PRIMS = [")
    for e in door_prims:
        print(f'    "{e["path"]}",')
    print("]")

print("\n" + "=" * 70)
print("Diagnostic complete. Review output above and update DOOR_PRIMS in")
print("simulations/robot/nova_carter_door_collection.py before running data collection.")
print("=" * 70)

simulation_app.close()
