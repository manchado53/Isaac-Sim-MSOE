#!/usr/bin/env python3
"""
Inspect MSOEBuildingTake2.usd to find door prims.
Checks: prim names, hierarchy, Revit metadata attributes, and geometry dimensions.

Run with:
  srun --partition=teaching --gres=gpu:1 --time=00:30:00 --pty bash
  cd ~/isaacsim && ./python.sh isaac-sim-msoe/test_building/inspect_door_prims.py
"""

import os
import sys

os.environ["OMNI_KIT_ACCEPT_EULA"] = "YES"

from isaacsim import SimulationApp
simulation_app = SimulationApp({"headless": True})

import omni
from pxr import Usd, UsdGeom, Gf

PROJECT_DIR = os.path.dirname(os.path.abspath(__file__))
USD_PATH = os.path.join(os.path.dirname(PROJECT_DIR),
                        "assets", "environments", "msoe_building", "MSOEBuildingTake2.usd")

print(f"Loading USD: {USD_PATH}")
omni.usd.get_context().open_stage(USD_PATH)
for i in range(3):
    simulation_app.update()

stage = omni.usd.get_context().get_stage()
if not stage:
    print("ERROR: Failed to open stage")
    simulation_app.close()
    sys.exit(1)

prim_count = len(list(stage.Traverse()))
print(f"Stage loaded: {prim_count} prims\n")

# ---------------------------------------------------------------
# 1. Find prims with "door" in the name (case-insensitive)
# ---------------------------------------------------------------
print("=" * 60)
print("1. Prims with 'door' in name (case-insensitive)")
print("=" * 60)
door_by_name = []
for prim in stage.Traverse():
    if "door" in prim.GetName().lower():
        door_by_name.append(prim)
        print(f"  {prim.GetPath()} [{prim.GetTypeName()}]")

print(f"\n  Total: {len(door_by_name)} prims with 'door' in name\n")

# ---------------------------------------------------------------
# 2. Check for Revit metadata attributes on a sample of prims
# ---------------------------------------------------------------
print("=" * 60)
print("2. Checking for Revit/custom attributes on all mesh prims")
print("=" * 60)

# Collect all unique attribute namespaces/prefixes across mesh prims
attr_prefixes = set()
sample_attrs = {}
mesh_count = 0

for prim in stage.Traverse():
    if prim.IsA(UsdGeom.Mesh):
        mesh_count += 1
        for attr in prim.GetAttributes():
            name = attr.GetName()
            # Skip standard USD/geometry attributes
            if name.startswith(("primvars:", "xformOp", "faceVertex",
                                "points", "normals", "extent", "purpose",
                                "visibility", "doubleSided", "orientation",
                                "subdivisionScheme")):
                continue
            prefix = name.split(":")[0] if ":" in name else name
            if prefix not in attr_prefixes:
                attr_prefixes.add(prefix)
                val = attr.Get()
                sample_attrs[name] = (str(prim.GetPath()), val)

print(f"  Scanned {mesh_count} mesh prims")
print(f"\n  Unique attribute prefixes (non-standard):")
for p in sorted(attr_prefixes):
    print(f"    - {p}")

print(f"\n  Sample custom attributes:")
for attr_name, (path, val) in sorted(sample_attrs.items()):
    print(f"    {attr_name} = {val}  (on {path})")

# ---------------------------------------------------------------
# 3. Check parent hierarchy structure (top 3 levels)
# ---------------------------------------------------------------
print("\n" + "=" * 60)
print("3. Top-level hierarchy (3 levels deep)")
print("=" * 60)

root = stage.GetPseudoRoot()
for child in root.GetChildren():
    print(f"  /{child.GetName()} [{child.GetTypeName()}]")
    for grandchild in child.GetChildren():
        print(f"    /{grandchild.GetName()} [{grandchild.GetTypeName()}]")
        count = len(grandchild.GetChildren())
        if count > 0:
            # Show first few great-grandchildren
            for i, ggc in enumerate(grandchild.GetChildren()):
                if i < 5:
                    print(f"      /{ggc.GetName()} [{ggc.GetTypeName()}]")
                elif i == 5:
                    print(f"      ... ({count - 5} more)")
                    break

# ---------------------------------------------------------------
# 4. If door prims found by name, inspect their attributes & bbox
# ---------------------------------------------------------------
if door_by_name:
    print("\n" + "=" * 60)
    print("4. Door prim details (first 10)")
    print("=" * 60)
    bbox_cache = UsdGeom.BBoxCache(Usd.TimeCode.Default(), ["default", "render"])

    for prim in door_by_name[:10]:
        print(f"\n  Path: {prim.GetPath()}")
        print(f"  Type: {prim.GetTypeName()}")
        print(f"  Parent: {prim.GetParent().GetPath()} [{prim.GetParent().GetTypeName()}]")

        # Bounding box
        try:
            bbox = bbox_cache.ComputeWorldBound(prim)
            bbox_range = bbox.ComputeAlignedRange()
            bmin = bbox_range.GetMin()
            bmax = bbox_range.GetMax()
            size = bmax - bmin
            print(f"  BBox min: ({bmin[0]:.1f}, {bmin[1]:.1f}, {bmin[2]:.1f})")
            print(f"  BBox max: ({bmax[0]:.1f}, {bmax[1]:.1f}, {bmax[2]:.1f})")
            print(f"  Size: ({size[0]:.1f} x {size[1]:.1f} x {size[2]:.1f}) cm")
        except Exception as e:
            print(f"  BBox error: {e}")

        # All attributes
        print(f"  Attributes:")
        for attr in prim.GetAttributes():
            name = attr.GetName()
            if not name.startswith(("primvars:", "xformOp", "faceVertex",
                                     "points", "normals", "extent")):
                val = attr.Get()
                if val is not None:
                    print(f"    {name} = {val}")

# ---------------------------------------------------------------
# 5. Look for prims with door-like geometry dimensions
#    (tall ~200cm, wide ~80-100cm, thin <15cm)
# ---------------------------------------------------------------
print("\n" + "=" * 60)
print("5. Prims with door-like dimensions (thin geometry)")
print("=" * 60)
bbox_cache = UsdGeom.BBoxCache(Usd.TimeCode.Default(), ["default", "render"])
door_like = []

for prim in stage.Traverse():
    if not prim.IsA(UsdGeom.Mesh):
        continue
    try:
        bbox = bbox_cache.ComputeWorldBound(prim)
        bbox_range = bbox.ComputeAlignedRange()
        bmin = bbox_range.GetMin()
        bmax = bbox_range.GetMax()
        sx = bmax[0] - bmin[0]
        sy = bmax[1] - bmin[1]
        sz = bmax[2] - bmin[2]

        dims = sorted([sx, sy, sz])
        thin = dims[0]  # thinnest dimension
        mid = dims[1]
        tall = dims[2]

        # Door-like: thin (<15cm), one dim ~60-120cm (width), one dim ~180-250cm (height)
        if (thin < 15 and
            60 < mid < 130 and
            180 < tall < 260):
            door_like.append((prim, thin, mid, tall))
    except:
        continue

print(f"  Found {len(door_like)} prims with door-like dimensions")
for prim, thin, mid, tall in door_like[:20]:
    print(f"    {prim.GetPath()}")
    print(f"      dims: {thin:.1f} x {mid:.1f} x {tall:.1f} cm (thin x wide x tall)")

print(f"\n{'='*60}")
print("Inspection complete!")
print(f"{'='*60}")
sys.stdout.flush()

simulation_app.close()
