#!/usr/bin/env python3
"""Quick check: list all Room prims with their Z bounding boxes to find basement rooms."""
import os, sys
os.environ["OMNI_KIT_ACCEPT_EULA"] = "YES"
from isaacsim import SimulationApp
simulation_app = SimulationApp({"headless": True})
import omni
from pxr import Usd, UsdGeom

PROJECT_DIR = os.path.dirname(os.path.abspath(__file__))
USD_PATH = os.path.join(os.path.dirname(PROJECT_DIR),
                        "assets", "environments", "msoe_building", "MSOEBuildingTake2.usd")

print(f"Loading USD: {USD_PATH}")
omni.usd.get_context().open_stage(USD_PATH)
for i in range(3):
    simulation_app.update()
stage = omni.usd.get_context().get_stage()

bbox_cache = UsdGeom.BBoxCache(Usd.TimeCode.Default(), ["default", "render"])

# Collect all Room prims and their Z positions
print("\n" + "=" * 80)
print("All Room prims sorted by Z-min (lowest first)")
print("=" * 80)

rooms = []
for prim in stage.Traverse():
    path_str = str(prim.GetPath())
    if "/Rooms/" not in path_str:
        continue
    if not prim.IsA(UsdGeom.Mesh):
        continue
    try:
        bbox = bbox_cache.ComputeWorldBound(prim)
        bbox_range = bbox.ComputeAlignedRange()
        bmin = bbox_range.GetMin()
        bmax = bbox_range.GetMax()
        rooms.append((bmin[2], bmax[2], prim.GetName(), path_str))
    except:
        pass

rooms.sort(key=lambda x: x[0])
for z_min, z_max, name, path in rooms:
    floor_guess = "BASEMENT" if z_min < 1300 else "FL1" if z_min < 1700 else "FL2" if z_min < 2200 else "FL3"
    print(f"  Z [{z_min:.0f} - {z_max:.0f}] ({floor_guess})  {name}")

# Also check wall prims with B0 in name
print("\n" + "=" * 80)
print("Basement walls (B0 in path) - Z ranges")
print("=" * 80)
b0_walls = []
for prim in stage.Traverse():
    path_str = str(prim.GetPath())
    if "_B0_" not in path_str and "_B_" not in path_str:
        continue
    if not prim.IsA(UsdGeom.Mesh):
        continue
    try:
        bbox = bbox_cache.ComputeWorldBound(prim)
        bbox_range = bbox.ComputeAlignedRange()
        bmin = bbox_range.GetMin()
        bmax = bbox_range.GetMax()
        b0_walls.append((bmin[2], bmax[2], prim.GetName(), path_str))
    except:
        pass

b0_walls.sort(key=lambda x: x[0])
if b0_walls:
    z_mins = [w[0] for w in b0_walls]
    z_maxs = [w[1] for w in b0_walls]
    print(f"  Total B0 walls: {len(b0_walls)}")
    print(f"  Z range: {min(z_mins):.0f} - {max(z_maxs):.0f}")
    print(f"  First 20:")
    for z_min, z_max, name, path in b0_walls[:20]:
        print(f"    Z [{z_min:.0f} - {z_max:.0f}]  {name}")
else:
    print("  No B0 walls found")

# Count all wall meshes in Z bands
print("\n" + "=" * 80)
print("Wall mesh count by Z band")
print("=" * 80)
z_bands = {}
for prim in stage.Traverse():
    path_str = str(prim.GetPath())
    if "/Walls/" not in path_str:
        continue
    if not prim.IsA(UsdGeom.Mesh):
        continue
    try:
        bbox = bbox_cache.ComputeWorldBound(prim)
        bbox_range = bbox.ComputeAlignedRange()
        bmin = bbox_range.GetMin()
        bmax = bbox_range.GetMax()
        # Which Z bands does this wall span?
        for band_start in range(800, 3600, 100):
            band_end = band_start + 100
            if bmin[2] < band_end and bmax[2] > band_start:
                key = f"Z {band_start}-{band_end}"
                z_bands[key] = z_bands.get(key, 0) + 1
    except:
        pass

for key in sorted(z_bands.keys()):
    count = z_bands[key]
    bar = "#" * min(count // 10, 50)
    print(f"  {key}: {count:4d} walls  {bar}")

print("\n" + "=" * 80)
print("Done!")
print("=" * 80)
sys.stdout.flush()
simulation_app.close()
