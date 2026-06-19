#!/usr/bin/env python3
"""Find every Mesh whose bbox overlaps the floor-4 problem region and report
its category ancestry + whether it has CollisionAPI.

Region: x in [-70,-35], y in [75,90], z in [26,31]
"""
import os, sys
from collections import defaultdict
from pxr import Usd, UsdGeom, UsdPhysics

_HERE = os.path.dirname(os.path.abspath(__file__))
_ROOT = os.path.abspath(os.path.join(_HERE, ".."))
USD = os.path.join(_ROOT, "assets/environments/msoe_building/MSOEBuildingV4_physics.usd")
if len(sys.argv) > 1:
    USD = sys.argv[1]
XR, YR, ZR = (-70.0, -35.0), (75.0, 90.0), (26.0, 31.0)

stage = Usd.Stage.Open(USD)
bcache = UsdGeom.BBoxCache(Usd.TimeCode.Default(), [UsdGeom.Tokens.default_],
                           useExtentsHint=True)

# The root of building hierarchy
GEO = "/World/Building/_22_129___MSOE_Engineering_Building___A___2_25_2026/Geometry"

def category_folder(prim):
    """Walk up to find the Geometry/<Folder> ancestor name."""
    p = prim
    while p and p.GetParent():
        parent = p.GetParent()
        if str(parent.GetPath()) == GEO:
            return p.GetName()
        p = parent
    return "<unknown>"

# Counts
total_meshes = 0
in_region = 0
cat_stats = defaultdict(lambda: {"total": 0, "coll": 0, "no_coll": []})

# Use PrimRange but de-instance as we go so instanced meshes are visible
def walk(prim):
    if prim.IsInstanceable():
        prim.SetInstanceable(False)
    yield prim
    for c in prim.GetChildren():
        yield from walk(c)

geo_prim = stage.GetPrimAtPath(GEO)
for prim in walk(geo_prim):
    if not prim.IsA(UsdGeom.Mesh):
        continue
    total_meshes += 1
    bbox = bcache.ComputeWorldBound(prim)
    r = bbox.ComputeAlignedRange()
    if r.IsEmpty():
        continue
    mn, mx = r.GetMin(), r.GetMax()
    if (mx[0] < XR[0] or mn[0] > XR[1] or
        mx[1] < YR[0] or mn[1] > YR[1] or
        mx[2] < ZR[0] or mn[2] > ZR[1]):
        continue
    in_region += 1
    cat = category_folder(prim)
    has = prim.HasAPI(UsdPhysics.CollisionAPI)
    s = cat_stats[cat]
    s["total"] += 1
    if has:
        s["coll"] += 1
    else:
        s["no_coll"].append((prim.GetName(), str(prim.GetPath()), mn, mx))

print(f"Total meshes in stage: {total_meshes}")
print(f"Meshes overlapping region: {in_region}\n")
print(f"{'Category':<28s} {'Meshes':>8s} {'WithColl':>10s} {'NoColl':>8s}")
print("-" * 60)
for cat in sorted(cat_stats):
    s = cat_stats[cat]
    print(f"{cat:<28s} {s['total']:>8d} {s['coll']:>10d} {len(s['no_coll']):>8d}")

print("\nUncolliderized meshes (sample of 30):")
count = 0
for cat in sorted(cat_stats):
    for name, path, mn, mx in cat_stats[cat]["no_coll"]:
        print(f"  [{cat:<22s}] {name[:55]:<55s} z=[{mn[2]:.1f},{mx[2]:.1f}]")
        count += 1
        if count >= 30:
            break
    if count >= 30:
        break
