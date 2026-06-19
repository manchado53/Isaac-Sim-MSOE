#!/usr/bin/env python3
"""
C1 Wall Exploration — map wall mesh prims near Doors_13606713.

All spatial values are in USD native units (cm) because BBoxCache operates
on the raw USD stage regardless of World(stage_units_in_meters=1.0).

Door reference values from catalog:
    mesh translation: (-2047.73, 8014.94, 2677.16) cm
    center_z: 2803 cm  (= 28.03 m)
    floor 5 z range: ~2650–2950 cm

Run:
    sbatch submit_isaac_job.sh simulations/robot/c1_wall_exploration.py
"""

import os
import math
import functools

print = functools.partial(print, flush=True)

os.environ["OMNI_KIT_ACCEPT_EULA"] = "YES"

import time as _time
_t0 = _time.time()
def _ts(msg):
    print(f"[C1Explore] [+{_time.time()-_t0:6.1f}s] {msg}")

_ts("Importing SimulationApp ...")
from isaacsim import SimulationApp
simulation_app = SimulationApp({"headless": True, "hide_ui": True})
_ts("SimulationApp ready")

import omni.usd
from pxr import Usd, UsdGeom, UsdShade
from isaacsim.core.api import World

_script_dir  = os.path.dirname(os.path.abspath(__file__))
_project_dir = os.path.abspath(os.path.join(_script_dir, "..", ".."))
USD_PATH     = os.path.join(_project_dir, "MSOEBuildingV4_physics.usd")

# All in USD cm units
DOOR_CENTER_CM  = (-2047.73, 8014.94)   # door XY in cm
FLOOR5_Z_RANGE  = (2600.0,  3000.0)     # floor-5 z band in cm (~26–30 m)
PROXIMITY_CM    = 500.0                  # 5 m search radius in cm

BUILDING_ROOT = ("/World/Building"
                 "/_22_129___MSOE_Engineering_Building___A___2_25_2026")

_ts("Loading building USD ...")
world = World(stage_units_in_meters=1.0)
stage = omni.usd.get_context().get_stage()
stage.GetRootLayer().subLayerPaths.insert(0, USD_PATH)
world.reset()
for _ in range(5):
    simulation_app.update()
_ts("Stage ready")

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------
_bbox_cache = UsdGeom.BBoxCache(Usd.TimeCode.Default(), ["default", "render"])

def _bbox(prim):
    """Returns (cx,cy,cz,sx,sy,sz) all in USD cm, or None if empty."""
    bb  = _bbox_cache.ComputeWorldBound(prim)
    rng = bb.GetRange()
    if rng.IsEmpty():
        return None
    mn, mx = rng.GetMin(), rng.GetMax()
    return ((mn[0]+mx[0])/2, (mn[1]+mx[1])/2, (mn[2]+mx[2])/2,
            mx[0]-mn[0],     mx[1]-mn[1],     mx[2]-mn[2])

def _dist_cm(cx, cy):
    return math.sqrt((cx - DOOR_CENTER_CM[0])**2 + (cy - DOOR_CENTER_CM[1])**2)

def _mat(prim):
    b = UsdShade.MaterialBindingAPI(prim).GetDirectBinding()
    return str(b.GetMaterialPath()) if b.GetMaterialPath() else "none"

# ---------------------------------------------------------------------------
# Step 1 — geometry categories
# ---------------------------------------------------------------------------
print(f"\n{'='*70}")
print("STEP 1 — Geometry categories")
print(f"{'='*70}")
geom_prim = stage.GetPrimAtPath(f"{BUILDING_ROOT}/Geometry")
if geom_prim.IsValid():
    for child in sorted(geom_prim.GetChildren(), key=lambda p: p.GetName()):
        n = sum(1 for p in Usd.PrimRange(child) if p.IsA(UsdGeom.Mesh))
        print(f"  {child.GetName():<55}  meshes={n:4d}")
else:
    print("  /Geometry not found — listing building root children:")
    bp = stage.GetPrimAtPath(BUILDING_ROOT)
    for child in (bp.GetChildren() if bp.IsValid() else []):
        print(f"  {child.GetName()}")

building_prim = stage.GetPrimAtPath(BUILDING_ROOT)
if not building_prim.IsValid():
    building_prim = stage.GetPrimAtPath("/World")

# ---------------------------------------------------------------------------
# Step 2 — wall instance prims near the door (read xformOp translation
#           directly — BBoxCache misses the instance-level transform offset)
# ---------------------------------------------------------------------------
print(f"\n{'='*70}")
print(f"STEP 2 — Wall instance prims (Walls_XXXXXXX) within {PROXIMITY_CM} cm of door")
print(f"         Reading xformOp:transform row-3 for world translation.")
print(f"         door centre: {DOOR_CENTER_CM} cm  |  z range: {FLOOR5_Z_RANGE} cm")
print(f"{'='*70}")

nearby    = []
all_wall_instances = 0

walls_root = stage.GetPrimAtPath(f"{BUILDING_ROOT}/Geometry/Walls")

def _xform_translation(prim):
    """Read translation from xformOp:transform row 3, or xformOp:translate."""
    xf = UsdGeom.Xformable(prim)
    for op in xf.GetOrderedXformOps():
        name = op.GetOpName()
        if name == "xformOp:transform":
            m = op.Get()
            if m:
                return m[3][0], m[3][1], m[3][2]
        if name == "xformOp:translate":
            v = op.Get()
            if v:
                return v[0], v[1], v[2]
    return None

if walls_root.IsValid():
    for prim in Usd.PrimRange(walls_root):
        # Target the Walls_XXXXXXX instance level (name starts with "Walls_")
        if not prim.GetName().startswith("Walls_"):
            continue
        all_wall_instances += 1
        t = _xform_translation(prim)
        if t is None:
            continue
        tx, ty, tz = t
        if not (FLOOR5_Z_RANGE[0] <= tz <= FLOOR5_Z_RANGE[1]):
            continue
        d = _dist_cm(tx, ty)
        if d > PROXIMITY_CM:
            continue
        # Collect mesh children + their materials
        meshes = []
        for child in Usd.PrimRange(prim):
            if child.IsA(UsdGeom.Mesh):
                meshes.append({"name": child.GetName(),
                                "path": str(child.GetPath()),
                                "mat":  _mat(child)})
        nearby.append({"wall_path": str(prim.GetPath()),
                       "wall_name": prim.GetName(),
                       "tx": tx, "ty": ty, "tz": tz,
                       "dist": d, "meshes": meshes})
else:
    print("  /Geometry/Walls not found — falling back to building-wide search")
    for prim in Usd.PrimRange(building_prim):
        if not prim.GetName().startswith("Walls_"):
            continue
        if "wall" not in str(prim.GetPath()).lower():
            continue
        all_wall_instances += 1
        t = _xform_translation(prim)
        if t is None:
            continue
        tx, ty, tz = t
        if not (FLOOR5_Z_RANGE[0] <= tz <= FLOOR5_Z_RANGE[1]):
            continue
        d = _dist_cm(tx, ty)
        if d > PROXIMITY_CM:
            continue
        meshes = [{"name": c.GetName(), "path": str(c.GetPath()),
                   "mat": _mat(c)}
                  for c in Usd.PrimRange(prim) if c.IsA(UsdGeom.Mesh)]
        nearby.append({"wall_path": str(prim.GetPath()),
                       "wall_name": prim.GetName(),
                       "tx": tx, "ty": ty, "tz": tz,
                       "dist": d, "meshes": meshes})

print(f"Total Walls_XXXXXXX instance prims scanned: {all_wall_instances}")
print(f"Nearby (floor 5, ≤{PROXIMITY_CM} cm):       {len(nearby)}")

if nearby:
    print()
    for w in sorted(nearby, key=lambda x: x["dist"]):
        print(f"  [{w['dist']:.0f} cm = {w['dist']/100:.1f} m]  "
              f"{w['wall_name']}  pos=({w['tx']:.0f},{w['ty']:.0f},{w['tz']:.0f}) cm")
        print(f"    path: {w['wall_path']}")
        for m in w["meshes"]:
            print(f"    mesh: {m['name']}")
            print(f"      mat: {m['mat'][:80]}")
else:
    # Diagnostic: print z distribution of all Walls_ instance translations
    print()
    print("No instances in range. Walls_ instance z-translation sample:")
    z_vals = []
    src = walls_root if walls_root.IsValid() else building_prim
    for prim in Usd.PrimRange(src):
        if not prim.GetName().startswith("Walls_"):
            continue
        t = _xform_translation(prim)
        if t:
            z_vals.append(t[2])
    if z_vals:
        z_vals.sort()
        print(f"  Count={len(z_vals)}  Min z={z_vals[0]:.0f} cm  Max z={z_vals[-1]:.0f} cm")
        for i in range(0, len(z_vals), max(1, len(z_vals)//20)):
            print(f"    z={z_vals[i]:8.0f} cm  ({z_vals[i]/100:.1f} m)")
    else:
        print("  No Walls_ prims found with xformOp translations.")

# ---------------------------------------------------------------------------
# Step 3 — any large mesh near door (catches monolithic slabs without 'wall')
# ---------------------------------------------------------------------------
print(f"\n{'='*70}")
print(f"STEP 3 — Any non-Walls prim type near door (Rooms, Curtain_Panels, etc.)")
print(f"         Using xformOp translation. Categories: Rooms, Curtain_Panels,")
print(f"         Curtain_Wall_Mullions, Floors, Ceilings")
print(f"{'='*70}")

OTHER_CATS = ["Rooms", "Curtain_Panels", "Curtain_Wall_Mullions", "Floors", "Ceilings"]
for cat in OTHER_CATS:
    cat_prim = stage.GetPrimAtPath(f"{BUILDING_ROOT}/Geometry/{cat}")
    if not cat_prim.IsValid():
        print(f"  {cat}: not found")
        continue
    found = []
    for prim in Usd.PrimRange(cat_prim):
        # Check direct prim xform translation
        t = _xform_translation(prim)
        if t is None:
            continue
        tx, ty, tz = t
        if not (FLOOR5_Z_RANGE[0] <= tz <= FLOOR5_Z_RANGE[1]):
            continue
        d = _dist_cm(tx, ty)
        if d > PROXIMITY_CM:
            continue
        found.append({"name": prim.GetName(), "path": str(prim.GetPath()),
                      "tx": tx, "ty": ty, "tz": tz, "dist": d,
                      "is_mesh": prim.IsA(UsdGeom.Mesh), "mat": _mat(prim)})
    print(f"  {cat}: {len(found)} prims near door")
    for f in sorted(found, key=lambda x: x["dist"])[:5]:
        print(f"    [{f['dist']:.0f} cm]  {f['name']}  "
              f"pos=({f['tx']:.0f},{f['ty']:.0f},{f['tz']:.0f})")
        print(f"      path: {f['path']}")
        if f["is_mesh"]:
            print(f"      mat:  {f['mat'][:80]}")

_ts("Exploration complete")
simulation_app.close()
