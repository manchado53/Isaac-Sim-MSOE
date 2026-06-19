#!/usr/bin/env python3
"""
door_mesh_inspector.py — Diagnostic: print every mesh name and local transform
for the 6 problematic doors in door_hinge_test.py.

For each door this script:
  1. Gets the door prim
  2. Calls SetInstanceable(False) to concretise the subtree
  3. Walks Usd.PrimRange and prints, for every UsdGeom.Mesh:
       - full prim path
       - prim name (last path component)
       - local origin (ox, oy, oz) from xformOp:transform mat[3]
       - local extent (min/max) if a UsdGeom.BBoxCache entry is available

This output tells us:
  a) exact local origins → correct hinge coordinates
  b) prim name conventions → which meshes are door leaf vs sidelite

Run
---
  sbatch submit_isaac_job.sh simulations/robot/door_mesh_inspector.py
"""

import os
import sys
import json
import functools

print = functools.partial(print, flush=True)  # noqa: A001

# ---------------------------------------------------------------------------
# Doors to inspect
# ---------------------------------------------------------------------------

_BASE = (
    "/World/Building/_22_129___MSOE_Engineering_Building___A___2_25_2026"
    "/Geometry/Doors"
)

# Targeting both Type_N doors to identify frame vs leaf mesh names.
# We need to know which mesh prims carry the fixed HM frame so they can
# be excluded from rotation via leaf_mesh_names in door_hinge_test.py.
INSPECT = [
    {
        "id":        "Doors_12590968",
        "label":     "Type_N floor3 train",
        "prim_path": f"{_BASE}/RS___Type_N_Door/NEW___HM_Frame___36_x96/Doors_12590968",
    },
    {
        "id":        "Doors_2429516",
        "label":     "Type_N floor4 train",
        "prim_path": f"{_BASE}/RS___Type_N_Door/NEW___HM_Frame___36_x96/Doors_2429516",
    },
]

# ---------------------------------------------------------------------------
# Isaac Sim startup
# ---------------------------------------------------------------------------

os.environ["OMNI_KIT_ACCEPT_EULA"] = "YES"

import time as _time
_t0 = _time.time()
def _ts(msg):
    print(f"[Inspector] [+{_time.time() - _t0:6.1f}s] {msg}")

_ts("Starting SimulationApp ...")
from isaacsim import SimulationApp
simulation_app = SimulationApp({"headless": True})
_ts("SimulationApp ready")

import omni.usd
from pxr import Usd, UsdGeom, Gf

from isaacsim.core.api import World

_script_dir  = os.path.dirname(os.path.abspath(__file__))
_project_dir = os.path.abspath(os.path.join(_script_dir, "..", ".."))
USD_PATH     = os.path.join(_project_dir, "MSOEBuildingV4_physics.usd")

world = World(stage_units_in_meters=1.0)
stage = omni.usd.get_context().get_stage()
stage.GetRootLayer().subLayerPaths.insert(0, USD_PATH)
_ts("Building USD inserted")

world.reset()
for _ in range(3):
    world.step(render=False)

# ---------------------------------------------------------------------------
# Inspect each door — print FULL prim tree (all types, all xform ops)
# ---------------------------------------------------------------------------

for door in INSPECT:
    print()
    print(f"{'='*70}")
    print(f"  {door['id']}  —  {door['label']}")
    print(f"  prim: {door['prim_path']}")
    print(f"{'='*70}")

    prim = stage.GetPrimAtPath(door["prim_path"])
    if not prim.IsValid():
        print(f"  !! PRIM NOT FOUND — check path")
        continue

    if prim.IsInstance() or prim.IsInstanceable():
        prim.SetInstanceable(False)
        simulation_app.update()
        print(f"  [concretised via SetInstanceable(False)]")

    base_path = door["prim_path"]
    prim_count = 0

    for p in Usd.PrimRange(prim):
        prim_count += 1
        rel  = str(p.GetPath()).replace(base_path, "") or "/"
        kind = p.GetTypeName()   # e.g. "Mesh", "Xform", "Scope", ""

        # Gather all xform ops on this prim
        xf   = UsdGeom.Xformable(p)
        ops  = [o.GetOpName() for o in xf.GetOrderedXformOps()]

        # For prims that have xformOp:transform, read the translation column
        origin_str = ""
        xform_op = next((o for o in xf.GetOrderedXformOps()
                         if o.GetOpName() == "xformOp:transform"), None)
        if xform_op is not None:
            mat = xform_op.Get()
            if mat is not None:
                ox, oy, oz = mat[3][0], mat[3][1], mat[3][2]
                origin_str = f"  origin=({ox:.2f},{oy:.2f},{oz:.2f})"

        # For meshes, try to read the extent attribute (local bbox)
        extent_str = ""
        if p.IsA(UsdGeom.Mesh):
            ext_attr = UsdGeom.Boundable(p).GetExtentAttr()
            if ext_attr and ext_attr.HasValue():
                ext = ext_attr.Get()
                if ext and len(ext) == 2:
                    mn, mx = ext[0], ext[1]
                    extent_str = (f"  extent_x=[{mn[0]:.2f},{mx[0]:.2f}]"
                                  f"  extent_y=[{mn[1]:.2f},{mx[1]:.2f}]"
                                  f"  extent_z=[{mn[2]:.2f},{mx[2]:.2f}]")

        ops_str = f"  ops={ops}" if ops else ""
        print(f"  {rel:<55s}  [{kind:<6s}]{ops_str}{origin_str}{extent_str}")

    print(f"  --- {prim_count} prims total ---")

print()
print("="*70)
_ts("Inspection complete")

world.stop()
simulation_app.close()
