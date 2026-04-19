#!/usr/bin/env python3
"""List all descendants of /Specialty_Equipment/Soccer_Goal with their types."""
import os
os.environ["OMNI_KIT_ACCEPT_EULA"] = "YES"
from isaacsim import SimulationApp
simulation_app = SimulationApp({"headless": True})

from pxr import Usd, UsdGeom

_project_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
USD = os.path.join(_project_dir, "assets", "environments", "msoe_building",
                   "VietsField_physics.usd")
stage = Usd.Stage.Open(USD)
cache = UsdGeom.BBoxCache(Usd.TimeCode.Default(),
                          [UsdGeom.Tokens.default_, UsdGeom.Tokens.render])

print("=== All descendants of /Specialty_Equipment/Soccer_Goal ===")
root = stage.GetPrimAtPath(
    "/World/Building/_20_104_MSOE_VF_v20/Geometry/Specialty_Equipment/Soccer_Goal")
if root:
    for prim in Usd.PrimRange(root):
        try:
            rng = cache.ComputeWorldBound(prim).ComputeAlignedRange()
            if rng.IsEmpty():
                bb = "EMPTY"
            else:
                bmin, bmax = rng.GetMin(), rng.GetMax()
                ext = [bmax[i] - bmin[i] for i in range(3)]
                cx = 0.5 * (bmin[0] + bmax[0])
                cy = 0.5 * (bmin[1] + bmax[1])
                cz = 0.5 * (bmin[2] + bmax[2])
                bb = (f"c=({cx:6.1f},{cy:6.1f},{cz:5.1f}) "
                      f"e=({ext[0]:5.1f},{ext[1]:5.1f},{ext[2]:5.1f})")
        except Exception:
            bb = "ERR"
        print(f"  type={prim.GetTypeName():10s} {bb}  {prim.GetPath()}")

simulation_app.close()
