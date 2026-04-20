"""Inspect ocean.usda prim structure.

exec(open('/home/ad.msoe.edu/somed/isaac-sim-msoe/simulations/crowd/inspect_ocean.py').read())
"""
from pxr import Usd, UsdGeom

OCEAN_USD = "/home/ad.msoe.edu/somed/isaacsim/extscache/omni.warp-1.8.2/data/scenes/ocean.usda"

stage = Usd.Stage.Open(OCEAN_USD)
for p in stage.Traverse():
    prim = stage.GetPrimAtPath(p.GetPath())
    print(f"{p.GetPath()}  [{p.GetTypeName()}]")
    # Print xform if it has one
    if prim.IsA(UsdGeom.Xformable):
        xf = UsdGeom.Xformable(prim)
        ops = xf.GetOrderedXformOps()
        for op in ops:
            print(f"    xformOp: {op.GetOpName()} = {op.Get()}")
