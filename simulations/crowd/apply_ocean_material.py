"""Inspect ocean_material.usda then apply it to WaterPlane.

exec(open('/home/ad.msoe.edu/somed/isaac-sim-msoe/simulations/crowd/apply_ocean_material.py').read())
"""
import omni.usd
from pxr import Usd, UsdShade, Sdf

OCEAN_MAT = "/home/ad.msoe.edu/somed/isaacsim/extscache/omni.warp-1.8.2/data/scenes/assets/materials/ocean_material.usda"

# Peek inside to find material prim paths
tmp = Usd.Stage.Open(OCEAN_MAT)
print("=== Prims in ocean_material.usda ===")
for p in tmp.Traverse():
    print(f"  {p.GetPath()}  [{p.GetTypeName()}]")
