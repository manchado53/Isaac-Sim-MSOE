"""Reference the Warp ocean_material.usda and bind it to WaterPlane.

exec(open('/home/ad.msoe.edu/somed/isaac-sim-msoe/simulations/crowd/apply_ocean_material.py').read())
"""
import omni.usd
from pxr import UsdShade, Sdf

OCEAN_MAT_FILE = "/home/ad.msoe.edu/somed/isaacsim/extscache/omni.warp-1.8.2/data/scenes/assets/materials/ocean_material.usda"

stage = omni.usd.get_context().get_stage()

# Remove old water material
old = stage.GetPrimAtPath("/World/Looks/WaterMaterial")
if old.IsValid():
    stage.RemovePrim("/World/Looks/WaterMaterial")

# Pull ocean material into stage via reference
mat_prim = stage.DefinePrim("/World/Looks/OceanMaterial")
mat_prim.GetReferences().AddReference(OCEAN_MAT_FILE, "/World/OceanMaterial")

# Bind to WaterPlane
water = stage.GetPrimAtPath("/World/WaterPlane")
if not water.IsValid():
    print("ERROR: /World/WaterPlane not found — run full_flood.py first")
else:
    mat = UsdShade.Material(mat_prim)
    UsdShade.MaterialBindingAPI.Apply(water).Bind(mat)
    print("Ocean material applied to WaterPlane.")
