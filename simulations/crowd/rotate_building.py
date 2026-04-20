"""Rotate Building -90 on X to match ocean.usda Y-up coordinate space.

exec(open('/home/ad.msoe.edu/somed/isaac-sim-msoe/simulations/crowd/rotate_building.py').read())
"""
import omni.usd
from pxr import UsdGeom, Gf

stage = omni.usd.get_context().get_stage()
building = stage.GetPrimAtPath("/World/Building")

xf = UsdGeom.Xformable(building)
xf.ClearXformOpOrder()
print("Building rotation cleared.")
