"""Soft ambient light only — no sun, no flicker, enough to see inside.

Run from Isaac Sim Script Editor:
  exec(open('/home/ad.msoe.edu/somed/isaac-sim-msoe/simulations/crowd/add_daylight.py').read())
"""
import omni.usd
from pxr import UsdLux, Gf

stage = omni.usd.get_context().get_stage()

for p in ["/World/FloodDomeLight", "/World/DomeLight", "/World/SunLight"]:
    prim = stage.GetPrimAtPath(p)
    if prim.IsValid():
        stage.RemovePrim(p)

print("All custom lights removed.")
