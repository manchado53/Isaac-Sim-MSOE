"""Search Isaac Sim install for water/ocean USD and MDL files only.

exec(open('/home/ad.msoe.edu/somed/isaac-sim-msoe/simulations/crowd/find_water_assets.py').read())
"""
import os

base = os.path.expanduser("~/isaacsim")
for root, dirs, files in os.walk(base):
    for f in files:
        fl = f.lower()
        if any(kw in fl for kw in ["water", "ocean", "liquid", "fluid", "wave"]):
            if f.endswith((".usd", ".usda", ".usdc", ".mdl")):
                print(os.path.join(root, f))
