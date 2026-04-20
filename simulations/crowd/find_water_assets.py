"""Search for omni.ocean extension and all ocean USD files.

exec(open('/home/ad.msoe.edu/somed/isaac-sim-msoe/simulations/crowd/find_water_assets.py').read())
"""
import os, glob

base = os.path.expanduser("~/isaacsim")

# Check for omni.ocean extension
print("=== omni.ocean extension ===")
ocean_ext = glob.glob(os.path.join(base, "extscache", "omni.ocean*"), recursive=False)
for p in ocean_ext:
    print(p)
if not ocean_ext:
    print("NOT FOUND")

# List all USD files in warp ocean scenes folder
print("\n=== Warp ocean scene files ===")
warp_scenes = os.path.join(base, "extscache/omni.warp-1.8.2/data/scenes")
for f in os.listdir(warp_scenes):
    print(os.path.join(warp_scenes, f))
