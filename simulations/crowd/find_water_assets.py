"""Search Isaac Sim install for water/liquid MDL materials and textures.

exec(open('/home/ad.msoe.edu/somed/isaac-sim-msoe/simulations/crowd/find_water_assets.py').read())
"""
import os, glob

base = os.path.expanduser("~/isaacsim")
results = []

for root, dirs, files in os.walk(base):
    for f in files:
        fl = f.lower()
        if any(kw in fl for kw in ["water", "liquid", "ocean", "fluid", "wave"]):
            full = os.path.join(root, f)
            results.append(full)

if results:
    for r in results:
        print(r)
    print(f"\n{len(results)} files found")
else:
    print("Nothing found")
