"""Open original innovation center headlessly, remove plants, save clean copy.

cd ~/isaacsim && ./python.sh ~/isaac-sim-msoe/simulations/crowd/headless_remove_plants.py
"""
import os
os.environ["OMNI_KIT_ACCEPT_EULA"] = "YES"

from isaacsim import SimulationApp
simulation_app = SimulationApp({"headless": True})

from pxr import Usd, UsdGeom, Sdf

ORIG = "/home/ad.msoe.edu/somed/isaac-sim-msoe/assets/environments/msoe_building/innovation_center_finalV1.usd"
OUT  = "/home/ad.msoe.edu/somed/isaac-sim-msoe/assets/environments/msoe_building/innovation_center_clean.usd"

PLANT_KEYWORDS = ["plant","Plant","tree","Tree","shrub","Shrub","fern","Fern",
                  "bush","Bush","palm","Palm","ivy","Ivy","moss","Moss",
                  "flower","Flower","grass","Grass","vegetation","Vegetation",
                  "Planting","planting","Landscape","landscape"]

SAFE_KEYWORDS  = ["floor","Floor","ceil","Ceil","wall","Wall","slab","Slab",
                  "stair","Stair","beam","Beam","column","Column","roof","Roof",
                  "ground","Ground","deck","Deck","struct","Struct","ramp","Ramp"]

print("Opening original USD via Usd.Stage.Open (may take ~60s)...")
stage = Usd.Stage.Open(ORIG)
all_prims = list(stage.Traverse())
print(f"Stage loaded — {len(all_prims)} total prims")

to_remove = []
for prim in stage.Traverse():
    name = prim.GetName()
    path = str(prim.GetPath())
    if not any(k in name for k in PLANT_KEYWORDS):
        continue
    parts = path.split("/")
    if any(any(s in part for s in SAFE_KEYWORDS) for part in parts):
        continue
    to_remove.append(path)

def is_child_of_any(path, paths):
    return any(path != p and path.startswith(p + "/") for p in paths)

top_level = [p for p in to_remove if not is_child_of_any(p, to_remove)]
print(f"Removing {len(top_level)} plant prims...")

for path in top_level:
    stage.RemovePrim(path)

print(f"Saving to {OUT}...")
stage.GetRootLayer().Export(OUT)
print("Done.")
simulation_app.close()
