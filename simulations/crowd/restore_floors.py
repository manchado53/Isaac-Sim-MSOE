"""Headless: compare original vs no_plants USD, restore missing floor/structural prims.

Run with:
cd ~/isaacsim && ./python.sh ~/isaac-sim-msoe/simulations/crowd/restore_floors.py
"""
import os, sys
os.environ["OMNI_KIT_ACCEPT_EULA"] = "YES"

from isaacsim import SimulationApp
simulation_app = SimulationApp({"headless": True})

from pxr import Usd, UsdGeom, Sdf

ORIG  = "/home/ad.msoe.edu/somed/isaac-sim-msoe/assets/environments/msoe_building/innovation_center_finalV1.usd"
CLEAN = "/home/ad.msoe.edu/somed/isaac-sim-msoe/assets/environments/msoe_building/floor4_no_plants.usd"
OUT   = "/home/ad.msoe.edu/somed/isaac-sim-msoe/assets/environments/msoe_building/innovation_center_no_plants.usd"

PLANT_KEYWORDS = ["plant","Plant","tree","Tree","shrub","Shrub","fern","Fern",
                  "bush","Bush","palm","Palm","ivy","Ivy","moss","Moss","flower","Flower"]

FLOOR_KEYWORDS = ["floor","Floor","ceiling","Ceiling","slab","Slab","ground","Ground",
                  "deck","Deck","roof","Roof","stair","Stair"]

print("Opening original USD (this may take a minute)...")
orig  = Usd.Stage.Open(ORIG)
print("Opening no_plants USD...")
clean = Usd.Stage.Open(CLEAN)

orig_paths  = {str(p.GetPath()) for p in orig.Traverse()}
clean_paths = {str(p.GetPath()) for p in clean.Traverse()}
removed     = orig_paths - clean_paths

print(f"Original prims:  {len(orig_paths)}")
print(f"No_plants prims: {len(clean_paths)}")
print(f"Removed prims:   {len(removed)}")

# Find removed prims that look like floors
floor_prims = []
plant_prims = []
other_removed = []

for path in removed:
    parts = path.split("/")
    name = parts[-1]
    is_plant = any(k in name for k in PLANT_KEYWORDS)
    is_floor = any(k in name for k in FLOOR_KEYWORDS)
    if is_floor:
        floor_prims.append(path)
    elif is_plant:
        plant_prims.append(path)
    else:
        other_removed.append(path)

print(f"\nRemoved breakdown:")
print(f"  Plant prims:   {len(plant_prims)}")
print(f"  Floor prims:   {len(floor_prims)}")
print(f"  Other removed: {len(other_removed)}")

if floor_prims:
    print(f"\nMissing floor prims (first 20):")
    for p in floor_prims[:20]:
        print(f"  {p}")

print(f"\nOther removed (first 30):")
for p in sorted(other_removed)[:30]:
    print(f"  {p}")

# Copy floor4_no_plants to output, then restore floor prims from original
print(f"\nCopying {CLEAN} -> {OUT}...")
import shutil
shutil.copy2(CLEAN, OUT)

print("Restoring floor prims from original...")
out_stage = Usd.Stage.Open(OUT)

restored = 0
for path in floor_prims:
    orig_prim = orig.GetPrimAtPath(path)
    if not orig_prim.IsValid():
        continue
    parent_path = str(Sdf.Path(path).GetParentPath())
    parent = out_stage.GetPrimAtPath(parent_path)
    if not parent.IsValid():
        out_stage.DefinePrim(parent_path)
    Usd.ModelAPI(out_stage.OverridePrim(path))
    restored += 1

out_stage.GetRootLayer().Save()
print(f"Restored {restored} floor prims -> {OUT}")
print("Done.")
simulation_app.close()
