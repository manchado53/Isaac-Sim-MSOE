"""Safely remove only plant/tree prims — skips anything that looks like a floor/structure.

Load innovation_center_finalV1.usd first, then run this, then File > Save As.

exec(open('/home/ad.msoe.edu/somed/isaac-sim-msoe/simulations/crowd/remove_plants_safe.py').read())
"""
import omni.usd
from pxr import UsdGeom

stage = omni.usd.get_context().get_stage()

PLANT_KEYWORDS = [
    "plant", "Plant", "tree", "Tree", "shrub", "Shrub",
    "fern", "Fern", "bush", "Bush", "palm", "Palm",
    "ivy", "Ivy", "moss", "Moss", "flower", "Flower",
]

# Never remove anything with these in the name — floors, ceilings, walls, structure
SAFE_KEYWORDS = [
    "floor", "Floor", "ceiling", "Ceiling", "wall", "Wall",
    "slab", "Slab", "stair", "Stair", "ramp", "Ramp",
    "beam", "Beam", "column", "Column", "struct", "Struct",
    "ground", "Ground", "deck", "Deck", "roof", "Roof",
]

to_remove = []
skipped = []

for prim in stage.Traverse():
    name = prim.GetName()
    path = str(prim.GetPath())

    is_plant = any(k in name for k in PLANT_KEYWORDS)
    if not is_plant:
        continue

    # Check if any ancestor or the prim itself has a safe keyword
    parts = path.split("/")
    is_structural = any(
        any(s in part for s in SAFE_KEYWORDS)
        for part in parts
    )

    if is_structural:
        skipped.append(path)
    else:
        to_remove.append(path)

# Keep only top-level (children auto-removed)
def is_child_of_any(path, paths):
    return any(path != p and path.startswith(p + "/") for p in paths)

top_level = [p for p in to_remove if not is_child_of_any(p, to_remove)]

print(f"Safe to remove: {len(top_level)} plant prims")
print(f"Skipped (structural parent): {len(skipped)}")
if skipped:
    for s in skipped[:10]:
        print(f"  KEPT: {s}")

print(f"\nRemoving...")
for path in top_level:
    stage.RemovePrim(path)
    print(f"  removed {path}")

print(f"\nDone. Use File > Save As -> 'innovation_center_no_plants.usd'")
