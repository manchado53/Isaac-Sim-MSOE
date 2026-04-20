"""Remove plant/tree prims from the loaded stage in-place.

Finds prims whose name contains plant/tree/shrub/fern keywords and removes them.
Safe to run multiple times. Does NOT save — use File > Save As to keep a clean copy.

exec(open('/home/ad.msoe.edu/somed/isaac-sim-msoe/simulations/crowd/remove_plants.py').read())
"""
import omni.usd
from pxr import Usd

stage = omni.usd.get_context().get_stage()

KEYWORDS = [
    "plant", "Plant", "tree", "Tree", "shrub", "Shrub",
    "fern", "Fern", "bush", "Bush", "vegetation", "Vegetation",
    "foliage", "Foliage", "flower", "Flower", "grass", "Grass",
    "palm", "Palm", "ivy", "Ivy", "moss", "Moss",
]

to_remove = []
for prim in stage.Traverse():
    name = prim.GetName()
    if any(k in name for k in KEYWORDS):
        to_remove.append(str(prim.GetPath()))

# Deduplicate — only keep top-level matches (children auto-removed)
def is_child_of_any(path, paths):
    for p in paths:
        if path != p and path.startswith(p + "/"):
            return True
    return False

top_level = [p for p in to_remove if not is_child_of_any(p, to_remove)]

print(f"Found {len(to_remove)} plant prims, {len(top_level)} top-level roots:")
for p in top_level:
    print(f"  {p}")

print(f"\nRemoving {len(top_level)} prims...")
for path in top_level:
    stage.RemovePrim(path)
    print(f"  removed {path}")

print(f"\nDone — {len(top_level)} plant prims removed.")
print("Use File > Save As to save a clean copy.")
