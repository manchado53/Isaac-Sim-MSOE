"""Inspect Innovation Center USD hierarchy to find trees and floor structure.
Output: videos/building_hierarchy.txt
Run: sbatch simulations/crowd/submit_inspect_hierarchy.sh
"""
import os
os.environ["OMNI_KIT_ACCEPT_EULA"] = "YES"
os.environ["PRIVACY_CONSENT"] = "Y"

from isaacsim import SimulationApp
app = SimulationApp({"headless": True})

from pxr import Usd, UsdGeom, Gf
import omni.usd

PROJECT_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), "../.."))
BUILDING_USD = os.path.join(PROJECT_DIR, "assets/environments/msoe_building/innovation_center_finalV1.usd")
OUT_TXT = os.path.join(PROJECT_DIR, "videos/building_hierarchy.txt")

lines = []
def w(s=""): lines.append(str(s)); print(s)

w(f"Opening: {BUILDING_USD}")
stage = Usd.Stage.Open(BUILDING_USD)
w(f"Stage opened: {stage is not None}")
w()

# Collect all prims with their bounding box Z ranges
w("=== Scanning hierarchy (depth <= 5) ===")
tree_candidates = []
floor_prims = []

for prim in stage.Traverse():
    path = str(prim.GetPath())
    tname = prim.GetTypeName()
    depth = len(path.strip("/").split("/"))

    # Print shallow hierarchy
    if depth <= 5:
        w(f"{'  '*depth}{path}  [{tname}]")

    path_lower = path.lower()

    # Flag tree candidates
    if any(kw in path_lower for kw in ["tree", "plant", "vegetation", "foliage", "shrub", "bush", "landscape"]):
        tree_candidates.append((path, tname, depth))

    # Flag floor-related
    if any(kw in path_lower for kw in ["floor", "level", "story", "storey", "f4", "floor4", "04"]):
        floor_prims.append((path, tname, depth))

w()
w(f"=== Tree/vegetation candidates ({len(tree_candidates)}) ===")
for path, tname, depth in tree_candidates[:50]:
    w(f"  {path}  [{tname}]")

w()
w(f"=== Floor-related prims ({len(floor_prims)}) ===")
for path, tname, depth in floor_prims[:50]:
    w(f"  {path}  [{tname}]")

w()
w("=== Top-level children of /World (if exists) ===")
world = stage.GetPrimAtPath("/World")
if world.IsValid():
    for child in world.GetChildren():
        w(f"  {child.GetPath()}  [{child.GetTypeName()}]")
        for grandchild in child.GetChildren():
            w(f"    {grandchild.GetPath()}  [{grandchild.GetTypeName()}]")

w()
w("=== Root layer children ===")
root = stage.GetPseudoRoot()
for child in root.GetChildren():
    w(f"  {child.GetPath()}  [{child.GetTypeName()}]")
    for grandchild in child.GetChildren():
        w(f"    {grandchild.GetPath()}  [{grandchild.GetTypeName()}]")
        for ggc in grandchild.GetChildren():
            w(f"      {ggc.GetPath()}  [{ggc.GetTypeName()}]")

os.makedirs(os.path.dirname(OUT_TXT), exist_ok=True)
with open(OUT_TXT, "w") as f:
    f.write("\n".join(lines))

print(f"\n[INFO] Written to {OUT_TXT}", flush=True)
app.close()
