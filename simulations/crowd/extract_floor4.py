"""Extract Floor 4 from Innovation Center USD with no plants.

Deactivates:
  - All Planting geometry (trees/shrubs)
  - Mesh prims whose world-space bounding box is entirely outside Floor 4 Z range

Output: assets/environments/msoe_building/floor4_no_plants.usd
Run: sbatch simulations/crowd/submit_extract_floor4.sh
"""
import os
os.environ["OMNI_KIT_ACCEPT_EULA"] = "YES"
os.environ["PRIVACY_CONSENT"] = "Y"

from isaacsim import SimulationApp
app = SimulationApp({"headless": True})

from pxr import Usd, UsdGeom, Gf

PROJECT_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), "../.."))
BUILDING_USD = os.path.join(PROJECT_DIR, "assets/environments/msoe_building/innovation_center_finalV1.usd")
OUT_USD      = os.path.join(PROJECT_DIR, "assets/environments/msoe_building/floor4_no_plants.usd")

# Floor 4: slab at Z=26.8m, next floor ~31m. Add margin for walls/columns.
FLOOR4_Z_MIN = 23.0
FLOOR4_Z_MAX = 33.0

GEOM_ROOT    = "/World/Building/_22_129___MSOE_Engineering_Building___A___2_25_2026/Geometry"
PLANTING_PATH = f"{GEOM_ROOT}/Planting"

print(f"Opening: {BUILDING_USD}")
stage = Usd.Stage.Open(BUILDING_USD)
print("Stage opened")

bbox_cache = UsdGeom.BBoxCache(
    Usd.TimeCode.Default(),
    [UsdGeom.Tokens.default_]
)

deactivated_plants = 0
deactivated_z      = 0
kept               = 0
errors             = 0

for prim in stage.Traverse():
    path  = str(prim.GetPath())
    tname = prim.GetTypeName()

    # Deactivate entire Planting subtree at root
    if path == PLANTING_PATH:
        prim.SetActive(False)
        deactivated_plants += 1
        print(f"Deactivated Planting subtree: {path}")
        continue

    # Skip prims inside Planting (already deactivated via parent)
    if path.startswith(PLANTING_PATH + "/"):
        continue

    # Only Z-filter mesh prims inside Geometry
    if tname != "Mesh" or not path.startswith(GEOM_ROOT):
        continue

    try:
        bbox  = bbox_cache.ComputeWorldBound(prim)
        rng   = bbox.GetRange()
        z_min = rng.GetMin()[2]
        z_max = rng.GetMax()[2]

        if z_max < FLOOR4_Z_MIN or z_min > FLOOR4_Z_MAX:
            prim.SetActive(False)
            deactivated_z += 1
        else:
            kept += 1
    except Exception as e:
        errors += 1

print(f"\nPlanting subtrees deactivated : {deactivated_plants}")
print(f"Meshes deactivated (Z filter) : {deactivated_z}")
print(f"Meshes kept (Floor 4)         : {kept}")
print(f"Bbox errors (kept by default) : {errors}")

os.makedirs(os.path.dirname(OUT_USD), exist_ok=True)
stage.Export(OUT_USD)
print(f"\nSaved: {OUT_USD}")

app.close()
