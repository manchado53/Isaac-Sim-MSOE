"""
discover_floor_positions.py

Runs WITHOUT Isaac Sim — plain Python + pxr (USD).
Opens MSOEBuildingV4_physics.usd and prints the world-space coordinates of
every floor mesh origin so we can see exactly what get_floor_positions()
returns before trusting those coordinates for camera placement.

Run with:
    python3 discover_floor_positions.py
"""

from pathlib import Path
from pxr import Gf, Usd, UsdGeom

HERE      = Path(__file__).parent
BUILD_USD = HERE / "MSOEBuildingV4_physics.usd"
BUILDING_BASE = (
    "/World/Building"
    "/_22_129___MSOE_Engineering_Building___A___2_25_2026"
    "/Geometry"
)

print(f"Opening {BUILD_USD.name} ...")
stage = Usd.Stage.Open(str(BUILD_USD))

root = stage.GetPrimAtPath(f"{BUILDING_BASE}/Floors")
if not root:
    print(f"ERROR: No prim at {BUILDING_BASE}/Floors")
    exit(1)

positions = []
for prim in Usd.PrimRange(root):
    if prim.GetTypeName() == "Mesh":
        m = UsdGeom.Xformable(prim).ComputeLocalToWorldTransform(Usd.TimeCode.Default())
        t = m.ExtractTranslation()
        positions.append((float(t[0]), float(t[1]), float(t[2]), prim.GetPath()))

print(f"\nFound {len(positions)} floor mesh(es).\n")
print(f"{'#':>4}  {'X':>8}  {'Y':>8}  {'Z':>8}  Path")
print("-" * 80)
for i, (x, y, z, path) in enumerate(positions):
    print(f"{i:>4}  {x:>8.2f}  {y:>8.2f}  {z:>8.2f}  {str(path)[-50:]}")

xs = [p[0] for p in positions]
ys = [p[1] for p in positions]
zs = [p[2] for p in positions]
print(f"\nX range: {min(xs):.2f} → {max(xs):.2f}")
print(f"Y range: {min(ys):.2f} → {max(ys):.2f}")
print(f"Z range: {min(zs):.2f} → {max(zs):.2f}")
print(f"\nUnique Z values: {sorted(set(round(z,1) for z in zs))}")
