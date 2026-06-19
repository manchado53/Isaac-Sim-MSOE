#!/usr/bin/env python3
"""Find XY centroid of building geometry at known floor levels."""
import os
os.environ["OMNI_KIT_ACCEPT_EULA"] = "YES"
from isaacsim import SimulationApp
simulation_app = SimulationApp({"headless": True})

from pxr import Usd, UsdGeom

_project_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
usd_path = os.path.join(_project_dir, "MSOEBuildingV4_physics.usd")
stage = Usd.Stage.Open(usd_path)

# Collect XY of mesh prims at the known floor levels (Z between 10 and 35)
xs, ys = [], []
for prim in stage.Traverse():
    if not prim.IsA(UsdGeom.Mesh):
        continue
    try:
        mat = UsdGeom.Xformable(prim).ComputeLocalToWorldTransform(Usd.TimeCode.Default())
        t = mat.ExtractTranslation()
        if 10 < t[2] < 35 and abs(t[0]) < 1e30 and abs(t[1]) < 1e30:
            xs.append(t[0])
            ys.append(t[1])
    except Exception:
        continue

print(f"Mesh prims at floor levels (Z 10-35): {len(xs)}")
if xs:
    print(f"X range: {min(xs):.2f}  to  {max(xs):.2f}")
    print(f"Y range: {min(ys):.2f}  to  {max(ys):.2f}")

    # Use median instead of mean to ignore outliers
    xs_sorted = sorted(xs)
    ys_sorted = sorted(ys)
    mx = xs_sorted[len(xs_sorted)//2]
    my = ys_sorted[len(ys_sorted)//2]
    print(f"Median XY: ({mx:.2f}, {my:.2f})")

    # Also find the XY of the most-clustered 50% of prims
    x_lo, x_hi = xs_sorted[len(xs_sorted)//4], xs_sorted[3*len(xs_sorted)//4]
    y_lo, y_hi = ys_sorted[len(ys_sorted)//4], ys_sorted[3*len(ys_sorted)//4]
    cx = (x_lo + x_hi) / 2
    cy = (y_lo + y_hi) / 2
    print(f"IQR center XY: ({cx:.2f}, {cy:.2f})")
    print(f"IQR X: {x_lo:.2f} to {x_hi:.2f}  (span {x_hi-x_lo:.1f})")
    print(f"IQR Y: {y_lo:.2f} to {y_hi:.2f}  (span {y_hi-y_lo:.1f})")

    print(f"\n=== Suggested spawn ===")
    print(f"ROBOT_SPAWN_XY = ({cx:.1f}, {cy:.1f})")
    print(f"ROBOT_SPAWN_Z  = 14.5  (just above floor level Z=13.6)")

simulation_app.close()
