"""Validate building Z bounds then reposition ocean to start just below floor.

exec(open('/home/ad.msoe.edu/somed/isaac-sim-msoe/simulations/crowd/reposition_ocean.py').read())
"""
import omni.usd
from pxr import UsdGeom, Gf

stage = omni.usd.get_context().get_stage()

# ── Validate building Z bounds ────────────────────────────────────────────────
bbox_cache = UsdGeom.BBoxCache(0, [UsdGeom.Tokens.default_])
building = stage.GetPrimAtPath("/World/Building")
b = bbox_cache.ComputeWorldBound(building).GetRange()
floor_z = b.GetMin()[2]
roof_z  = b.GetMax()[2]
cx      = (b.GetMin()[0] + b.GetMax()[0]) / 2
cy      = (b.GetMin()[1] + b.GetMax()[1]) / 2
print(f"Building floor Z = {floor_z:.1f}")
print(f"Building roof  Z = {roof_z:.1f}")
print(f"Building center  = ({cx:.1f}, {cy:.1f})")

FPS          = 30
RISE_FRAMES  = 3600   # 120 seconds — very slow dramatic rise
TOTAL_FRAMES = 4200   # 140s total, 20s fully flooded at end

START_Z = floor_z - 5     # spawn just 5 units below floor — visible almost immediately
END_Z   = roof_z  + 200   # end well above roof

print(f"\nOcean: Z {START_Z:.0f} -> {END_Z:.0f} over {RISE_FRAMES} frames ({RISE_FRAMES/FPS:.0f}s)")

# ── Reposition MeshOut ────────────────────────────────────────────────────────
mesh_out = stage.GetPrimAtPath("/World/MeshOut")
if not mesh_out.IsValid():
    print("ERROR: /World/MeshOut not found — run merge_ocean_building.py first")
else:
    xf = UsdGeom.Xformable(mesh_out)
    xf.ClearXformOpOrder()
    t_op = xf.AddTranslateOp()
    xf.AddRotateXYZOp().Set(Gf.Vec3f(-90.0, 0.0, 0.0))
    xf.AddScaleOp().Set(Gf.Vec3f(0.15, 0.15, 0.15))

    print(f"Baking {TOTAL_FRAMES+1} keyframes...")
    for frame in range(TOTAL_FRAMES + 1):
        frac = min(frame / RISE_FRAMES, 1.0)
        wz = START_Z + frac * (END_Z - START_Z)
        t_op.Set(Gf.Vec3d(cx, cy, wz), frame)
        if frame % 600 == 0:
            print(f"  frame {frame}  Z={wz:.0f}")

    old = stage.GetPrimAtPath("/World/WaterPlane")
    if old.IsValid():
        stage.RemovePrim("/World/WaterPlane")

    print(f"Done — set timeline end to {TOTAL_FRAMES}, scrub to 0, press Play.")
