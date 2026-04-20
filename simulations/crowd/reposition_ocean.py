"""Reposition/rescale MeshOut without rebaking. Run after merge_ocean_building.py.

exec(open('/home/ad.msoe.edu/somed/isaac-sim-msoe/simulations/crowd/reposition_ocean.py').read())
"""
import omni.usd
from pxr import UsdGeom, Gf

stage = omni.usd.get_context().get_stage()

FPS          = 30
RISE_FRAMES  = 1200   # 40s rise — slow and dramatic
TOTAL_FRAMES = 1500   # 50s total, 10s fully flooded at end

# Building Z: -8193 (floor) to 8192 (roof) — start just below, end just above
WORLD_Z_START = -8300.0
WORLD_Z_END   =  8400.0

WORLD_CX = -60.0
WORLD_CY = 3479.0

mesh_out = stage.GetPrimAtPath("/World/MeshOut")
if not mesh_out.IsValid():
    print("ERROR: /World/MeshOut not found — run merge_ocean_building.py first")
else:
    xf = UsdGeom.Xformable(mesh_out)
    xf.ClearXformOpOrder()
    t_op = xf.AddTranslateOp()
    xf.AddRotateXYZOp().Set(Gf.Vec3f(-90.0, 0.0, 0.0))
    xf.AddScaleOp().Set(Gf.Vec3f(0.15, 0.15, 0.15))

    print(f"Updating {TOTAL_FRAMES+1} keyframes...")
    for frame in range(TOTAL_FRAMES + 1):
        frac = min(frame / RISE_FRAMES, 1.0)
        wz = WORLD_Z_START + frac * (WORLD_Z_END - WORLD_Z_START)
        t_op.Set(Gf.Vec3d(WORLD_CX, WORLD_CY, wz), frame)
        if frame % 300 == 0:
            print(f"  {frame}/{TOTAL_FRAMES}  world_z={wz:.0f}")

    old = stage.GetPrimAtPath("/World/WaterPlane")
    if old.IsValid():
        stage.RemovePrim("/World/WaterPlane")
        print("Removed old WaterPlane.")

    print("Done — set timeline end to 1500, scrub to frame 0, press Play.")
