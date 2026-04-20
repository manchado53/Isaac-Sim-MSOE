"""Reposition/rescale MeshOut without rebaking. Run after merge_ocean_building.py.

exec(open('/home/ad.msoe.edu/somed/isaac-sim-msoe/simulations/crowd/reposition_ocean.py').read())
"""
import omni.usd
from pxr import UsdGeom, Gf

stage = omni.usd.get_context().get_stage()

FPS          = 30
RISE_FRAMES  = 600
TOTAL_FRAMES = 900

# Building Z range: -8193 to 8192
# Start fully below building floor, rise to above roof
WORLD_Z_START = -9000.0
WORLD_Z_END   =  9500.0

WORLD_CX = -60.0
WORLD_CY = 3479.0

mesh_out = stage.GetPrimAtPath("/World/MeshOut")
xf = UsdGeom.Xformable(mesh_out)
xf.ClearXformOpOrder()
t_op = xf.AddTranslateOp()
xf.AddRotateXYZOp().Set(Gf.Vec3f(-90.0, 0.0, 0.0))
xf.AddScaleOp().Set(Gf.Vec3f(0.15, 0.15, 0.15))  # smaller — building fills more of frame

print(f"Updating {TOTAL_FRAMES+1} keyframes...")
for frame in range(TOTAL_FRAMES + 1):
    frac = min(frame / RISE_FRAMES, 1.0)
    wz = WORLD_Z_START + frac * (WORLD_Z_END - WORLD_Z_START)
    t_op.Set(Gf.Vec3d(WORLD_CX, -wz, WORLD_CY), frame)
    if frame % 150 == 0:
        print(f"  {frame}/{TOTAL_FRAMES}  world_z={wz:.0f}")

# Also remove the pre-baked WaterPlane if it exists (conflicts)
old = stage.GetPrimAtPath("/World/WaterPlane")
if old.IsValid():
    stage.RemovePrim("/World/WaterPlane")
    print("Removed old WaterPlane.")

print("Done — scrub to frame 0 and press Play.")
