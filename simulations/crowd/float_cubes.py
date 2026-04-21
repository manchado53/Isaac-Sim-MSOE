"""Animate yellow_block prims to float and rise with the water.

exec(open('/home/ad.msoe.edu/somed/isaac-sim-msoe/simulations/crowd/float_cubes.py').read())
"""
import omni.usd, math, random
from pxr import UsdGeom, Gf

random.seed(42)
stage = omni.usd.get_context().get_stage()

FPS          = 30
RISE_FRAMES  = 4800
TOTAL_FRAMES = 5400

# Match merge_ocean_building.py water params
bbox_cache = UsdGeom.BBoxCache(0, [UsdGeom.Tokens.default_])
building = stage.GetPrimAtPath("/World/Building")
b = bbox_cache.ComputeWorldBound(building).GetRange()
floor_z = b.GetMin()[2]
roof_z  = b.GetMax()[2]
START_Z = floor_z - 5
END_Z   = roof_z + 200

def water_z(frame):
    return START_Z + min(frame / RISE_FRAMES, 1.0) * (END_Z - START_Z)

# Find all yellow_block prims
floaters = []
for prim in stage.Traverse():
    name = prim.GetName()
    if "yellow_block" in name and prim.IsA(UsdGeom.Xformable):
        if str(prim.GetPath()).count("/") <= 2:
            floaters.append(prim)

print(f"Found {len(floaters)} yellow blocks: {[str(p.GetPath()) for p in floaters]}")

for prim in floaters:
    b2 = bbox_cache.ComputeWorldBound(prim).GetRange()
    orig_x = (b2.GetMin()[0] + b2.GetMax()[0]) / 2
    orig_y = (b2.GetMin()[1] + b2.GetMax()[1]) / 2
    orig_z = b2.GetMin()[2]

    bob_amp  = random.uniform(0.05, 0.15)
    bob_freq = random.uniform(0.5, 1.2)
    spin     = random.uniform(-0.2, 0.2)

    xf = UsdGeom.Xformable(prim)
    xf.ClearXformOpOrder()
    t_op   = xf.AddTranslateOp()
    rot_op = xf.AddRotateXYZOp()
    xf.AddScaleOp().Set(Gf.Vec3f(1.0, 1.0, 1.0))

    for frame in range(TOTAL_FRAMES + 1):
        wz = water_z(frame)
        if wz >= orig_z:
            t = frame / FPS
            bob = bob_amp * math.sin(bob_freq * t * 2 * math.pi)
            fz  = wz + bob
            rot = spin * (frame - RISE_FRAMES * max(0, orig_z - START_Z) / (END_Z - START_Z))
        else:
            fz  = orig_z
            rot = 0.0
        t_op.Set(Gf.Vec3d(orig_x, orig_y, fz), frame)
        rot_op.Set(Gf.Vec3f(0.0, 0.0, rot), frame)

    print(f"  {prim.GetPath()} baked at world ({orig_x:.2f}, {orig_y:.2f}, {orig_z:.2f})")

print(f"Done — set timeline to {TOTAL_FRAMES}, press Play.")
