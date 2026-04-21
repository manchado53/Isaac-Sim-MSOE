"""Spawn multiple Roscoe robot instances floating on the rising water surface.

Run AFTER merge_ocean_building.py.

exec(open('/home/ad.msoe.edu/somed/isaac-sim-msoe/simulations/crowd/float_robots.py').read())
"""
import omni.usd, math, random
from pxr import UsdGeom, Gf, Sdf, Usd

random.seed(7)
stage = omni.usd.get_context().get_stage()

ROBOT_USD    = "/home/ad.msoe.edu/somed/isaac-sim-msoe/assets/robots/roscoe_robot.usdz"

# Find actual root prim path inside USDZ (handles missing defaultPrim)
from pxr import Usd as _Usd
_rs = _Usd.Stage.Open(ROBOT_USD)
_root_prims = list(_rs.GetPseudoRoot().GetChildren())
ROBOT_PRIM_PATH = str(_root_prims[0].GetPath()) if _root_prims else ""
print(f"USDZ root prim: '{ROBOT_PRIM_PATH}'")
NUM_ROBOTS   = 6
FPS          = 30
RISE_FRAMES  = 4800
TOTAL_FRAMES = 5400

# Match merge_ocean_building.py water params
from pxr import UsdGeom as UG
bbox_cache = UG.BBoxCache(0, [UG.Tokens.default_])
building = stage.GetPrimAtPath("/World/Building")
b = bbox_cache.ComputeWorldBound(building).GetRange()
floor_z = b.GetMin()[2]
roof_z  = b.GetMax()[2]
cx      = (b.GetMin()[0] + b.GetMax()[0]) / 2
cy      = (b.GetMin()[1] + b.GetMax()[1]) / 2
bx      = (b.GetMax()[0] - b.GetMin()[0])
by      = (b.GetMax()[1] - b.GetMin()[1])

START_Z = floor_z - 5
END_Z   = roof_z + 200

def water_z(frame):
    return START_Z + min(frame / RISE_FRAMES, 1.0) * (END_Z - START_Z)

# Atrium/lobby area — world meters (local cm × 0.01 from building scale)
ATRIUM_X = -27.49
ATRIUM_Y =  68.39
SPREAD   =  4.0     # scatter radius in world meters

positions = []
for i in range(NUM_ROBOTS):
    rx = ATRIUM_X + random.uniform(-SPREAD, SPREAD)
    ry = ATRIUM_Y + random.uniform(-SPREAD, SPREAD)
    positions.append((rx, ry))

print(f"Spawning {NUM_ROBOTS} robots in atrium at ({ATRIUM_X}, {ATRIUM_Y}) spread={SPREAD}")

robots_root = stage.GetPrimAtPath("/World/Robots")
if not robots_root.IsValid():
    stage.DefinePrim("/World/Robots", "Xform")

for i, (rx, ry) in enumerate(positions):
    path = f"/World/Robots/Roscoe_{i}"
    prim = stage.DefinePrim(path, "Xform")
    prim.GetReferences().AddReference(ROBOT_USD, ROBOT_PRIM_PATH)

    xf    = UsdGeom.Xformable(prim)
    xf.ClearXformOpOrder()
    t_op   = xf.AddTranslateOp()
    rot_op = xf.AddRotateXYZOp()
    xf.AddScaleOp().Set(Gf.Vec3f(0.5, 0.5, 0.5))

    bob_amp  = random.uniform(0.3, 0.8)
    bob_freq = random.uniform(0.4, 0.9)
    spin     = random.uniform(-0.05, 0.05)
    yaw      = random.uniform(0, 360)

    for frame in range(TOTAL_FRAMES + 1):
        wz = water_z(frame)
        t  = frame / FPS
        bob = bob_amp * math.sin(bob_freq * t * 2 * math.pi)
        fz  = wz + 0.5 + bob  # sit just above surface
        t_op.Set(Gf.Vec3d(rx, ry, fz), frame)
        rot_op.Set(Gf.Vec3f(0.0, 0.0, yaw + spin * frame), frame)

    print(f"  Robot {i} placed at ({rx:.0f}, {ry:.0f}), baked {TOTAL_FRAMES+1} frames")

print(f"\nDone — {NUM_ROBOTS} robots floating. Set timeline end to {TOTAL_FRAMES}, press Play.")
