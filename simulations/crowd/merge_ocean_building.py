"""Merge Warp ocean.usda into building stage as a sublayer.

After rotating MeshOut -90 on X:
  local_x = world_x
  local_y = -world_z   (so to rise in world Z, decrease local Y)
  local_z =  world_y

Translates use local-space coords to position correctly in Z-up building.

IMPORTANT: Load the building USD first, then run this script.

exec(open('/home/ad.msoe.edu/somed/isaac-sim-msoe/simulations/crowd/merge_ocean_building.py').read())
"""
import omni.usd
from pxr import UsdGeom, Gf

OCEAN_USD = "/home/ad.msoe.edu/somed/isaacsim/extscache/omni.warp-1.8.2/data/scenes/ocean.usda"

FPS          = 30
RISE_FRAMES  = 600
TOTAL_FRAMES = 900

# Building world Z: -8193 to 8192 — water rises through this range
WORLD_Z_START = -8500.0
WORLD_Z_END   =  9000.0

# Building world XY center (from check_bounds): X=-60, Y=3479
WORLD_CX = -60.0
WORLD_CY = 3479.0

stage = omni.usd.get_context().get_stage()
root_layer = stage.GetRootLayer()

if OCEAN_USD not in root_layer.subLayerPaths:
    root_layer.subLayerPaths.insert(0, OCEAN_USD)
    print("Ocean sublayer added.")
else:
    print("Ocean sublayer already present.")

mesh_out = stage.GetPrimAtPath("/World/MeshOut")
if not mesh_out.IsValid():
    print("ERROR: /World/MeshOut not found")
else:
    xf = UsdGeom.Xformable(mesh_out)
    xf.ClearXformOpOrder()
    t_op = xf.AddTranslateOp()
    xf.AddRotateXYZOp().Set(Gf.Vec3f(-90.0, 0.0, 0.0))
    xf.AddScaleOp().Set(Gf.Vec3f(0.4, 0.4, 0.4))

    print(f"Baking {TOTAL_FRAMES+1} keyframes...")
    for frame in range(TOTAL_FRAMES + 1):
        frac = min(frame / RISE_FRAMES, 1.0)
        wz = WORLD_Z_START + frac * (WORLD_Z_END - WORLD_Z_START)
        t_op.Set(Gf.Vec3d(WORLD_CX, WORLD_CY, wz), frame)
        if frame % 150 == 0:
            print(f"  {frame}/{TOTAL_FRAMES}  world_z={wz:.0f}")

    print("Done — press Play.")
