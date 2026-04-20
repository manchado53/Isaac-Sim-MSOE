"""Merge Warp ocean.usda into building stage as a sublayer.

Scales ocean plane to cover building footprint and animates Z rising
from ground floor to above roof.

IMPORTANT: Load the building USD first, then run this script.

exec(open('/home/ad.msoe.edu/somed/isaac-sim-msoe/simulations/crowd/merge_ocean_building.py').read())
"""
import omni.usd
from pxr import UsdGeom, Gf, Sdf

OCEAN_USD = "/home/ad.msoe.edu/somed/isaacsim/extscache/omni.warp-1.8.2/data/scenes/ocean.usda"

FPS          = 30
RISE_FRAMES  = 600
TOTAL_FRAMES = 900
START_Z      = -8500.0   # below building floor
FLOOD_RISE   = 17500.0   # rises to Z=9000, above roof at 8192

stage = omni.usd.get_context().get_stage()
root_layer = stage.GetRootLayer()

# Add ocean as sublayer if not already present
if OCEAN_USD not in root_layer.subLayerPaths:
    root_layer.subLayerPaths.insert(0, OCEAN_USD)
    print("Ocean sublayer added.")
else:
    print("Ocean sublayer already present.")

# Override MeshIn xform — scale to cover full building, animate Z
mesh_in_prim = stage.GetPrimAtPath("/World/MeshIn")
if not mesh_in_prim.IsValid():
    print("ERROR: /World/MeshIn not found — did the sublayer load?")
else:
    xf = UsdGeom.Xformable(mesh_in_prim)
    xf.ClearXformOpOrder()
    t_op = xf.AddTranslateOp()
    rot_op = xf.AddRotateXYZOp()
    rot_op.Set(Gf.Vec3f(-90.0, 0.0, 0.0))   # Y-up -> Z-up: lay ocean flat
    scale_op = xf.AddScaleOp()
    scale_op.Set(Gf.Vec3f(0.4, 0.4, 1.0))

    print(f"Baking {TOTAL_FRAMES+1} rising Z keyframes...")
    for frame in range(TOTAL_FRAMES + 1):
        frac = min(frame / RISE_FRAMES, 1.0)
        z = START_Z + frac * FLOOD_RISE
        t_op.Set(Gf.Vec3d(-60.0, 3479.0, z), frame)
        if frame % 150 == 0:
            print(f"  {frame}/{TOTAL_FRAMES}  Z={z:.2f}")

    print("Done — press Play to see Warp ocean rising through building.")
    print("If waves don't appear, enable the omni.warp extension:")
    print("  Window -> Extensions -> search 'warp' -> enable omni.warp")
