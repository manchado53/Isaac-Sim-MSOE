"""Merge Warp ocean.usda into building stage as a sublayer.

Rotates MeshOut (rendered output) -90 on X to lay flat in Z-up building space.
Animates MeshOut translate Y (= world Z after rotation) to rise from floor to roof.

IMPORTANT: Load the building USD first, then run this script.

exec(open('/home/ad.msoe.edu/somed/isaac-sim-msoe/simulations/crowd/merge_ocean_building.py').read())
"""
import omni.usd
from pxr import UsdGeom, Gf, Sdf

OCEAN_USD = "/home/ad.msoe.edu/somed/isaacsim/extscache/omni.warp-1.8.2/data/scenes/ocean.usda"

FPS          = 30
RISE_FRAMES  = 600
TOTAL_FRAMES = 900
# Building Z: -8193 to 8192 (in stage units)
START_Y      = -8500.0
FLOOD_RISE   = 17500.0   # ends at Y=9000 > roof at 8192

stage = omni.usd.get_context().get_stage()
root_layer = stage.GetRootLayer()

# Add ocean as sublayer if not already present
if OCEAN_USD not in root_layer.subLayerPaths:
    root_layer.subLayerPaths.insert(0, OCEAN_USD)
    print("Ocean sublayer added.")
else:
    print("Ocean sublayer already present.")

# Rotate and animate MeshOut (the rendered result Warp writes to)
mesh_out_prim = stage.GetPrimAtPath("/World/MeshOut")
if not mesh_out_prim.IsValid():
    print("ERROR: /World/MeshOut not found — did the sublayer load?")
else:
    xf = UsdGeom.Xformable(mesh_out_prim)
    xf.ClearXformOpOrder()
    t_op = xf.AddTranslateOp()
    xf.AddRotateXYZOp().Set(Gf.Vec3f(-90.0, 0.0, 0.0))  # lay flat in Z-up space
    xf.AddScaleOp().Set(Gf.Vec3f(0.4, 0.4, 1.0))

    print(f"Baking {TOTAL_FRAMES+1} rising keyframes...")
    for frame in range(TOTAL_FRAMES + 1):
        frac = min(frame / RISE_FRAMES, 1.0)
        y = START_Y + frac * FLOOD_RISE
        t_op.Set(Gf.Vec3d(-60.0, y, 3479.0), frame)
        if frame % 150 == 0:
            print(f"  {frame}/{TOTAL_FRAMES}")

    print("Done — press Play.")
