"""Merge Warp ocean.usda into building stage as a sublayer.

Computes building bounds dynamically so ocean spawns just below floor immediately.

exec(open('/home/ad.msoe.edu/somed/isaac-sim-msoe/simulations/crowd/merge_ocean_building.py').read())
"""
import omni.usd
from pxr import UsdGeom, Gf

OCEAN_USD = "/home/ad.msoe.edu/somed/isaacsim/extscache/omni.warp-1.8.2/data/scenes/ocean.usda"

FPS          = 30
RISE_FRAMES  = 4800   # 160 seconds — very slow dramatic rise
TOTAL_FRAMES = 5400   # 180s total, 20s fully flooded at end

stage = omni.usd.get_context().get_stage()
root_layer = stage.GetRootLayer()

if OCEAN_USD not in root_layer.subLayerPaths:
    root_layer.subLayerPaths.insert(0, OCEAN_USD)
    print("Ocean sublayer added.")
else:
    print("Ocean sublayer already present.")

# Compute building bounds dynamically
bbox_cache = UsdGeom.BBoxCache(0, [UsdGeom.Tokens.default_])
building = stage.GetPrimAtPath("/World/Building")
b = bbox_cache.ComputeWorldBound(building).GetRange()
floor_z = b.GetMin()[2]
roof_z  = b.GetMax()[2]
cx      = (b.GetMin()[0] + b.GetMax()[0]) / 2
cy      = (b.GetMin()[1] + b.GetMax()[1]) / 2

START_Z = floor_z - 5    # spawn just below floor — visible immediately
END_Z   = roof_z  + 200

print(f"Building Z: {floor_z:.1f} to {roof_z:.1f}  center=({cx:.1f},{cy:.1f})")
print(f"Ocean Z: {START_Z:.0f} -> {END_Z:.0f} over {RISE_FRAMES} frames ({RISE_FRAMES/FPS:.0f}s)")

mesh_out = stage.GetPrimAtPath("/World/MeshOut")
if not mesh_out.IsValid():
    print("ERROR: /World/MeshOut not found")
else:
    xf = UsdGeom.Xformable(mesh_out)
    xf.ClearXformOpOrder()
    t_op = xf.AddTranslateOp()
    xf.AddRotateXYZOp().Set(Gf.Vec3f(-90.0, 0.0, 0.0))
    xf.AddScaleOp().Set(Gf.Vec3f(0.08, 0.08, 0.08))  # smaller scale — proportional to building

    print(f"Baking {TOTAL_FRAMES+1} keyframes...")
    for frame in range(TOTAL_FRAMES + 1):
        frac = min(frame / RISE_FRAMES, 1.0)
        wz = START_Z + frac * (END_Z - START_Z)
        t_op.Set(Gf.Vec3d(cx, cy, wz), frame)
        if frame % 1200 == 0:
            print(f"  {frame}/{TOTAL_FRAMES}  world_z={wz:.0f}")

    old = stage.GetPrimAtPath("/World/WaterPlane")
    if old.IsValid():
        stage.RemovePrim("/World/WaterPlane")

    print(f"Done — set timeline end to {TOTAL_FRAMES}, scrub to 0, press Play.")
