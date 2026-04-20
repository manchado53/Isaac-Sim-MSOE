"""Boost wave amplitude by scaling MeshOut non-uniformly — tall in Z, wide enough to cover building.

The Warp ocean deforms a mesh in local Y (which maps to world Z after -90 X rotation).
Increasing scaleY amplifies wave height. Increasing scaleX/scaleZ widens the surface.

Run AFTER merge_ocean_building.py (ocean sublayer must already be loaded).
Then run reposition_ocean.py to re-bake the Z-rise keyframes.

exec(open('/home/ad.msoe.edu/somed/isaac-sim-msoe/simulations/crowd/violent_waves.py').read())
"""
import omni.usd
from pxr import UsdGeom, Gf

stage = omni.usd.get_context().get_stage()

mesh_out = stage.GetPrimAtPath("/World/MeshOut")
if not mesh_out.IsValid():
    print("ERROR: /World/MeshOut not found — run merge_ocean_building.py first")
else:
    xf = UsdGeom.Xformable(mesh_out)
    ops = xf.GetOrderedXformOps()
    for op in ops:
        if "scale" in op.GetOpName():
            current = op.Get()
            print(f"Current scale: {current}")

    # Scale: X=width, Y=wave height amplifier (local Y = world Z after rotation), Z=depth
    # 0.15 base scale; boost Y (wave amplitude) to 0.45 = 3x taller waves
    # Keep XZ wide enough: 0.30 covers the building footprint
    SCALE_XZ = 0.30   # wider surface coverage
    SCALE_Y  = 0.60   # 4x taller waves (local Y maps to world Z = wave height)

    xf.ClearXformOpOrder()
    t_op = xf.AddTranslateOp()
    xf.AddRotateXYZOp().Set(Gf.Vec3f(-90.0, 0.0, 0.0))
    xf.AddScaleOp().Set(Gf.Vec3f(SCALE_XZ, SCALE_Y, SCALE_XZ))

    # Re-bake position keyframes (same as reposition_ocean.py but combined)
    from pxr import UsdGeom as UG

    bbox_cache = UG.BBoxCache(0, [UG.Tokens.default_])
    building = stage.GetPrimAtPath("/World/Building")
    b = bbox_cache.ComputeWorldBound(building).GetRange()
    floor_z = b.GetMin()[2]
    roof_z  = b.GetMax()[2]
    cx      = (b.GetMin()[0] + b.GetMax()[0]) / 2
    cy      = (b.GetMin()[1] + b.GetMax()[1]) / 2

    FPS          = 30
    RISE_FRAMES  = 3600
    TOTAL_FRAMES = 4200
    START_Z = floor_z - 5
    END_Z   = roof_z  + 200

    print(f"Building Z: {floor_z:.1f} to {roof_z:.1f}  center=({cx:.1f},{cy:.1f})")
    print(f"Ocean Z: {START_Z:.0f} -> {END_Z:.0f} over {RISE_FRAMES} frames ({RISE_FRAMES/FPS:.0f}s)")
    print(f"Scale: XZ={SCALE_XZ}, Y(wave amp)={SCALE_Y}")
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

    print(f"\nDone — violent waves set. Timeline end={TOTAL_FRAMES}, press Play.")
