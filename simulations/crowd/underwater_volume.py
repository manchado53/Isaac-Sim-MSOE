"""Add a rising translucent blue underwater volume.

When the camera is below the water surface it will be inside this mesh,
giving a blue tinted underwater look. Run AFTER merge_ocean_building.py.

exec(open('/home/ad.msoe.edu/somed/isaac-sim-msoe/simulations/crowd/underwater_volume.py').read())
"""
import omni.usd
from pxr import UsdGeom, UsdShade, Gf, Sdf

stage = omni.usd.get_context().get_stage()

FPS          = 30
RISE_FRAMES  = 4800
TOTAL_FRAMES = 5400

# Match water params from merge_ocean_building.py
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
DEPTH   = 600.0   # how deep the volume extends below surface

# Volume is twice the building footprint so it covers camera at any angle
VOL_W = max(bx, by) * 3.0

# ── Create underwater volume mesh ─────────────────────────────────────────────
vol_path = "/World/UnderwaterVolume"
old = stage.GetPrimAtPath(vol_path)
if old.IsValid():
    stage.RemovePrim(vol_path)

cube = UsdGeom.Cube.Define(stage, vol_path)
cube.GetSizeAttr().Set(1.0)  # unit cube, scaled via xform

xf    = UsdGeom.Xformable(cube.GetPrim())
xf.ClearXformOpOrder()
t_op  = xf.AddTranslateOp()
xf.AddScaleOp().Set(Gf.Vec3f(VOL_W, VOL_W, DEPTH))

# ── Create translucent blue water material ────────────────────────────────────
mat_path = "/World/Looks/UnderwaterMat"
mat = UsdShade.Material.Define(stage, mat_path)
shader = UsdShade.Shader.Define(stage, mat_path + "/Shader")
shader.CreateIdAttr("UsdPreviewSurface")

shader.CreateInput("diffuseColor",  Sdf.ValueTypeNames.Color3f).Set(Gf.Vec3f(0.02, 0.15, 0.35))
shader.CreateInput("opacity",       Sdf.ValueTypeNames.Float).Set(0.55)
shader.CreateInput("opacityThreshold", Sdf.ValueTypeNames.Float).Set(0.0)
shader.CreateInput("roughness",     Sdf.ValueTypeNames.Float).Set(0.05)
shader.CreateInput("metallic",      Sdf.ValueTypeNames.Float).Set(0.0)
shader.CreateInput("ior",           Sdf.ValueTypeNames.Float).Set(1.33)

mat.CreateSurfaceOutput().ConnectToSource(shader.ConnectableAPI(), "surface")
UsdShade.MaterialBindingAPI(cube.GetPrim()).Bind(mat)

# Disable backface culling so volume is visible from inside
cube.GetPrim().SetInstanceable(False)

# ── Bake keyframes — volume center tracks water_z - DEPTH/2 ──────────────────
print(f"Baking {TOTAL_FRAMES+1} underwater volume keyframes...")
for frame in range(TOTAL_FRAMES + 1):
    frac = min(frame / RISE_FRAMES, 1.0)
    wz   = START_Z + frac * (END_Z - START_Z)
    # Center the volume so top face = water surface, extends DEPTH downward
    vol_z = wz - DEPTH / 2.0
    t_op.Set(Gf.Vec3d(cx, cy, vol_z), frame)
    if frame % 1200 == 0:
        print(f"  frame {frame}  water_z={wz:.0f}  vol_center={vol_z:.0f}")

print("Done — underwater volume added. Press Play and submerge the camera.")
