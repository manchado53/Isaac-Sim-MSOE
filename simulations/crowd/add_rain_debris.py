"""Remove debris/rain and add underwater blue tint volume.

Run from Isaac Sim Script Editor:
  exec(open('/home/ad.msoe.edu/somed/isaac-sim-msoe/simulations/crowd/add_rain_debris.py').read())
"""
from pxr import UsdGeom, UsdShade, Sdf, Gf, Vt
import omni.usd, math

stage = omni.usd.get_context().get_stage()

FPS          = 30
RISE_FRAMES  = 30 * 20
HOLD_FRAMES  = 30 * 10
TOTAL_FRAMES = RISE_FRAMES + HOLD_FRAMES
FLOOR_Z      = 26.9
FLOOD_RISE   = 1.5

# ── Remove debris and rain ────────────────────────────────────────────────────
for path in ["/World/Debris", "/World/Rain"]:
    p = stage.GetPrimAtPath(path)
    if p.IsValid():
        stage.RemovePrim(path)
        print(f"Removed {path}")

# ── Underwater volume ─────────────────────────────────────────────────────────
# Large box that fills the space between the floor and the water surface.
# When the camera dips below the water plane this box surrounds it
# and casts a blue atmospheric tint over everything.

vol_path = "/World/UnderwaterVolume"
old = stage.GetPrimAtPath(vol_path)
if old.IsValid():
    stage.RemovePrim(vol_path)

vol = UsdGeom.Cube.Define(stage, vol_path)
vol.CreateDoubleSidedAttr(True)

# Material: deep blue, very transparent, slight emissive glow
mat_path = "/World/Looks/UnderwaterMat"
old_mat  = stage.GetPrimAtPath(mat_path)
if old_mat.IsValid():
    stage.RemovePrim(mat_path)

mat = UsdShade.Material.Define(stage, mat_path)
sh  = UsdShade.Shader.Define(stage, mat_path + "/Shader")
sh.CreateIdAttr("UsdPreviewSurface")
sh.CreateInput("diffuseColor",  Sdf.ValueTypeNames.Color3f).Set(Gf.Vec3f(0.00, 0.10, 0.45))
sh.CreateInput("opacity",       Sdf.ValueTypeNames.Float).Set(0.18)   # very transparent
sh.CreateInput("roughness",     Sdf.ValueTypeNames.Float).Set(1.0)
sh.CreateInput("emissiveColor", Sdf.ValueTypeNames.Color3f).Set(Gf.Vec3f(0.00, 0.04, 0.20))
mat.CreateSurfaceOutput().ConnectToSource(sh.ConnectableAPI(), "surface")
UsdShade.MaterialBindingAPI.Apply(vol.GetPrim()).Bind(mat)

xf    = UsdGeom.Xformable(vol.GetPrim())
t_op  = xf.AddTranslateOp()
s_op  = xf.AddScaleOp()

# Unit cube is 2x2x2 by default — scale to cover the floor footprint
HALF_XY = 65.0   # covers the whole floor area

print(f"Baking underwater volume ({TOTAL_FRAMES+1} frames)...")
for frame in range(TOTAL_FRAMES + 1):
    frac       = min(frame / RISE_FRAMES, 1.0)
    flood_h    = frac * FLOOD_RISE          # current flood depth (0 → 1.5m)
    center_z   = FLOOR_Z + flood_h * 0.5   # vertical center of the volume
    half_h     = max(flood_h * 0.5, 0.01)  # half-height (avoid zero scale)

    t_op.Set(Gf.Vec3d(-20.0, 65.0, center_z), frame)
    s_op.Set(Gf.Vec3f(HALF_XY, HALF_XY, float(half_h)), frame)

print("Done — underwater blue tint volume added.")
print("Go below the water surface and press Play to see the effect.")
