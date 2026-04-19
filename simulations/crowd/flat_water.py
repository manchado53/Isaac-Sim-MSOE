"""Flat rising water — no vertex displacement, realistic material.

Run from Isaac Sim Script Editor:
  exec(open('/home/ad.msoe.edu/somed/isaac-sim-msoe/simulations/crowd/flat_water.py').read())
"""
import omni.usd
from pxr import UsdGeom, UsdShade, Sdf, Gf, Vt

stage = omni.usd.get_context().get_stage()

FPS=30; RISE_FRAMES=600; HOLD_FRAMES=300; TOTAL_FRAMES=900
FLOOR_Z=26.9; FLOOD_RISE=1.5

# ── Recreate water mesh as simple flat quad ───────────────────────────────────
old = stage.GetPrimAtPath("/World/WaterPlane")
if old.IsValid():
    stage.RemovePrim("/World/WaterPlane")

half = 60.0
mesh = UsdGeom.Mesh.Define(stage, "/World/WaterPlane")
mesh.GetPointsAttr().Set(Vt.Vec3fArray([
    Gf.Vec3f(-half, -half, 0),
    Gf.Vec3f( half, -half, 0),
    Gf.Vec3f( half,  half, 0),
    Gf.Vec3f(-half,  half, 0),
]))
mesh.GetFaceVertexCountsAttr().Set(Vt.IntArray([4]))
mesh.GetFaceVertexIndicesAttr().Set(Vt.IntArray([0, 1, 2, 3]))
mesh.GetSubdivisionSchemeAttr().Set("none")
mesh.CreateDoubleSidedAttr(True)

# Animate rising translate only
xf   = UsdGeom.Xformable(mesh.GetPrim())
t_op = xf.AddTranslateOp()

print(f"Baking {TOTAL_FRAMES+1} frames of rising water...")
for frame in range(TOTAL_FRAMES + 1):
    frac = min(frame / RISE_FRAMES, 1.0)
    t_op.Set(Gf.Vec3d(-20.0, 65.0, FLOOR_Z + frac * FLOOD_RISE), frame)
print("Rise baked")

# ── Realistic water material ──────────────────────────────────────────────────
old_mat = stage.GetPrimAtPath("/World/Looks/WaterMaterial")
if old_mat.IsValid():
    stage.RemovePrim("/World/Looks/WaterMaterial")

mat = UsdShade.Material.Define(stage, "/World/Looks/WaterMaterial")
sh  = UsdShade.Shader.Define(stage, "/World/Looks/WaterMaterial/Shader")
sh.CreateIdAttr("UsdPreviewSurface")
sh.CreateInput("diffuseColor",  Sdf.ValueTypeNames.Color3f).Set(Gf.Vec3f(0.01, 0.12, 0.45))
sh.CreateInput("opacity",       Sdf.ValueTypeNames.Float).Set(0.80)
sh.CreateInput("roughness",     Sdf.ValueTypeNames.Float).Set(0.08)   # glossy like still water
sh.CreateInput("metallic",      Sdf.ValueTypeNames.Float).Set(0.0)
sh.CreateInput("ior",           Sdf.ValueTypeNames.Float).Set(1.33)
sh.CreateInput("specularColor", Sdf.ValueTypeNames.Color3f).Set(Gf.Vec3f(0.5, 0.6, 0.8))
mat.CreateSurfaceOutput().ConnectToSource(sh.ConnectableAPI(), "surface")
UsdShade.MaterialBindingAPI.Apply(mesh.GetPrim()).Bind(mat)

print("Done — flat glossy water ready. Press Play to preview.")
