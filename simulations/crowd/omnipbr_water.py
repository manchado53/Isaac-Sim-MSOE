"""Apply OmniPBR water material — looks like real water in RTX Interactive.

Run from Isaac Sim Script Editor:
  exec(open('/home/ad.msoe.edu/somed/isaac-sim-msoe/simulations/crowd/omnipbr_water.py').read())
"""
import omni.usd
from pxr import UsdShade, UsdGeom, Sdf, Gf

stage = omni.usd.get_context().get_stage()

# Remove old material
old = stage.GetPrimAtPath("/World/Looks/WaterMaterial")
if old.IsValid():
    stage.RemovePrim("/World/Looks/WaterMaterial")

mat = UsdShade.Material.Define(stage, "/World/Looks/WaterMaterial")
sh  = UsdShade.Shader.Define(stage, "/World/Looks/WaterMaterial/Shader")

# Use OmniPBR MDL — required for realistic water in RTX Interactive
sh.SetSourceAsset(Sdf.AssetPath("OmniPBR.mdl"), "mdl")
sh.SetSourceAssetSubIdentifier("OmniPBR", "mdl")
sh.CreateIdAttr("OmniPBR")

# Water parameters
sh.CreateInput("diffuse_color_constant",          Sdf.ValueTypeNames.Color3f).Set(Gf.Vec3f(0.01, 0.12, 0.40))
sh.CreateInput("reflection_roughness_constant",   Sdf.ValueTypeNames.Float).Set(0.04)   # near mirror
sh.CreateInput("metallic_constant",               Sdf.ValueTypeNames.Float).Set(0.0)
sh.CreateInput("opacity_constant",                Sdf.ValueTypeNames.Float).Set(0.82)
sh.CreateInput("enable_opacity",                  Sdf.ValueTypeNames.Bool).Set(True)
sh.CreateInput("ior_constant",                    Sdf.ValueTypeNames.Float).Set(1.33)
sh.CreateInput("specular_level",                  Sdf.ValueTypeNames.Float).Set(0.6)

mat.CreateSurfaceOutput("mdl").ConnectToSource(sh.ConnectableAPI(), "out")
mat.CreateDisplacementOutput("mdl").ConnectToSource(sh.ConnectableAPI(), "out")
mat.CreateVolumeOutput("mdl").ConnectToSource(sh.ConnectableAPI(), "out")

mesh = UsdGeom.Mesh(stage.GetPrimAtPath("/World/WaterPlane"))
UsdShade.MaterialBindingAPI.Apply(mesh.GetPrim()).Bind(mat)

print("OmniPBR water material applied.")
print("Switch to RTX Interactive for best results.")
