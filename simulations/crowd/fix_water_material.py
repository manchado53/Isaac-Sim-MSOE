"""Recreate water material — no emissive, no clearcoat, no flicker.
Keeps full 0.35m wave displacement.

Run from Isaac Sim Script Editor:
  exec(open('/home/ad.msoe.edu/somed/isaac-sim-msoe/simulations/crowd/fix_water_material.py').read())
"""
import omni.usd
from pxr import UsdShade, UsdGeom, Sdf, Gf

stage = omni.usd.get_context().get_stage()

old = stage.GetPrimAtPath("/World/Looks/WaterMaterial")
if old.IsValid():
    stage.RemovePrim("/World/Looks/WaterMaterial")

mat = UsdShade.Material.Define(stage, "/World/Looks/WaterMaterial")
sh  = UsdShade.Shader.Define(stage, "/World/Looks/WaterMaterial/Shader")
sh.CreateIdAttr("UsdPreviewSurface")
sh.CreateInput("diffuseColor",  Sdf.ValueTypeNames.Color3f).Set(Gf.Vec3f(0.02, 0.18, 0.55))
sh.CreateInput("opacity",       Sdf.ValueTypeNames.Float).Set(0.75)
sh.CreateInput("roughness",     Sdf.ValueTypeNames.Float).Set(0.5)
sh.CreateInput("metallic",      Sdf.ValueTypeNames.Float).Set(0.0)
sh.CreateInput("specularColor", Sdf.ValueTypeNames.Color3f).Set(Gf.Vec3f(0.1, 0.1, 0.1))
mat.CreateSurfaceOutput().ConnectToSource(sh.ConnectableAPI(), "surface")

mesh = UsdGeom.Mesh(stage.GetPrimAtPath("/World/WaterPlane"))
UsdShade.MaterialBindingAPI.Apply(mesh.GetPrim()).Bind(mat)
print("Water material fixed — no flicker, full wave displacement retained.")
