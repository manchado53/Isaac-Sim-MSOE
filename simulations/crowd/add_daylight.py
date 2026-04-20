"""Add daylight — dome light + directional sun.

Run from Isaac Sim Script Editor:
  exec(open('/home/ad.msoe.edu/somed/isaac-sim-msoe/simulations/crowd/add_daylight.py').read())
"""
import omni.usd
from pxr import UsdLux, Gf

stage = omni.usd.get_context().get_stage()

# Remove old lights we added
for p in ["/World/FloodDomeLight", "/World/DomeLight", "/World/SunLight"]:
    prim = stage.GetPrimAtPath(p)
    if prim.IsValid():
        stage.RemovePrim(p)

# Dome light — bright daylight sky
dome = UsdLux.DomeLight.Define(stage, "/World/DomeLight")
dome.CreateIntensityAttr(2000.0)
dome.CreateColorAttr(Gf.Vec3f(0.85, 0.92, 1.0))  # soft blue sky

# Directional sun
sun = UsdLux.DistantLight.Define(stage, "/World/SunLight")
sun.CreateIntensityAttr(1500.0)
sun.CreateColorAttr(Gf.Vec3f(1.0, 0.95, 0.85))   # warm sunlight
sun.CreateAngleAttr(0.53)

from pxr import UsdGeom
xf = UsdGeom.Xformable(sun.GetPrim())
xf.AddRotateXYZOp().Set(Gf.Vec3f(-45.0, 30.0, 0.0))  # angle from upper-right

print("Daylight added — dome sky + sun.")
