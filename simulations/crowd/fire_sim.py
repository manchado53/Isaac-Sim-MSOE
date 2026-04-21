"""Spawn NVIDIA Flow fire emitters in the building corridor/rooms.

Run with the building USD loaded. Places fire at specified world positions.

exec(open('/home/ad.msoe.edu/somed/isaac-sim-msoe/simulations/crowd/fire_sim.py').read())
"""
import omni.usd
from pxr import UsdGeom, Gf, Sdf, UsdShade

stage = omni.usd.get_context().get_stage()

# Fire locations — corridor + a couple rooms (world meters)
# Fire positions on building facade — multiple floors, window locations
# Format: (x, y, z) in world meters
FLOOR_Z = [22.34, 26.0, 29.5, 33.0]  # ground + 3 upper floors
WINDOW_X = [-35.0, -28.0, -20.0, -12.0, -5.0]
WINDOW_Y = 75.0  # building facade Y

FIRE_POSITIONS = []
for z in FLOOR_Z:
    for x in WINDOW_X:
        FIRE_POSITIONS.append((x, WINDOW_Y, z))

FIRE_SCALE = 0.4  # smaller, more contained per window

# Remove old fire prims
old = stage.GetPrimAtPath("/World/FireSim")
if old.IsValid():
    stage.RemovePrim("/World/FireSim")

root = stage.DefinePrim("/World/FireSim", "Xform")

for i, (fx, fy, fz) in enumerate(FIRE_POSITIONS):
    fire_path = f"/World/FireSim/Fire_{i}"

    # Flow emitter prim
    fire_prim = stage.DefinePrim(fire_path, "Xform")
    xf = UsdGeom.Xformable(fire_prim)
    xf.AddTranslateOp().Set(Gf.Vec3d(fx, fy, fz))
    xf.AddRotateXYZOp().Set(Gf.Vec3f(90.0, 0.0, 0.0))  # make fire vertical Z-up
    xf.AddScaleOp().Set(Gf.Vec3f(FIRE_SCALE, FIRE_SCALE, FIRE_SCALE))

    FIRE_USD = "/home/ad.msoe.edu/somed/isaacsim/extscache/omni.flowusd-107.1.8+107.1.0.lx64.r.cp311.u353/data/presets/Fire/Fire.usda"
    fire_prim.GetReferences().AddReference(FIRE_USD)

    # Override gravity to Z-up so fire rises vertically
    adv_path = f"{fire_path}/flowSimulate/advection"
    adv_prim = stage.OverridePrim(adv_path)
    adv_prim.CreateAttribute("gravity", Sdf.ValueTypeNames.Float3).Set(Gf.Vec3f(0, 0, -100))

    print(f"  Fire_{i} placed at ({fx}, {fy}, {fz})")

# Dim dome light to make building dark
from pxr import UsdLux
for prim in stage.Traverse():
    if prim.IsA(UsdLux.DomeLight) or prim.IsA(UsdLux.DistantLight):
        UsdLux.LightAPI(prim).GetIntensityAttr().Set(50.0)
        print(f"  Dimmed {prim.GetPath()}")

# Add orange point lights at each fire for glow
for i, (fx, fy, fz) in enumerate(FIRE_POSITIONS):
    light_path = f"/World/FireSim/FireLight_{i}"
    light_prim = UsdLux.SphereLight.Define(stage, light_path)
    light_prim.GetIntensityAttr().Set(5000.0)
    light_prim.GetRadiusAttr().Set(0.5)
    light_prim.GetColorAttr().Set(Gf.Vec3f(1.0, 0.3, 0.05))
    UsdGeom.Xformable(light_prim).AddTranslateOp().Set(Gf.Vec3d(fx, fy, fz + 1.0))

print(f"\nDone — {len(FIRE_POSITIONS)} fire emitters + lights placed, building darkened. Press Play.")
