"""Spawn NVIDIA Flow fire emitters in the building corridor/rooms.

Run with the building USD loaded. Places fire at specified world positions.

exec(open('/home/ad.msoe.edu/somed/isaac-sim-msoe/simulations/crowd/fire_sim.py').read())
"""
import omni.usd
from pxr import UsdGeom, Gf, Sdf, UsdShade

stage = omni.usd.get_context().get_stage()

# Fire locations — corridor + a couple rooms (world meters)
FIRE_POSITIONS = [
    (-22.79, 66.0,  22.34),  # corridor start
    (-22.79, 62.0,  22.34),  # corridor mid
    (-30.0,  70.0,  22.34),  # side room
    (-15.0,  68.0,  22.34),  # opposite room
]

FIRE_SCALE = 2.0  # increase for bigger flames

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
    xf.AddScaleOp().Set(Gf.Vec3f(FIRE_SCALE, FIRE_SCALE, FIRE_SCALE))

    fire_prim.GetReferences().AddReference(
        "/home/ad.msoe.edu/somed/isaacsim/extscache/omni.flowusd-107.1.8+107.1.0.lx64.r.cp311.u353/data/presets/Fire/Fire.usda"
    )
    print(f"  Fire_{i} placed at ({fx}, {fy}, {fz})")

print(f"\nDone — {len(FIRE_POSITIONS)} fire emitters placed. Press Play to ignite.")
