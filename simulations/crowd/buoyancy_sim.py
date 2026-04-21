"""Apply real buoyancy physics to props floating on the rising flood water.

Adds RigidBody to any prop prim under /World that isn't the Building/Ocean,
then applies upward buoyancy force + water damping each physics step.

Run AFTER merge_ocean_building.py. Keep the script running (it registers a
physics callback). Stop simulation to end.

exec(open('/home/ad.msoe.edu/somed/isaac-sim-msoe/simulations/crowd/buoyancy_sim.py').read())
"""
import math, numpy as np
import omni.usd
import omni.physx
from omni.physx import get_physx_scene_query_interface
from pxr import UsdGeom, UsdPhysics, Gf, PhysxSchema

stage = omni.usd.get_context().get_stage()

# ── Water level (matches merge_ocean_building.py) ────────────────────────────
bbox_cache = UsdGeom.BBoxCache(0, [UsdGeom.Tokens.default_])
building   = stage.GetPrimAtPath("/World/Building")
b          = bbox_cache.ComputeWorldBound(building).GetRange()
floor_z    = b.GetMin()[2]
roof_z     = b.GetMax()[2]
START_Z    = floor_z - 5
END_Z      = roof_z + 200
RISE_FRAMES = 4800
FPS         = 30

def water_z_at_time(t):
    frame = t * FPS
    return START_Z + min(frame / RISE_FRAMES, 1.0) * (END_Z - START_Z)

# ── Find floatable props ──────────────────────────────────────────────────────
SKIP = {"Building", "MeshOut", "MeshIn", "ActionGraph", "Looks",
        "PhysicsScene", "PhysicsMaterials", "Environment", "CameraRoot",
        "NavMeshVolume", "Robots", "physicsScene"}

props = []
for prim in stage.GetPrimAtPath("/World").GetChildren():
    if prim.GetName() in SKIP:
        continue
    if not prim.IsA(UsdGeom.Xformable):
        continue
    props.append(prim)
    print(f"Floatable prop: {prim.GetPath()}")

if not props:
    print("No props found under /World — drag a forklift/dolly/etc into the scene first")
else:
    print(f"\nFound {len(props)} props to apply buoyancy to")

# ── Add RigidBody + collision to each prop if missing ────────────────────────
for prim in props:
    if not prim.HasAPI(UsdPhysics.RigidBodyAPI):
        UsdPhysics.RigidBodyAPI.Apply(prim)
        print(f"  Added RigidBodyAPI to {prim.GetPath()}")
    rb = UsdPhysics.RigidBodyAPI(prim)
    rb.GetRigidBodyEnabledAttr().Set(True)

    # Add mass if missing
    if not prim.HasAPI(UsdPhysics.MassAPI):
        mass_api = UsdPhysics.MassAPI.Apply(prim)
        mass_api.GetMassAttr().Set(50.0)  # 50kg default

    # Add PhysX damping
    if not prim.HasAPI(PhysxSchema.PhysxRigidBodyAPI):
        PhysxSchema.PhysxRigidBodyAPI.Apply(prim)
    physx_rb = PhysxSchema.PhysxRigidBodyAPI(prim)
    physx_rb.GetLinearDampingAttr().Set(2.0)
    physx_rb.GetAngularDampingAttr().Set(3.0)

# ── Buoyancy callback ─────────────────────────────────────────────────────────
WATER_DENSITY = 1000.0  # kg/m^3
GRAVITY       = 9.81

def apply_buoyancy(dt):
    from omni.physx import get_physx_interface
    physx = get_physx_interface()

    sim_time = omni.timeline.get_timeline_interface().get_current_time()
    wz = water_z_at_time(sim_time)

    for prim in props:
        try:
            b2  = UsdGeom.BBoxCache(0, [UsdGeom.Tokens.default_]).ComputeWorldBound(prim).GetRange()
            obj_z   = (b2.GetMin()[2] + b2.GetMax()[2]) / 2
            obj_h   = b2.GetMax()[2] - b2.GetMin()[2]
            obj_bot = b2.GetMin()[2]

            if obj_bot > wz:
                continue  # above water, no force

            # Submerged fraction
            submerged = min(wz - obj_bot, obj_h) / obj_h if obj_h > 0 else 0
            volume    = max(obj_h ** 3, 0.001)  # rough cube volume estimate
            force_z   = WATER_DENSITY * volume * submerged * GRAVITY

            path = str(prim.GetPath())
            physx.apply_force_at_pos(path, Gf.Vec3f(0, 0, force_z),
                                     Gf.Vec3f(obj_z, 0, 0), "Force")

            # Increase damping as object submerges
            if prim.HasAPI(PhysxSchema.PhysxRigidBodyAPI):
                d = 2.0 + submerged * 8.0
                PhysxSchema.PhysxRigidBodyAPI(prim).GetLinearDampingAttr().Set(d)
        except Exception as e:
            pass

# Register physics callback
import omni.physx as _physx
_sub = _physx.get_physx_interface().subscribe_physics_step_events(apply_buoyancy)

print("\nBuoyancy physics active — press Play. Objects will float as water rises.")
print("(Re-run script to reset if simulation is restarted)")
