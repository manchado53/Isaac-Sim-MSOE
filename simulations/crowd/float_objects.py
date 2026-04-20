"""Animate door/furniture prims to float and rise with the flood water.

Water params match full_flood.py: START_Z=-1, FLOOD_RISE=42 over 600 frames.

exec(open('/home/ad.msoe.edu/somed/isaac-sim-msoe/simulations/crowd/float_objects.py').read())
"""
import omni.usd, math, random
from pxr import UsdGeom, Gf, UsdPhysics

random.seed(42)
stage = omni.usd.get_context().get_stage()

FPS          = 30
RISE_FRAMES  = 600
TOTAL_FRAMES = 900
WATER_START  = -1.0
WATER_RISE   = 42.0

def water_z(frame):
    return WATER_START + min(frame / RISE_FRAMES, 1.0) * WATER_RISE

# Find door/furniture prims — search FamilyData and Rooms for moveable objects
keywords = ["door", "Door", "chair", "Chair", "table", "Table", "furniture"]
candidates = []
for prim in stage.Traverse():
    path = str(prim.GetPath())
    name = prim.GetName()
    if not prim.IsA(UsdGeom.Xform):
        continue
    if any(k in name for k in keywords):
        # Only top-level objects (not deep children)
        if path.count("/") <= 6:
            candidates.append(prim)

print(f"Found {len(candidates)} floating candidates:")
for p in candidates[:20]:
    print(f"  {p.GetPath()}")

if not candidates:
    print("No doors/furniture found — searching all Xforms under FamilyData...")
    fd = stage.GetPrimAtPath("/World/Building")
    for prim in stage.Traverse():
        path = str(prim.GetPath())
        if "FamilyData" in path and prim.IsA(UsdGeom.Xform):
            if path.count("/") <= 7:
                candidates.append(prim)
                if len(candidates) >= 15:
                    break
    print(f"Using {len(candidates)} prims from FamilyData")
    for p in candidates:
        print(f"  {p.GetPath()}")

# Animate up to 15 objects floating
floaters = candidates[:15]
print(f"\nAnimating {len(floaters)} floating objects...")

for prim in floaters:
    # Disable physics so keyframe animation takes over
    rb = UsdPhysics.RigidBodyAPI(prim)
    if rb:
        try:
            rb.GetRigidBodyEnabledAttr().Set(False)
        except:
            pass

    xf = UsdGeom.Xformable(prim)

    # Get current world position at frame 0
    try:
        bbox = UsdGeom.BBoxCache(0, [UsdGeom.Tokens.default_])
        b = bbox.ComputeWorldBound(prim).GetRange()
        orig_x = (b.GetMin()[0] + b.GetMax()[0]) / 2
        orig_y = (b.GetMin()[1] + b.GetMax()[1]) / 2
        orig_z = b.GetMin()[2]
    except:
        orig_x, orig_y, orig_z = 0, 0, 0

    # Random float params
    bob_amp   = random.uniform(0.05, 0.15)   # bobbing amplitude
    bob_freq  = random.uniform(0.8, 2.0)     # bobbing frequency
    drift_x   = random.uniform(-0.005, 0.005) # slow horizontal drift per frame
    drift_y   = random.uniform(-0.005, 0.005)
    spin_rate = random.uniform(-0.3, 0.3)    # rotation per frame (degrees)
    surface_offset = random.uniform(0.1, 0.4) # how high above water surface

    xf.ClearXformOpOrder()
    t_op   = xf.AddTranslateOp()
    rot_op = xf.AddRotateXYZOp()

    for frame in range(TOTAL_FRAMES + 1):
        wz = water_z(frame)
        # Object starts at original position, rises when water reaches it
        if orig_z < wz - 0.5:
            # Floating: rise with water + bobbing
            t = frame / FPS
            bob = bob_amp * math.sin(bob_freq * t * 2 * math.pi)
            fz = wz + surface_offset + bob
            fx = orig_x + drift_x * (frame - max(0, frame))
            fy = orig_y + drift_y * (frame - max(0, frame))
            rot = spin_rate * (frame - RISE_FRAMES * (orig_z - WATER_START) / WATER_RISE)
        else:
            fz = orig_z
            fx, fy = orig_x, orig_y
            rot = 0.0

        t_op.Set(Gf.Vec3d(fx, fy, fz), frame)
        rot_op.Set(Gf.Vec3f(0.0, 0.0, rot), frame)

print("Done — floating objects animated. Press Play.")
