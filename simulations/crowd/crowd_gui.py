"""Crowd GUI script — run via Isaac Sim Script Editor with:
exec(open('/home/ad.msoe.edu/somed/isaac-sim-msoe/simulations/crowd/crowd_gui.py').read())
"""
import omni.usd
import omni.timeline
import omni.kit.app
from pxr import UsdSkel, Sdf

stage = omni.usd.get_context().get_stage()
app   = omni.kit.app.get_app()
tl    = omni.timeline.get_timeline_interface()

# ── Find SkelRoot and SkelAnimations ──────────────────────────────────────────
skel_roots = []
skel_anims = []
for prim in stage.Traverse():
    t = prim.GetTypeName()
    if t == "SkelRoot":
        skel_roots.append(prim)
    elif t == "SkelAnimation":
        skel_anims.append(prim)

print(f"SkelRoots: {[str(p.GetPath()) for p in skel_roots]}")
print(f"SkelAnims: {[str(p.GetPath()) for p in skel_anims]}")

# ── Bind the first SkelAnimation to the first SkelRoot ───────────────────────
if skel_roots and skel_anims:
    binding = UsdSkel.BindingAPI.Apply(skel_roots[0])
    binding.CreateAnimationSourceRel().SetTargets([skel_anims[0].GetPath()])
    print(f"Bound {skel_anims[0].GetPath()} -> {skel_roots[0].GetPath()}")
    app.update()
    app.update()

# ── Diagnostic: manually set time to frame 20 and check joint positions ───────
# If the skeleton deformer is working, the pose should differ from T-pose at frame 20
cache = UsdSkel.Cache()
skel_query = None
if skel_roots:
    cache.Populate(UsdSkel.Root(skel_roots[0]), None)
    bindings = cache.ComputeSkelBindings(UsdSkel.Root(skel_roots[0]), None)
    print(f"SkelBindings found: {len(bindings)}")
    if bindings:
        skel_query = cache.GetSkelQuery(bindings[0].GetSkeleton())

# Set time to frame 20 and read joint xforms
tl.set_time_codes_per_second(24)
tl.set_current_time(20.0 / 24.0)
app.update()
app.update()

if skel_query and skel_query.IsValid():
    import numpy as np
    xforms = skel_query.ComputeJointLocalTransforms(20)
    if xforms:
        print(f"Joint xforms at frame 20: {len(xforms)} joints")
        print(f"  Root xform: {xforms[0]}")
    else:
        print("No joint xforms returned at frame 20 — animation not evaluating")
else:
    print("SkelQuery not valid — cannot read joint transforms")

# ── Now just play and see ─────────────────────────────────────────────────────
tl.set_start_time(0)
tl.set_end_time(200)
tl.set_auto_update(True)
tl.play()
print("Timeline playing — watch viewport for movement")
print("If still T-posing, UsdSkel deformation is not running in this session")
