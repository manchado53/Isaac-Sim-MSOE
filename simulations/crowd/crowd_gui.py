"""Crowd GUI script — run via Isaac Sim Script Editor with:
exec(open('/home/ad.msoe.edu/somed/isaac-sim-msoe/simulations/crowd/crowd_gui.py').read())
"""
import omni.usd
import omni.timeline
from pxr import UsdSkel, Sdf

stage = omni.usd.get_context().get_stage()

# ── Find all SkelRoots and SkelAnimations in the stage ────────────────────────
skel_roots = []
skel_anims = []

for prim in stage.Traverse():
    t = prim.GetTypeName()
    if t == "SkelRoot":
        skel_roots.append(prim)
    elif t == "SkelAnimation":
        skel_anims.append(prim)

print(f"Found {len(skel_roots)} SkelRoot(s):")
for p in skel_roots:
    print(f"  {p.GetPath()}")

print(f"Found {len(skel_anims)} SkelAnimation(s):")
for p in skel_anims:
    print(f"  {p.GetPath()}")

# ── Bind first animation to first SkelRoot ────────────────────────────────────
if skel_roots and skel_anims:
    skel_root = skel_roots[0]
    anim      = skel_anims[0]
    binding   = UsdSkel.BindingAPI.Apply(skel_root)
    binding.CreateAnimationSourceRel().SetTargets([anim.GetPath()])
    print(f"\nBound: {anim.GetPath()} -> {skel_root.GetPath()}")
else:
    print("ERROR: missing SkelRoot or SkelAnimation in stage")

# ── Play timeline ─────────────────────────────────────────────────────────────
tl = omni.timeline.get_timeline_interface()
tl.play()
print("Timeline playing — character should now animate")
