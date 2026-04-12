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

# ── Find SkelRoot and all SkelAnimations ──────────────────────────────────────
skel_roots = []
skel_anims = []
for prim in stage.Traverse():
    t = prim.GetTypeName()
    if t == "SkelRoot":
        skel_roots.append(str(prim.GetPath()))
    elif t == "SkelAnimation":
        skel_anims.append(str(prim.GetPath()))

print(f"SkelRoots: {skel_roots}")
print(f"SkelAnims: {skel_anims}")

# ── Pick the walk animation — prefer anything with "walk" in path ─────────────
walk_anim_path = None
for p in skel_anims:
    if "walk" in p.lower():
        walk_anim_path = p
        break

# Fallback to first anim
if walk_anim_path is None and skel_anims:
    walk_anim_path = skel_anims[0]

print(f"Using animation: {walk_anim_path}")

# ── Bind to SkelRoot ──────────────────────────────────────────────────────────
if skel_roots and walk_anim_path:
    skel_root_prim = stage.GetPrimAtPath(skel_roots[0])
    binding = UsdSkel.BindingAPI.Apply(skel_root_prim)
    binding.CreateAnimationSourceRel().SetTargets([Sdf.Path(walk_anim_path)])
    print(f"Bound: {walk_anim_path} -> {skel_roots[0]}")
    app.update()
    app.update()
else:
    print("ERROR: missing SkelRoot or SkelAnimation")

# ── Play timeline ─────────────────────────────────────────────────────────────
tl.set_start_time(0)
tl.set_end_time(200)
tl.set_time_codes_per_second(24)
tl.set_auto_update(True)
tl.play()
print("Done — watch viewport for walk animation")
