"""Crowd GUI script — run via Isaac Sim Script Editor with:
exec(open('/home/ad.msoe.edu/somed/isaac-sim-msoe/simulations/crowd/crowd_gui.py').read())
"""
import omni.usd
import omni.timeline
import omni.kit.app
from pxr import UsdSkel, UsdGeom, Sdf

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

# ── Load the walk animation as a new reference on the character root ──────────
# The character root prim (parent of ManRoot)
char_root_path = "/female_adult_police_01_new"
char_root = stage.GetPrimAtPath(char_root_path)

assets_root = "https://omniverse-content-production.s3-us-west-2.amazonaws.com/Assets/Isaac/5.1"
walk_usd = assets_root + "/Isaac/People/Animations/stand_walk_1.skelanim.usd"

# Add walk anim as a child reference
walk_ref_path = char_root_path + "/WalkAnim"
walk_prim = stage.DefinePrim(walk_ref_path)
walk_prim.GetReferences().AddReference(walk_usd)
print(f"\nAdded walk anim reference at {walk_ref_path}")

# Flush so reference resolves
app = omni.kit.app.get_app()
for _ in range(5):
    app.update()

# Find the SkelAnimation inside the walk reference
walk_anim_path = None
for prim in stage.Traverse():
    if walk_ref_path in str(prim.GetPath()) and prim.GetTypeName() == "SkelAnimation":
        walk_anim_path = prim.GetPath()
        print(f"Walk SkelAnimation found at: {walk_anim_path}")
        break

if walk_anim_path is None:
    print("WARNING: Walk SkelAnimation not found, listing all prims under WalkAnim:")
    for prim in stage.Traverse():
        if walk_ref_path in str(prim.GetPath()):
            print(f"  {prim.GetPath()}  [{prim.GetTypeName()}]")

# ── Bind walk animation to SkelRoot ──────────────────────────────────────────
if skel_roots and walk_anim_path:
    skel_root = skel_roots[0]
    binding = UsdSkel.BindingAPI.Apply(skel_root)
    binding.CreateAnimationSourceRel().SetTargets([walk_anim_path])
    print(f"\nBound walk anim: {walk_anim_path} -> {skel_root.GetPath()}")
elif skel_roots and skel_anims:
    # Fallback: bind whatever anim is in the stage
    skel_root = skel_roots[0]
    anim = skel_anims[0]
    binding = UsdSkel.BindingAPI.Apply(skel_root)
    binding.CreateAnimationSourceRel().SetTargets([anim.GetPath()])
    print(f"\nFallback bound: {anim.GetPath()} -> {skel_root.GetPath()}")
else:
    print("ERROR: missing SkelRoot or SkelAnimation")

# ── Configure and play timeline ───────────────────────────────────────────────
tl = omni.timeline.get_timeline_interface()
tl.set_start_time(0)
tl.set_end_time(100)
tl.set_time_codes_per_second(24)
tl.set_auto_update(True)
tl.play()
print("Timeline playing")
