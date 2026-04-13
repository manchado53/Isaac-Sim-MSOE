"""Crowd GUI script — AnimationGraph approach (confirmed working per tutorial).

Steps:
  1. Load character USD
  2. Create AnimationGraph on the character
  3. Add AnimationClip node pointing to walk skelanim
  4. Assign graph to SkelRoot via SkelBindingAPI

exec(open('/home/ad.msoe.edu/somed/isaac-sim-msoe/simulations/crowd/crowd_gui.py').read())
"""
import omni.usd
import omni.kit.app
import omni.anim.graph.core as ag
from pxr import UsdGeom, UsdSkel, Sdf, Gf, Usd
from isaacsim.storage.native import get_assets_root_path
from isaacsim.core.utils.stage import add_reference_to_stage

stage = omni.usd.get_context().get_stage()
app   = omni.kit.app.get_app()

ASSETS_ROOT = get_assets_root_path()
CHAR_USD    = ASSETS_ROOT + "/Isaac/People/Characters/female_adult_police_01_new/female_adult_police_01_new.usd"
WALK_USD    = ASSETS_ROOT + "/Isaac/People/Animations/stand_walk_1.skelanim.usd"
FLOOR_Z     = 26.9
CHAR_ROOT   = "/World/Characters"

# Characters: (name, x, y)
CHARACTERS = [
    ("officer_01", -19.7, 70.1),
    ("officer_02", -22.0, 72.5),
]

# ── Create character root ─────────────────────────────────────────────────────
if not stage.GetPrimAtPath(CHAR_ROOT).IsValid():
    stage.DefinePrim(CHAR_ROOT, "Xform")

# ── Load characters ───────────────────────────────────────────────────────────
for char_name, cx, cy in CHARACTERS:
    path = f"{CHAR_ROOT}/{char_name}"
    if not stage.GetPrimAtPath(path).IsValid():
        prim = stage.DefinePrim(path, "Xform")
        prim.GetReferences().AddReference(CHAR_USD)
        UsdGeom.XformCommonAPI(prim).SetTranslate(Gf.Vec3d(cx, cy, FLOOR_Z))
        print(f"Loaded {path}")
    else:
        print(f"{path} already exists")

app.update()
app.update()
app.update()

# ── Wire up AnimationGraph on each character ──────────────────────────────────
for char_name, _, _ in CHARACTERS:
    char_path = f"{CHAR_ROOT}/{char_name}"

    # Find SkelRoot under this character
    skel_root_prim = None
    for prim in stage.Traverse():
        if str(prim.GetPath()).startswith(char_path) and prim.GetTypeName() == "SkelRoot":
            skel_root_prim = prim
            break

    if skel_root_prim is None:
        print(f"ERROR: No SkelRoot found under {char_path}")
        continue

    skel_root_path = str(skel_root_prim.GetPath())
    print(f"SkelRoot: {skel_root_path}")

    # ── Load walk animation as a reference under the character ────────────────
    walk_ref_path = f"{char_path}/WalkAnim"
    if not stage.GetPrimAtPath(walk_ref_path).IsValid():
        walk_prim = stage.DefinePrim(walk_ref_path)
        walk_prim.GetReferences().AddReference(WALK_USD)
        print(f"Walk anim loaded at {walk_ref_path}")

    app.update()
    app.update()

    # Find the SkelAnimation prim inside WalkAnim
    walk_anim_path = None
    for prim in stage.Traverse():
        p = str(prim.GetPath())
        if p.startswith(walk_ref_path) and prim.GetTypeName() == "SkelAnimation":
            walk_anim_path = p
            print(f"SkelAnimation found: {walk_anim_path}")
            break

    if walk_anim_path is None:
        print(f"ERROR: No SkelAnimation found under {walk_ref_path}")
        continue

    # ── Create AnimationGraph prim ────────────────────────────────────────────
    graph_path = f"{char_path}/AnimationGraph"
    if not stage.GetPrimAtPath(graph_path).IsValid():
        graph_prim = stage.DefinePrim(graph_path, "AnimationGraph")
        print(f"Created AnimationGraph at {graph_path}")
    else:
        graph_prim = stage.GetPrimAtPath(graph_path)
        print(f"AnimationGraph already exists at {graph_path}")

    # ── Add AnimationClip node inside graph ───────────────────────────────────
    clip_path = f"{graph_path}/AnimationClip"
    if not stage.GetPrimAtPath(clip_path).IsValid():
        clip_prim = stage.DefinePrim(clip_path, "AnimationClip")
        # Point AnimationClip to the walk skelanim
        clip_prim.CreateAttribute("animationSource", Sdf.ValueTypeNames.String).Set(walk_anim_path)
        print(f"Created AnimationClip at {clip_path} -> {walk_anim_path}")

    app.update()
    app.update()

    # ── Assign AnimationGraph to SkelRoot ─────────────────────────────────────
    binding = UsdSkel.BindingAPI.Apply(skel_root_prim)
    binding.CreateAnimationSourceRel().SetTargets([Sdf.Path(graph_path)])
    print(f"Assigned AnimationGraph to SkelRoot")

app.update()
print("\n=== DONE — press Play in toolbar ===")
