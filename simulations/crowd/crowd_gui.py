"""Crowd GUI — use omni.anim.graph.core API to attach walk animation.
exec(open('/home/ad.msoe.edu/somed/isaac-sim-msoe/simulations/crowd/crowd_gui.py').read())
"""
import omni.usd
import omni.kit.app
import omni.kit.commands
import omni.anim.graph.core as ag
from pxr import UsdGeom, Sdf, Gf
from isaacsim.storage.native import get_assets_root_path
from isaacsim.core.utils.stage import add_reference_to_stage

stage = omni.usd.get_context().get_stage()
app   = omni.kit.app.get_app()

ASSETS_ROOT = get_assets_root_path()
CHAR_USD    = ASSETS_ROOT + "/Isaac/People/Characters/female_adult_police_01_new/female_adult_police_01_new.usd"
WALK_USD    = ASSETS_ROOT + "/Isaac/People/Animations/stand_walk_1.skelanim.usd"
FLOOR_Z     = 26.9
CHAR_ROOT   = "/World/Characters"

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

for _ in range(5):
    app.update()

# ── Apply AnimationGraph and load walk animation ──────────────────────────────
for char_name, _, _ in CHARACTERS:
    char_path = f"{CHAR_ROOT}/{char_name}"

    # Find SkelRoot
    skel_root_path = None
    for prim in stage.Traverse():
        if str(prim.GetPath()).startswith(char_path) and prim.GetTypeName() == "SkelRoot":
            skel_root_path = str(prim.GetPath())
            break

    if not skel_root_path:
        print(f"ERROR: No SkelRoot under {char_path}")
        continue
    print(f"SkelRoot: {skel_root_path}")

    # Apply AnimationGraph API to the SkelRoot using the confirmed command
    graph_path = f"{char_path}/WalkGraph"
    omni.kit.commands.execute(
        "CreateAnimationGraphCommand",
        path=Sdf.Path(graph_path),
        usd_context=omni.usd.get_context(),
    )
    for _ in range(3):
        app.update()

    graph_prim = stage.GetPrimAtPath(graph_path)
    print(f"AnimationGraph created: valid={graph_prim.IsValid()} type={graph_prim.GetTypeName()}")

    # Apply the graph to the SkelRoot
    omni.kit.commands.execute(
        "ApplyAnimationGraphAPICommand",
        paths=[Sdf.Path(skel_root_path)],
        animation_graph_path=Sdf.Path(graph_path),
    )
    for _ in range(3):
        app.update()
    print(f"AnimationGraph applied to {skel_root_path}")

    # Load walk animation onto the character
    character = ag.get_character(skel_root_path)
    if character:
        result = ag.load_animation(character, WALK_USD)
        print(f"load_animation result: {result}")
    else:
        print(f"WARNING: ag.get_character({skel_root_path}) returned None — need Play first")
        print(f"  Will try after Play is pressed via set_variable")

for _ in range(3):
    app.update()

print("\n=== Press Play in toolbar ===")
print("ag.get_character() only works at runtime — if it returned None above,")
print("press Play then run this in Script Editor:")
print("  import omni.anim.graph.core as ag")
print("  c = ag.get_character('/World/Characters/officer_01/ManRoot/female_adult_police_01')")
print("  print(c)")
