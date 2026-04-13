"""Crowd GUI script.
exec(open('/home/ad.msoe.edu/somed/isaac-sim-msoe/simulations/crowd/crowd_gui.py').read())
"""
import os
import carb
import omni.usd
import omni.kit.app
import omni.kit.commands
from pxr import UsdGeom, Sdf, Gf
from isaacsim.storage.native import get_assets_root_path

stage = omni.usd.get_context().get_stage()
app   = omni.kit.app.get_app()

# ── Verify building is loaded ─────────────────────────────────────────────────
top_prims = [str(p.GetPath()) for p in stage.GetPseudoRoot().GetChildren()]
print(f"Top-level prims: {top_prims}")

# ── Config ────────────────────────────────────────────────────────────────────
ASSETS_ROOT = get_assets_root_path()
CHAR_USD    = ASSETS_ROOT + "/Isaac/People/Characters/female_adult_police_01_new/female_adult_police_01_new.usd"
CHAR_ROOT   = "/World/Characters"
CHAR_PATH   = f"{CHAR_ROOT}/officer_01"
CMD_FILE    = "/home/ad.msoe.edu/somed/isaac-sim-msoe/videos/crowd_commands.txt"
FLOOR_Z     = 26.9

# ── Get behavior script path ──────────────────────────────────────────────────
ext_mgr     = omni.kit.app.get_app().get_extension_manager()
ext_path    = ext_mgr.get_extension_path_by_module("omni.anim.people")
BHVR_SCRIPT = ext_path + "/omni/anim/people/scripts/character_behavior.py"
print(f"BehaviorScript exists: {os.path.isfile(BHVR_SCRIPT)}")

# ── Create character root ─────────────────────────────────────────────────────
if not stage.GetPrimAtPath(CHAR_ROOT).IsValid():
    stage.DefinePrim(CHAR_ROOT, "Xform")

# ── Character definitions: (prim_name, x, y) ─────────────────────────────────
CHARACTERS = [
    ("officer_01", -19.7, 70.1),
    ("officer_02", -22.0, 72.0),
]

for char_name, cx, cy in CHARACTERS:
    path = f"{CHAR_ROOT}/{char_name}"
    if stage.GetPrimAtPath(path).IsValid():
        print(f"{path} already exists, skipping")
        continue
    prim = stage.DefinePrim(path, "Xform")
    prim.GetReferences().AddReference(CHAR_USD)
    UsdGeom.XformCommonAPI(prim).SetTranslate(Gf.Vec3d(cx, cy, FLOOR_Z))
    print(f"Loaded {path} at ({cx}, {cy}, {FLOOR_Z})")

app.update()
app.update()

prim = stage.GetPrimAtPath(CHAR_PATH)
print(f"Character prim valid: {prim.IsValid()}")

# ── Set DAP settings ──────────────────────────────────────────────────────────
settings = carb.settings.get_settings()
settings.set("/exts/omni.anim.people/character_prim_path", CHAR_ROOT)
settings.set("/exts/omni.anim.people/command_file_path",   CMD_FILE)
settings.set("/exts/omni.anim.people/navmesh_enabled",     False)
settings.set("/exts/omni.anim.people/dynamic_avoidance_enabled", False)

# ── Write command file ────────────────────────────────────────────────────────
with open(CMD_FILE, "w") as f:
    f.write("officer_01 GoTo -22.0 72.0 26.9 0\n")
    f.write("officer_01 GoTo -17.0 68.5 26.9 180\n")
    f.write("officer_01 GoTo -19.7 70.1 26.9 90\n")
    f.write("officer_02 GoTo -17.0 68.5 26.9 0\n")
    f.write("officer_02 GoTo -22.0 72.0 26.9 180\n")
    f.write("officer_02 GoTo -19.7 70.1 26.9 270\n")
print(f"Command file written")

# ── Attach BehaviorScript to all characters ───────────────────────────────────
for char_name, _, _ in CHARACTERS:
    path = f"{CHAR_ROOT}/{char_name}"
    omni.kit.commands.execute("ApplyScriptingAPICommand", paths=[Sdf.Path(path)])
    attr = stage.GetPrimAtPath(path).GetAttribute("omni:scripting:scripts")
    if attr:
        attr.Set([BHVR_SCRIPT])
        print(f"BehaviorScript attached to {char_name}")
    else:
        print(f"WARNING: Could not attach BehaviorScript to {char_name}")

app.update()
app.update()

print("\n=== DONE ===")
print("Now press the PLAY button in the toolbar.")
print("Character should walk between the GoTo points.")
print("Check Console for any DAP errors.")
