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

# ── Load character ────────────────────────────────────────────────────────────
if stage.GetPrimAtPath(CHAR_PATH).IsValid():
    print(f"Character already exists at {CHAR_PATH}, skipping load")
else:
    prim = stage.DefinePrim(CHAR_PATH, "Xform")
    prim.GetReferences().AddReference(CHAR_USD)
    print(f"Character referenced at {CHAR_PATH}")

app.update()
app.update()

# Verify it loaded
prim = stage.GetPrimAtPath(CHAR_PATH)
print(f"Character prim valid: {prim.IsValid()}")
print(f"Character prim type: {prim.GetTypeName()}")

# ── Position at Floor 4 ───────────────────────────────────────────────────────
xform = UsdGeom.XformCommonAPI(prim)
xform.SetTranslate(Gf.Vec3d(-19.7, 70.1, FLOOR_Z))
print(f"Positioned at Floor 4 Z={FLOOR_Z}")

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
print(f"Command file written")

# ── Attach BehaviorScript ─────────────────────────────────────────────────────
omni.kit.commands.execute("ApplyScriptingAPICommand", paths=[Sdf.Path(CHAR_PATH)])
attr = stage.GetPrimAtPath(CHAR_PATH).GetAttribute("omni:scripting:scripts")
if attr:
    attr.Set([BHVR_SCRIPT])
    print(f"BehaviorScript attached")
else:
    print("WARNING: Could not attach BehaviorScript")

app.update()
app.update()

print("\n=== DONE ===")
print("Now press the PLAY button in the toolbar.")
print("Character should walk between the GoTo points.")
print("Check Console for any DAP errors.")
