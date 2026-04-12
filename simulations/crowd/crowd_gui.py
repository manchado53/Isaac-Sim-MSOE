"""Crowd GUI script — uses omni.anim.people (DAP) properly.

The DAP system requires:
  1. Characters loaded under a character root prim path
  2. A command file with lines: "char_name GoTo x y z angle"
  3. CharacterBehavior BehaviorScript attached to each character prim
  4. Hit Play — omni.anim.graph.core drives the animation

Run via Isaac Sim Script Editor:
  exec(open('/home/ad.msoe.edu/somed/isaac-sim-msoe/simulations/crowd/crowd_gui.py').read())
"""
import os
import carb
import omni.usd
import omni.kit.app
import omni.kit.commands
import omni.timeline
from pxr import UsdGeom, Sdf, Gf
from isaacsim.core.utils.stage import add_reference_to_stage
from isaacsim.storage.native import get_assets_root_path

stage  = omni.usd.get_context().get_stage()
app    = omni.kit.app.get_app()

# ── Config ────────────────────────────────────────────────────────────────────
ASSETS_ROOT   = get_assets_root_path()
CHAR_USD      = ASSETS_ROOT + "/Isaac/People/Characters/female_adult_police_01_new/female_adult_police_01_new.usd"
CMD_FILE_PATH = "/home/ad.msoe.edu/somed/isaac-sim-msoe/videos/crowd_commands.txt"

# Character root — DAP requires all characters to be under this path
CHAR_ROOT = "/World/Characters"

# BehaviorScript path inside the DAP extension
_ext_mgr  = omni.kit.app.get_app().get_extension_manager()
_ext_path = _ext_mgr.get_extension_path_by_module("omni.anim.people")
BEHAVIOR_SCRIPT = _ext_path + "/omni/anim/people/scripts/character_behavior.py"

print(f"Assets root: {ASSETS_ROOT}")
print(f"Behavior script: {BEHAVIOR_SCRIPT}")
print(f"Script exists: {os.path.isfile(BEHAVIOR_SCRIPT)}")

# ── Floor 4 target positions (a few nav node XY positions) ───────────────────
# Z=26.9 is the floor, but DAP GoTo uses world XYZ
FLOOR_Z = 26.9
TARGETS = [
    (-19.7, 70.1),
    (-22.0, 72.0),
    (-17.0, 68.5),
    (-21.0, 67.0),
]

# ── Ensure character root xform exists ───────────────────────────────────────
if not stage.GetPrimAtPath(CHAR_ROOT).IsValid():
    stage.DefinePrim(CHAR_ROOT, "Xform")
    print(f"Created {CHAR_ROOT}")

# ── Set DAP settings ──────────────────────────────────────────────────────────
settings = carb.settings.get_settings()
settings.set("/exts/omni.anim.people/character_prim_path", CHAR_ROOT)
settings.set("/exts/omni.anim.people/command_file_path",   CMD_FILE_PATH)
settings.set("/exts/omni.anim.people/navmesh_enabled",     False)
settings.set("/exts/omni.anim.people/dynamic_avoidance_enabled", False)
print("DAP settings configured")

# ── Load one test character ───────────────────────────────────────────────────
char_prim_path = f"{CHAR_ROOT}/officer_01"
add_reference_to_stage(CHAR_USD, char_prim_path)

# Position at Floor 4
char_prim = stage.GetPrimAtPath(char_prim_path)
xform     = UsdGeom.XformCommonAPI(char_prim)
xform.SetTranslate(Gf.Vec3d(-19.7, 70.1, FLOOR_Z))

for _ in range(3):
    app.update()

print(f"Character loaded at {char_prim_path}")

# ── Attach CharacterBehavior script ───────────────────────────────────────────
omni.kit.commands.execute(
    "ApplyScriptingAPICommand",
    paths=[Sdf.Path(char_prim_path)]
)
attr = char_prim.GetAttribute("omni:scripting:scripts")
attr.Set([BEHAVIOR_SCRIPT])
print(f"BehaviorScript attached")

for _ in range(3):
    app.update()

# ── Write command file ────────────────────────────────────────────────────────
# Format: "character_name GoTo x y z angle"
# character_name = prim name under CHAR_ROOT = "officer_01"
with open(CMD_FILE_PATH, "w") as f:
    # Walk between two points
    f.write("officer_01 GoTo -22.0 72.0 26.9 0\n")
    f.write("officer_01 GoTo -17.0 68.5 26.9 180\n")
    f.write("officer_01 GoTo -19.7 70.1 26.9 90\n")

print(f"Command file written: {CMD_FILE_PATH}")

# ── Play ──────────────────────────────────────────────────────────────────────
tl = omni.timeline.get_timeline_interface()
tl.set_start_time(0)
tl.set_end_time(500)
tl.set_time_codes_per_second(24)
tl.play()
print("Playing — character should start walking on Play")
print("If still T-posing, check Console for DAP errors about animation graph")
