"""Inspect S3 character + animation USD hierarchy and test binding.

Loads female_adult_police_01_new.usd + stand_walk_1.skelanim.usd from S3,
binds the animation, plays timeline, captures a frame to confirm it works.

Output: videos/s3_character_test.png + videos/s3_character_inspect.txt
"""
import os, sys
os.environ["OMNI_KIT_ACCEPT_EULA"] = "YES"
os.environ["PRIVACY_CONSENT"] = "Y"

from isaacsim import SimulationApp
app = SimulationApp({"headless": True, "width": 640, "height": 480})

from pxr import Usd, UsdSkel, UsdGeom, Sdf, UsdLux, Gf
import omni.usd
import omni.timeline
import numpy as np
from isaacsim.storage.native import get_assets_root_path
from isaacsim.core.api import World
from isaacsim.core.utils.stage import add_reference_to_stage
from isaacsim.sensors.camera import Camera
import carb

PROJECT_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), "../.."))
OUT_TXT = os.path.join(PROJECT_DIR, "videos/s3_character_inspect.txt")
OUT_PNG = os.path.join(PROJECT_DIR, "videos/s3_character_test.png")

lines = []
def w(s=""): lines.append(str(s))

assets_root = get_assets_root_path()
w(f"Assets root: {assets_root}")

CHAR_USD = assets_root + "/Isaac/People/Characters/female_adult_police_01_new/female_adult_police_01_new.usd"
ANIM_USD = assets_root + "/Isaac/People/Animations/stand_walk_1.skelanim.usd"

w(f"Character USD: {CHAR_USD}")
w(f"Animation USD: {ANIM_USD}")
w()

# ── Inspect animation USD to find SkelAnimation prim path ─────────────────────
w("=== stand_walk_1.skelanim.usd hierarchy ===")
anim_stage = Usd.Stage.Open(ANIM_USD)
anim_prim_path = None
for prim in anim_stage.Traverse():
    tname = prim.GetTypeName()
    w(f"  {prim.GetPath()}  [{tname}]")
    if tname == "SkelAnimation":
        anim_prim_path = str(prim.GetPath())
        anim = UsdSkel.Animation(prim)
        joints = anim.GetJointsAttr().Get()
        times  = anim.GetTranslationsAttr().GetTimeSamples() if anim.GetTranslationsAttr() else []
        w(f"    -> joints={len(joints) if joints else 0}, time_samples={len(times)}")
        if times:
            w(f"    -> time range: {times[0]:.1f} – {times[-1]:.1f}")
w(f"\nSkelAnimation prim path: {anim_prim_path}")
w()

# ── Inspect character USD to find SkelRoot ─────────────────────────────────────
w("=== female_adult_police_01_new.usd top-level hierarchy ===")
char_stage = Usd.Stage.Open(CHAR_USD)
skel_root_path = None
for prim in char_stage.Traverse():
    tname = prim.GetTypeName()
    depth = len(str(prim.GetPath()).strip("/").split("/"))
    if depth <= 4:  # only show top levels to keep output readable
        w(f"  {'  '*depth}{prim.GetPath()}  [{tname}]")
    if tname == "SkelRoot" and skel_root_path is None:
        skel_root_path = str(prim.GetPath())
        w(f"    *** FOUND SkelRoot: {skel_root_path} ***")
w(f"\nSkelRoot path: {skel_root_path}")
w()

# ── Now set up a live stage and test the binding ───────────────────────────────
w("=== Live binding test ===")
world = World(stage_units_in_meters=1.0)
stage = omni.usd.get_context().get_stage()

# Light
dome = UsdLux.DomeLight.Define(stage, Sdf.Path("/World/Dome"))
dome.CreateIntensityAttr(5000)

# Load character
CHAR_ROOT = "/World/TestChar"
add_reference_to_stage(CHAR_USD, CHAR_ROOT)
w(f"Character loaded at {CHAR_ROOT}")

# Load animation as child reference
ANIM_REF = f"{CHAR_ROOT}/WalkAnim"
anim_prim = stage.DefinePrim(ANIM_REF)
anim_prim.GetReferences().AddReference(ANIM_USD)
w(f"Animation referenced at {ANIM_REF}")

# Flush so prims are resolved
app.update()
app.update()

# Find SkelRoot in live stage
live_skel_root = None
for prim in stage.Traverse():
    if prim.GetTypeName() == "SkelRoot":
        live_skel_root = str(prim.GetPath())
        w(f"Live SkelRoot found: {live_skel_root}")
        break

if live_skel_root and anim_prim_path:
    live_anim_path = f"{ANIM_REF}{anim_prim_path}"
    skel_root_prim = stage.GetPrimAtPath(live_skel_root)
    binding = UsdSkel.BindingAPI.Apply(skel_root_prim)
    binding.CreateAnimationSourceRel().SetTargets([Sdf.Path(live_anim_path)])
    w(f"Bound animation: {live_anim_path} → {live_skel_root}")
else:
    w("WARNING: Could not bind — SkelRoot or anim path missing")

# Play timeline
timeline = omni.timeline.get_timeline_interface()
timeline.set_end_time(100.0)
timeline.set_time_codes_per_second(24.0)
timeline.play()
w("Timeline playing")

# Place camera facing character
cam = Camera(
    prim_path="/World/TestCam",
    position=np.array([0.0, -3.0, 1.5]),
    orientation=np.array([1.0, 0.0, 0.0, 0.0]),
    frequency=30,
    resolution=(640, 480),
)

world.reset()
app.update()
cam.initialize()

for _ in range(30):
    world.step(render=True)

# Set camera to look at character at time=0.5s (12 frames in)
timeline.set_current_time(0.5)
world.step(render=True)
world.step(render=True)

rgba = cam.get_rgba()
w(f"Captured frame: shape={rgba.shape if rgba is not None else None}, "
  f"mean={rgba.mean():.1f if rgba is not None else -1}")

# Save PNG
if rgba is not None:
    from PIL import Image
    img = Image.fromarray(rgba[:, :, :3].astype(np.uint8))
    img.save(OUT_PNG)
    w(f"Saved: {OUT_PNG}")

os.makedirs(os.path.dirname(OUT_TXT), exist_ok=True)
with open(OUT_TXT, "w") as f:
    f.write("\n".join(lines))
    f.flush()

print(f"[INFO] Written to {OUT_TXT}", flush=True)
timeline.stop()
app.close()
