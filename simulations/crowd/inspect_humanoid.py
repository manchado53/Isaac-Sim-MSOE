"""Inspect human_skeleton.usd and walk.skelanim.usd hierarchy.

Writes full prim tree + SkelAnimation details to videos/humanoid_inspect.txt.
Also checks whether omni.anim.people extension is available.

Run: sbatch simulations/crowd/submit_inspect_humanoid.sh
"""
import os, sys
os.environ["OMNI_KIT_ACCEPT_EULA"] = "YES"
os.environ["PRIVACY_CONSENT"] = "Y"

from isaacsim import SimulationApp
app = SimulationApp({"headless": True})

from pxr import Usd, UsdSkel, UsdGeom, Sdf
import omni.usd

PROJECT_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), "../.."))
OUT_PATH    = os.path.join(PROJECT_DIR, "videos/humanoid_inspect.txt")

_EXT_BASE = os.path.expanduser("~/isaacsim/extscache")

# Find actual extension directory (version may differ on Rosie)
def find_ext(name_prefix):
    if not os.path.isdir(_EXT_BASE):
        return None
    for d in os.listdir(_EXT_BASE):
        if d.startswith(name_prefix):
            return os.path.join(_EXT_BASE, d)
    return None

retarget_dir = find_ext("omni.anim.retarget.core")
people_dir   = find_ext("omni.anim.people")

lines = []

def w(s=""):
    lines.append(str(s))

w("=" * 70)
w("HUMANOID USD INSPECTION")
w("=" * 70)
w()

# ── Extension discovery ───────────────────────────────────────────────────────
w(f"extscache: {_EXT_BASE}")
w(f"  omni.anim.retarget.core dir: {retarget_dir}")
w(f"  omni.anim.people dir:        {people_dir}")
w()

# List all anim-related extensions
w("All anim-related extensions in extscache:")
if os.path.isdir(_EXT_BASE):
    for d in sorted(os.listdir(_EXT_BASE)):
        if any(k in d for k in ["anim", "people", "character", "metropolis", "human"]):
            w(f"  {d}")
w()

# ── Helper ────────────────────────────────────────────────────────────────────
def dump_stage(usd_path, label):
    w("=" * 60)
    w(f"  {label}")
    w(f"  {usd_path}")
    w("=" * 60)
    if not os.path.isfile(usd_path):
        w("  *** FILE NOT FOUND ***")
        w()
        return
    stage = Usd.Stage.Open(usd_path)
    for prim in stage.Traverse():
        depth  = len(prim.GetPath().pathString.strip("/").split("/"))
        indent = "  " * depth
        tname  = prim.GetTypeName()
        w(f"{indent}{prim.GetPath()}  [{tname}]")
        if tname == "SkelAnimation":
            anim   = UsdSkel.Animation(prim)
            joints = anim.GetJointsAttr().Get()
            trans  = anim.GetTranslationsAttr()
            times  = trans.GetTimeSamples() if trans else []
            w(f"{indent}  -> joints={len(joints) if joints else 0}, time_samples={len(times)}")
            if times:
                w(f"{indent}  -> time range: {times[0]:.1f} – {times[-1]:.1f}")
        elif tname == "Skeleton":
            skel   = UsdSkel.Skeleton(prim)
            joints = skel.GetJointsAttr().Get()
            w(f"{indent}  -> joints={len(joints) if joints else 0}")
        elif tname in ("SkelRoot", "Mesh"):
            pass  # just show the path
    w()

# ── Inspect USD files ─────────────────────────────────────────────────────────
if retarget_dir:
    skel_usd = os.path.join(retarget_dir, "data/rigs/Human/assets/human_skeleton.usd")
    walk_usd = os.path.join(retarget_dir, "data/rigs/Human/assets/walk.skelanim.usd")
    dump_stage(skel_usd, "human_skeleton.usd")
    dump_stage(walk_usd, "walk.skelanim.usd")

    # Also list all files in that directory
    anim_dir = os.path.join(retarget_dir, "data/rigs/Human/assets")
    if os.path.isdir(anim_dir):
        w(f"Files in {anim_dir}:")
        for f in sorted(os.listdir(anim_dir)):
            w(f"  {f}")
        w()
else:
    w("omni.anim.retarget.core NOT FOUND in extscache")
    w()

# ── Check omni.anim.people ────────────────────────────────────────────────────
w("=" * 60)
w("  omni.anim.people check")
w("=" * 60)
if people_dir:
    w(f"  Found: {people_dir}")
    # Try to list scripts/config inside
    for root, dirs, files in os.walk(people_dir):
        rel = os.path.relpath(root, people_dir)
        for fn in files:
            if fn.endswith((".py", ".usd", ".json", ".yaml")):
                w(f"  {os.path.join(rel, fn)}")
    w()
    # Try importing
    try:
        import omni.anim.people
        w("  import omni.anim.people: SUCCESS")
    except Exception as e:
        w(f"  import omni.anim.people: FAILED — {e}")
else:
    w("  omni.anim.people NOT FOUND in extscache")
w()

# ── Write results ─────────────────────────────────────────────────────────────
os.makedirs(os.path.dirname(OUT_PATH), exist_ok=True)
with open(OUT_PATH, "w") as f:
    f.write("\n".join(lines))
    f.flush()

print(f"[INFO] Written to {OUT_PATH}", flush=True)
app.close()
