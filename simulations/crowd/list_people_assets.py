"""List available People/Character assets from Isaac Sim Nucleus.

Writes results to videos/people_assets.txt.
Run: sbatch simulations/crowd/submit_list_people.sh
"""
import os, sys
os.environ["OMNI_KIT_ACCEPT_EULA"] = "YES"
os.environ["PRIVACY_CONSENT"] = "Y"

from isaacsim import SimulationApp
app = SimulationApp({"headless": True})

from isaacsim.storage.native import get_assets_root_path
import carb

PROJECT_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), "../.."))
OUT_PATH    = os.path.join(PROJECT_DIR, "videos/people_assets.txt")

assets_root = get_assets_root_path()

lines = []
lines.append(f"Assets root: {assets_root}")
lines.append("")

if assets_root is None:
    lines.append("ERROR: get_assets_root_path() returned None")
else:
    # Try to list people-related paths
    import omni.client

    def list_path(path, label):
        lines.append(f"=== {label} ===")
        lines.append(f"    {path}")
        result, entries = omni.client.list(path)
        if str(result) == "Result.OK":
            for e in entries:
                lines.append(f"  {e.relative_path}  ({'folder' if e.flags & omni.client.ItemFlags.CAN_HAVE_CHILDREN else 'file'})")
        else:
            lines.append(f"  FAILED: {result}")
        lines.append("")

    list_path(assets_root + "/Isaac/People",              "Isaac/People")
    list_path(assets_root + "/Isaac/People/Characters",   "Isaac/People/Characters")
    list_path(assets_root + "/Characters",                "Characters (top-level)")

os.makedirs(os.path.dirname(OUT_PATH), exist_ok=True)
with open(OUT_PATH, "w") as f:
    f.write("\n".join(lines))
    f.flush()

print(f"[INFO] Written to {OUT_PATH}", flush=True)
app.close()
