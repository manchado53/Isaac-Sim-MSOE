"""Check what omni.anim.graph.ui actually exposes and find animation graph API.
exec(open('/home/ad.msoe.edu/somed/isaac-sim-msoe/simulations/crowd/crowd_gui.py').read())
"""
import omni.usd
import omni.kit.app
stage = omni.usd.get_context().get_stage()
app = omni.kit.app.get_app()

out = []

# Check what anim graph extensions are loaded
ext_mgr = app.get_extension_manager()
for ext in ["omni.anim.graph.core", "omni.anim.graph.ui", "omni.anim.graph.schema"]:
    enabled = ext_mgr.is_extension_enabled(ext)
    out.append(f"{ext}: enabled={enabled}")

# Try importing core
try:
    import omni.anim.graph.core as ag
    out.append(f"omni.anim.graph.core imported OK")
    out.append(f"  dir: {[x for x in dir(ag) if not x.startswith('_')]}")
except Exception as e:
    out.append(f"omni.anim.graph.core import FAILED: {e}")

# Try importing schema
try:
    import omni.anim.graph.schema as ags
    out.append(f"omni.anim.graph.schema imported OK")
    out.append(f"  dir: {[x for x in dir(ags) if not x.startswith('_')]}")
except Exception as e:
    out.append(f"omni.anim.graph.schema import FAILED: {e}")

# Try to create an AnimationGraph prim directly
char_path = "/female_adult_police_01_new"
graph_path = char_path + "/AnimGraph_Test"
try:
    prim = stage.DefinePrim(graph_path, "AnimationGraph")
    out.append(f"DefinePrim AnimationGraph: valid={prim.IsValid()} type={prim.GetTypeName()}")
except Exception as e:
    out.append(f"DefinePrim AnimationGraph FAILED: {e}")

result = "\n".join(out)
print(result)
with open("/home/ad.msoe.edu/somed/isaac-sim-msoe/videos/animgraph_check.txt", "w") as f:
    f.write(result)
print("Saved to videos/animgraph_check.txt")
