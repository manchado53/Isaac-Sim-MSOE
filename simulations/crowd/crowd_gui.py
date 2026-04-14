"""Diagnostic: dump everything inside the loaded character prim.
exec(open('/home/ad.msoe.edu/somed/isaac-sim-msoe/simulations/crowd/crowd_gui.py').read())
"""
import omni.usd
stage = omni.usd.get_context().get_stage()

# Dump all prims under /World/Characters
out = []
for prim in stage.Traverse():
    p = str(prim.GetPath())
    if p.startswith("/World/Characters"):
        depth = p.count("/") - 2
        t = prim.GetTypeName()
        apis = [str(a) for a in prim.GetAppliedSchemas()]
        attrs = [a.GetName() for a in prim.GetAttributes()]
        line = f"{'  '*depth}{p}  [{t}]"
        if apis:
            line += f"  schemas={apis}"
        if any(k in str(attrs) for k in ["anim","graph","script","skel"]):
            line += f"  attrs={[a for a in attrs if any(k in a for k in ['anim','graph','script','skel'])]}"
        out.append(line)

result = "\n".join(out)
print(result)

with open("/home/ad.msoe.edu/somed/isaac-sim-msoe/videos/char_dump.txt", "w") as f:
    f.write(result)
print("\nSaved to videos/char_dump.txt")
