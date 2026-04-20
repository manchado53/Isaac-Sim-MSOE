"""Print stage bounding box to find building Z range.

exec(open('/home/ad.msoe.edu/somed/isaac-sim-msoe/simulations/crowd/check_bounds.py').read())
"""
import omni.usd
from pxr import UsdGeom

stage = omni.usd.get_context().get_stage()
bbox_cache = UsdGeom.BBoxCache(0, [UsdGeom.Tokens.default_])
world = stage.GetPseudoRoot()

total_range = None
for prim in stage.Traverse():
    if prim.IsA(UsdGeom.Mesh) or prim.IsA(UsdGeom.Xform):
        try:
            b = bbox_cache.ComputeWorldBound(prim)
            r = b.GetRange()
            if r.IsEmpty():
                continue
            if total_range is None:
                total_range = r
            else:
                total_range = total_range.UnionWith(r)
        except:
            pass

if total_range:
    mn = total_range.GetMin()
    mx = total_range.GetMax()
    print(f"X: {mn[0]:.1f} to {mx[0]:.1f}")
    print(f"Y: {mn[1]:.1f} to {mx[1]:.1f}")
    print(f"Z: {mn[2]:.1f} to {mx[2]:.1f}  <-- floor to ceiling")
else:
    print("No geometry found")
