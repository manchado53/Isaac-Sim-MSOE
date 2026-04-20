"""Print Building prim bounding box only (excludes NavMeshVolume inflation).

exec(open('/home/ad.msoe.edu/somed/isaac-sim-msoe/simulations/crowd/check_bounds.py').read())
"""
import omni.usd
from pxr import UsdGeom

stage = omni.usd.get_context().get_stage()
bbox_cache = UsdGeom.BBoxCache(0, [UsdGeom.Tokens.default_])

building = stage.GetPrimAtPath("/World/Building")
b = bbox_cache.ComputeWorldBound(building)
r = b.GetRange()
mn, mx = r.GetMin(), r.GetMax()
cx = (mn[0]+mx[0])/2
cy = (mn[1]+mx[1])/2
print(f"X: {mn[0]:.1f} to {mx[0]:.1f}  center={cx:.1f}")
print(f"Y: {mn[1]:.1f} to {mx[1]:.1f}  center={cy:.1f}")
print(f"Z: {mn[2]:.1f} to {mx[2]:.1f}")
print(f"\ncenter_x={cx:.1f}, center_y={cy:.1f}")
