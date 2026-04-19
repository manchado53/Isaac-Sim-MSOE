"""Rebake water with subtle waves — no faceting artifacts.

Run from Isaac Sim Script Editor:
  exec(open('/home/ad.msoe.edu/somed/isaac-sim-msoe/simulations/crowd/calm_waves.py').read())
"""
import omni.usd, math
from pxr import UsdGeom, Gf, Vt

stage = omni.usd.get_context().get_stage()

FPS=30; RISE_FRAMES=600; HOLD_FRAMES=300; TOTAL_FRAMES=900
FLOOR_Z=26.9; FLOOD_RISE=1.5
WAVE_AMP=0.08; WAVE_FREQ=0.08; WAVE_SPEED=1.2
N=10; half=60.0

base_xy=[(-half+2*half*i/N, -half+2*half*j/N) for j in range(N+1) for i in range(N+1)]

mesh = UsdGeom.Mesh(stage.GetPrimAtPath("/World/WaterPlane"))
xf = UsdGeom.Xformable(mesh.GetPrim())
xf.ClearXformOpOrder()
t_op = xf.AddTranslateOp()

print("Baking calm waves...")
for frame in range(TOTAL_FRAMES+1):
    frac = min(frame/RISE_FRAMES, 1.0)
    t_op.Set(Gf.Vec3d(-20.0, 65.0, FLOOR_Z + frac*FLOOD_RISE), frame)
    time_rad = frame/FPS*WAVE_SPEED
    pts = []
    for (x,y) in base_xy:
        dz = WAVE_AMP*math.sin(WAVE_FREQ*x+time_rad)*math.cos(WAVE_FREQ*y*0.7+time_rad*0.8)
        pts.append(Gf.Vec3f(float(x), float(y), float(dz)))
    mesh.GetPointsAttr().Set(Vt.Vec3fArray(pts), frame)
    if frame % 150 == 0:
        print(f"  {frame}/{TOTAL_FRAMES}")
print("Done — subtle 8cm waves baked.")
