# Nova Carter Robot Specifications

## Overview

NVIDIA's flagship autonomous mobile robot (AMR) built on the Segway RMP Lite 220 platform. Uses real differential-drive wheel physics with passive casters.

**Source**: [Segway Nova Carter specs](https://robotics.segway.com/nova-carter/)
**USD Asset**: `Isaac/Robots/NVIDIA/NovaCarter/nova_carter.usd`

---

## Physical Dimensions

| Parameter | Value |
|-----------|-------|
| Overall length | 0.722 m |
| Overall width | 0.500 m |
| Overall height | 0.556 m |
| Net weight | 51.2 kg |
| Drive type | Differential drive (2 powered wheels + passive casters) |

---

## Drive System

| Parameter | Value |
|-----------|-------|
| Wheel radius | 0.14 m |
| Wheel base | 0.4 m |
| Caster wheel radius | 0.0755 m |
| Max linear velocity | 0.5 m/s (training limit) |
| Max angular velocity | 1.0 rad/s (training limit) |

Differential drive kinematics:
```
v_left  = (v - omega * wheel_base/2) / wheel_radius
v_right = (v + omega * wheel_base/2) / wheel_radius
```

---

## Rigid Body Structure (USD)

The robot has **8 rigid body links** and **7 joints**:

```
chassis_link                        (52.91 kg) — entire robot shell
├── wheel_left                      (1.0 kg)   — left drive wheel
├── wheel_right                     (1.0 kg)   — right drive wheel
├── nova_carter_caster_frame_base   (1.688 kg) — caster suspension frame
│   ├── caster_swivel_left          (0.346 kg) — left caster pivot
│   │   └── caster_wheel_left       (0.1 kg)   — left caster wheel
│   └── caster_swivel_right         (0.346 kg) — right caster pivot
│       └── caster_wheel_right      (0.1 kg)   — right caster wheel
└── sensors/                                    — rigidly attached to chassis
    ├── front_hawk/                             — front stereo camera
    ├── back_hawk/                              — back stereo camera
    ├── left_hawk/                              — left stereo camera
    ├── right_hawk/                             — right stereo camera
    ├── XT_32/                                  — 3D LiDAR (Hesai XT-32)
    ├── front_2d_lidar/                         — front 2D LiDAR (RPLidar S2E)
    └── back_2d_lidar/                          — rear 2D LiDAR (RPLidar S2E)
```

### Joints

| Joint | Type | Parent -> Child |
|-------|------|----------------|
| `joint_wheel_left` | continuous | chassis_link -> wheel_left |
| `joint_wheel_right` | continuous | chassis_link -> wheel_right |
| `joint_caster_base` | revolute (+/-2 deg) | chassis_link -> caster_frame_base |
| `joint_swing_left` | continuous | caster_frame_base -> caster_swivel_left |
| `joint_swing_right` | continuous | caster_frame_base -> caster_swivel_right |
| `joint_caster_left` | continuous | caster_swivel_left -> caster_wheel_left |
| `joint_caster_right` | continuous | caster_swivel_right -> caster_wheel_right |

### What `chassis_link` covers

`chassis_link` is the **entire robot body** — not just the center platform. It includes:
- The Segway RMP Lite 220 base platform
- The bumper skirt surrounding the base
- The sensor arch/tower (vertical structure with camera and LiDAR mounts)
- All sensor housings (rigidly attached as children)
- The compute enclosure (Jetson AGX Orin housing)

The only parts NOT on `chassis_link` are the wheels and casters (separate rigid bodies). The chassis skirt covers most of the perimeter, so `chassis_link` contact detection covers ~95% of the robot's collision surface.

---

## Collision Geometry

### LiDAR-to-bumper offsets

The LiDAR sensor is mounted at `chassis_link` center, 0.3m above ground. The robot body extends beyond the LiDAR origin in all directions:

| Direction | Distance from chassis center | What it means |
|-----------|----------------------------|---------------|
| Front | 0.36 m | Front bumper contact |
| Back | 0.36 m | Rear bumper contact |
| Sides | 0.25 m | Side panel contact |

A LiDAR reading of **0.36m at the front = physical contact**. Readings below 0.36m mean the obstacle is inside the robot body.

### Implications for training

- **LiDAR-based obstacle avoidance** (`obstacle_proximity`): threshold must be > 0.36m to start avoiding before physical contact
- **Contact sensors** on `chassis_link` detect actual physics collisions regardless of direction — no offset math needed
- **Goal reached threshold**: should be >= 0.36m (robot's half-length) to avoid requiring the robot to overlap with the goal point

---

## Sensors (on real robot)

| Sensor | Count | Location |
|--------|-------|----------|
| Hesai XT-32 3D LiDAR | 1 | Top of sensor tower |
| RPLidar S2E 2D LiDAR | 2 | Front and rear |
| Owl (Hawk) Stereo Camera | 4 | Front, back, left, right |
| IMU | 1 | Internal |

### Training sensor setup (simplified)

In our Isaac Lab environment, we use a single 2D LiDAR (20 rays, 360-degree, 10m range) mounted on `chassis_link` at 0.3m height. This is a simplified approximation of the real dual-2D-LiDAR setup.
