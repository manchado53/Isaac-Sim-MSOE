"""Animated LiDAR diagram with goal navigation for presentation."""

import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as patches
import matplotlib.animation as animation

ROOM = 10.0
OBSTACLES = [
    (-5, 4, 2.5, 1.2),
    (4, 3, 2.0, 1.0),
    (3, -5, 1.0, 1.8),
    (-4, -4, 3.0, 1.2),
]
GOAL_CLEAR_RADIUS = 2.0  # min distance from obstacle center to goal

def is_goal_valid(gx, gy):
    if abs(gx) > ROOM - 1 or abs(gy) > ROOM - 1:
        return False
    for (ox, oy, ow, oh) in OBSTACLES:
        if np.hypot(gx - ox, gy - oy) < GOAL_CLEAR_RADIUS + max(ow, oh) / 2:
            return False
    return True

def random_goal(rng):
    while True:
        gx = rng.uniform(-ROOM + 1.5, ROOM - 1.5)
        gy = rng.uniform(-ROOM + 1.5, ROOM - 1.5)
        if is_goal_valid(gx, gy):
            return gx, gy

rng = np.random.default_rng(42)
GOALS = [random_goal(rng) for _ in range(6)]
NUM_RAYS = 20
MAX_RANGE = 10.0
FPS = 30
DURATION = 12
ROBOT_SPEED = 0.07   # units per frame
TURN_SPEED = 0.08


def cast_ray(rx, ry, angle):
    dx, dy = np.cos(angle), np.sin(angle)
    t = MAX_RANGE

    for wall_t in [
        ((-ROOM - rx) / dx) if abs(dx) > 1e-6 else np.inf,
        ((ROOM - rx) / dx) if abs(dx) > 1e-6 else np.inf,
        ((-ROOM - ry) / dy) if abs(dy) > 1e-6 else np.inf,
        ((ROOM - ry) / dy) if abs(dy) > 1e-6 else np.inf,
    ]:
        if 0.1 < wall_t < t:
            t = wall_t

    for (ox, oy, ow, oh) in OBSTACLES:
        x1, x2 = ox - ow / 2, ox + ow / 2
        y1, y2 = oy - oh / 2, oy + oh / 2
        for sx in [x1, x2]:
            if abs(dx) > 1e-6:
                st = (sx - rx) / dx
                hy = ry + st * dy
                if 0.1 < st < t and y1 <= hy <= y2:
                    t = st
        for sy in [y1, y2]:
            if abs(dy) > 1e-6:
                st = (sy - ry) / dy
                hx = rx + st * dx
                if 0.1 < st < t and x1 <= hx <= x2:
                    t = st

    return rx + t * dx, ry + t * dy


# Pre-compute robot path navigating between goals
total_frames = FPS * DURATION
rx_path = np.zeros(total_frames)
ry_path = np.zeros(total_frames)
yaw_path = np.zeros(total_frames)
goal_idx_path = np.zeros(total_frames, dtype=int)

rx, ry, yaw = 0.0, 0.0, 0.0
goal_idx = 0
gx, gy = GOALS[goal_idx]

for f in range(total_frames):
    rx_path[f] = rx
    ry_path[f] = ry
    yaw_path[f] = yaw
    goal_idx_path[f] = goal_idx

    # Attraction toward goal
    ax_force = gx - rx
    ay_force = gy - ry
    dist_goal = np.hypot(ax_force, ay_force)
    if dist_goal > 0:
        ax_force /= dist_goal
        ay_force /= dist_goal

    # Repulsion from obstacles (potential field)
    rep_x, rep_y = 0.0, 0.0
    rep_radius = 2.5
    for (ox, oy, ow, oh) in OBSTACLES:
        cx, cy = ox, oy
        dx, dy = rx - cx, ry - cy
        d = np.hypot(dx, dy)
        if 0 < d < rep_radius:
            strength = (1.0 / d - 1.0 / rep_radius) * (1.0 / d ** 2)
            rep_x += strength * dx / d
            rep_y += strength * dy / d

    # Combined steering
    fx = ax_force + rep_x * 3.0
    fy = ay_force + rep_y * 3.0
    desired_yaw = np.arctan2(fy, fx)
    yaw_err = (desired_yaw - yaw + np.pi) % (2 * np.pi) - np.pi
    yaw += np.clip(yaw_err, -TURN_SPEED, TURN_SPEED)

    rx += np.cos(yaw) * ROBOT_SPEED
    ry += np.sin(yaw) * ROBOT_SPEED

    rx = np.clip(rx, -ROOM + 1, ROOM - 1)
    ry = np.clip(ry, -ROOM + 1, ROOM - 1)

    if np.hypot(gx - rx, gy - ry) < 0.5:
        goal_idx = (goal_idx + 1) % len(GOALS)
        gx, gy = GOALS[goal_idx]


# ── Build figure ──
fig, ax = plt.subplots(figsize=(7, 7), facecolor='#1a1a1a')
ax.set_facecolor('#1a1a1a')
ax.set_xlim(-11.5, 11.5)
ax.set_ylim(-11.5, 11.5)
ax.set_aspect('equal')
ax.axis('off')
ax.set_title('20-ray 360° LiDAR Scan  |  Max Range = 10m',
             color='white', fontsize=13, fontweight='bold', pad=10)

# Walls
ax.plot([-ROOM, ROOM, ROOM, -ROOM, -ROOM],
        [-ROOM, -ROOM, ROOM, ROOM, -ROOM], color='#888888', linewidth=2)

# Obstacles
for (ox, oy, ow, oh) in OBSTACLES:
    rect = patches.FancyBboxPatch(
        (ox - ow / 2, oy - oh / 2), ow, oh,
        boxstyle="round,pad=0.05",
        linewidth=1.5, edgecolor='#aaaaaa', facecolor='#4a3728',
    )
    ax.add_patch(rect)

# Dynamic elements
ray_lines = [ax.plot([], [], color='#00ff88', linewidth=1.0, alpha=0.7)[0] for _ in range(NUM_RAYS)]
hit_dots = [ax.plot([], [], 'o', color='#00ff88', markersize=4)[0] for _ in range(NUM_RAYS)]

# Robot body
robot_patch = patches.FancyBboxPatch(
    (-0.48, -0.335), 0.96, 0.67,
    boxstyle="round,pad=0.08",
    linewidth=2, edgecolor='white', facecolor='#444444', zorder=5,
)
ax.add_patch(robot_patch)

# Direction triangle on robot (replaces annotation arrow)
dir_triangle = plt.Polygon([[0, 0], [0, 0], [0, 0]],
                             closed=True, facecolor='white', zorder=6)
ax.add_patch(dir_triangle)

# Goal marker
goal_dot, = ax.plot([], [], '*', color='#ffdd00', markersize=18, zorder=4, label='Goal')
goal_circle = plt.Circle((0, 0), 0.5, color='#ffdd00', fill=False,
                          linewidth=1.5, linestyle='--', zorder=4)
ax.add_patch(goal_circle)

# Goal line (dashed line from robot to goal)
goal_line, = ax.plot([], [], '--', color='#ffdd0055', linewidth=1.0, zorder=3)

# Legend
ax.legend(
    handles=[
        plt.Line2D([0], [0], color='#00ff88', lw=1.5, label='LiDAR ray'),
        plt.Line2D([0], [0], color='#00ff88', lw=0, marker='o', markersize=5, label='Hit point'),
        plt.Line2D([0], [0], color='#ffdd00', lw=0, marker='*', markersize=10, label='Goal'),
        patches.Patch(facecolor='#4a3728', edgecolor='#aaaaaa', label='Obstacle'),
    ],
    loc='lower right', facecolor='#2a2a2a', labelcolor='white', fontsize=9,
)


def update(frame):
    rx = rx_path[frame]
    ry = ry_path[frame]
    yaw = yaw_path[frame]
    gi = goal_idx_path[frame]
    gx, gy = GOALS[gi]

    # Robot body
    robot_patch.set_x(rx - 0.48)
    robot_patch.set_y(ry - 0.335)

    # Direction triangle
    tip = np.array([rx + np.cos(yaw) * 0.7, ry + np.sin(yaw) * 0.7])
    left = np.array([rx + np.cos(yaw + 2.4) * 0.35, ry + np.sin(yaw + 2.4) * 0.35])
    right = np.array([rx + np.cos(yaw - 2.4) * 0.35, ry + np.sin(yaw - 2.4) * 0.35])
    dir_triangle.set_xy([tip, left, right])

    # Goal
    goal_dot.set_data([gx], [gy])
    goal_circle.set_center((gx, gy))
    goal_line.set_data([rx, gx], [ry, gy])

    # LiDAR rays
    angles = np.linspace(0, 2 * np.pi, NUM_RAYS, endpoint=False)
    for i, a in enumerate(angles):
        hx, hy = cast_ray(rx, ry, a)
        ray_lines[i].set_data([rx, hx], [ry, hy])
        hit_dots[i].set_data([hx], [hy])

    return ray_lines + hit_dots + [robot_patch, dir_triangle, goal_dot, goal_circle, goal_line]


ani = animation.FuncAnimation(fig, update, frames=total_frames, interval=1000 / FPS, blit=False)
writer = animation.FFMpegWriter(fps=FPS, bitrate=2000)
ani.save('lidar_animation.mp4', writer=writer, dpi=120, savefig_kwargs={'facecolor': '#1a1a1a'})
print("Saved: lidar_animation.mp4")
plt.close()
