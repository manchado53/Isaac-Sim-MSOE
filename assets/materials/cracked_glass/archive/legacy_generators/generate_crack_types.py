#!/usr/bin/env python3
"""
Generate normal-map textures for 5 realistic glass crack types.

Each type uses the generation method best suited to its geometry:
  1. Impact    — PIL radial spokes + concentric rings
  2. Stress    — Perlin-noise-guided meandering path from edge
  3. Edge      — PIL zigzag from border inward
  4. Thermal   — PIL straight line from edge at 90° + 1-2 branches
  5. Tempered  — Voronoi diagram for dice/cube fragment pattern

All output as black-on-white opacity PNGs + Sobel-derived normal maps,
ready to bind to OmniGlass via the existing pipeline.

Usage:
    ~/isaacsim/python.sh assets/materials/cracked_glass/generate_crack_types.py
    ~/isaacsim/python.sh assets/materials/cracked_glass/generate_crack_types.py --types impact,stress
    ~/isaacsim/python.sh assets/materials/cracked_glass/generate_crack_types.py --size 1024 --seed 42
"""

import argparse
import math
import os
import random
import sys

import numpy as np
from PIL import Image, ImageDraw, ImageFilter

_HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, _HERE)
from generate_textures import opacity_to_normal


def parse_args():
    p = argparse.ArgumentParser(description="Generate 5 glass crack types")
    p.add_argument("--types", default="impact,stress,edge,thermal,tempered",
                   help="Comma-separated crack types to generate")
    p.add_argument("--size", type=int, default=2048)
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--out-dir", default=_HERE)
    return p.parse_args()


def _jagged_line(draw, x0, y0, x1, y1, width=10, jitter=20, segments=16, color=0):
    """Draw a jagged line between two points."""
    pts = []
    for i in range(segments + 1):
        t = i / segments
        x = x0 + (x1 - x0) * t
        y = y0 + (y1 - y0) * t
        if 0 < i < segments:
            x += random.uniform(-jitter, jitter)
            y += random.uniform(-jitter, jitter)
        pts.append((x, y))
    for p, q in zip(pts, pts[1:]):
        draw.line([p, q], fill=color, width=width)


# =========================================================================
# TYPE 1: IMPACT — radial star from central hit point
# =========================================================================

def generate_impact(size, seed):
    """Radial spokes + concentric rings from a central impact point."""
    random.seed(seed)
    img = Image.new("L", (size, size), 255)
    draw = ImageDraw.Draw(img)

    cx = size // 2 + random.randint(-size // 8, size // 8)
    cy = size // 2 + random.randint(-size // 8, size // 8)

    # Radial spokes
    n_spokes = random.randint(8, 14)
    for i in range(n_spokes):
        angle = 2 * math.pi * i / n_spokes + random.uniform(-0.15, 0.15)
        length = size * random.uniform(0.35, 0.48)
        ex = cx + math.cos(angle) * length
        ey = cy + math.sin(angle) * length
        _jagged_line(draw, cx, cy, ex, ey,
                     width=random.randint(8, 14), jitter=25, segments=18)

    # Concentric rings
    for r_frac in [0.06, 0.15, 0.28, 0.40]:
        r = size * r_frac + random.uniform(-15, 15)
        pts = []
        for s in range(50):
            a = 2 * math.pi * s / 50
            rr = r + random.uniform(-12, 12)
            pts.append((cx + math.cos(a) * rr, cy + math.sin(a) * rr))
        pts.append(pts[0])
        draw.line(pts, fill=0, width=random.randint(5, 9))

    # Impact center
    r_center = random.randint(25, 45)
    draw.ellipse((cx - r_center, cy - r_center, cx + r_center, cy + r_center), fill=0)

    return img.filter(ImageFilter.GaussianBlur(1.5))


# =========================================================================
# TYPE 2: STRESS — single meandering line from edge, no branching
# =========================================================================

def _perlin_1d(n, octaves=4, persistence=0.5, seed=0):
    """Simple 1D Perlin-like noise via octave summation of random interpolation."""
    rng = np.random.RandomState(seed)
    result = np.zeros(n)
    for o in range(octaves):
        freq = 2 ** o
        amp = persistence ** o
        phase = rng.uniform(0, 2 * math.pi)
        x = np.linspace(0, freq * 2 * math.pi + phase, n)
        result += amp * np.sin(x + rng.uniform(-math.pi, math.pi))
    return result


def generate_stress(size, seed):
    """Single meandering crack from a random edge, wandering toward the opposite side."""
    random.seed(seed)
    np.random.seed(seed)
    img = Image.new("L", (size, size), 255)
    draw = ImageDraw.Draw(img)

    # Pick a random edge and starting point
    edge = random.choice(["top", "bottom", "left", "right"])
    margin = size * 0.05
    if edge == "top":
        x0, y0 = random.uniform(margin, size - margin), 0
        angle = math.pi / 2  # go downward
    elif edge == "bottom":
        x0, y0 = random.uniform(margin, size - margin), size
        angle = -math.pi / 2
    elif edge == "left":
        x0, y0 = 0, random.uniform(margin, size - margin)
        angle = 0
    else:
        x0, y0 = size, random.uniform(margin, size - margin)
        angle = math.pi

    # Generate meandering path using Perlin-like noise for the direction offset
    n_steps = 200
    step_len = size * 0.004
    noise = _perlin_1d(n_steps, octaves=5, persistence=0.6, seed=seed)
    noise = noise / (np.abs(noise).max() + 1e-6) * 0.4  # normalize to ±0.4 radians

    pts = [(x0, y0)]
    x, y = x0, y0
    for i in range(n_steps):
        a = angle + noise[i]
        x += math.cos(a) * step_len
        y += math.sin(a) * step_len
        # Stop if we've left the image
        if x < -50 or x > size + 50 or y < -50 or y > size + 50:
            break
        pts.append((x, y))

    # Draw the path
    for p, q in zip(pts, pts[1:]):
        draw.line([p, q], fill=0, width=random.randint(7, 10))

    return img.filter(ImageFilter.GaussianBlur(2.0))


# =========================================================================
# TYPE 3: EDGE — zigzag from border spreading inward
# =========================================================================

def generate_edge(size, seed):
    """Zigzag crack pattern starting at the glass edge, spreading inward."""
    random.seed(seed)
    img = Image.new("L", (size, size), 255)
    draw = ImageDraw.Draw(img)

    # Pick edge side
    edge = random.choice(["top", "bottom", "left", "right"])

    # Generate 2-4 zigzag lines from the edge
    n_lines = random.randint(2, 4)
    for _ in range(n_lines):
        if edge == "top":
            x0 = random.uniform(size * 0.1, size * 0.9)
            y0 = random.uniform(0, size * 0.03)
            dx, dy = random.uniform(-0.3, 0.3), 1.0
        elif edge == "bottom":
            x0 = random.uniform(size * 0.1, size * 0.9)
            y0 = random.uniform(size * 0.97, size)
            dx, dy = random.uniform(-0.3, 0.3), -1.0
        elif edge == "left":
            x0 = random.uniform(0, size * 0.03)
            y0 = random.uniform(size * 0.1, size * 0.9)
            dx, dy = 1.0, random.uniform(-0.3, 0.3)
        else:
            x0 = random.uniform(size * 0.97, size)
            y0 = random.uniform(size * 0.1, size * 0.9)
            dx, dy = -1.0, random.uniform(-0.3, 0.3)

        # Zigzag path
        length = size * random.uniform(0.2, 0.45)
        _jagged_line(draw, x0, y0,
                     x0 + dx * length, y0 + dy * length,
                     width=random.randint(8, 12),
                     jitter=random.randint(30, 50),
                     segments=random.randint(18, 28))

    # Small "oyster" chip fragments at the edge origin
    for _ in range(random.randint(3, 8)):
        if edge == "top":
            cx, cy = random.uniform(size * 0.2, size * 0.8), random.randint(0, 40)
        elif edge == "bottom":
            cx, cy = random.uniform(size * 0.2, size * 0.8), random.randint(size - 40, size)
        elif edge == "left":
            cx, cy = random.randint(0, 40), random.uniform(size * 0.2, size * 0.8)
        else:
            cx, cy = random.randint(size - 40, size), random.uniform(size * 0.2, size * 0.8)
        r = random.randint(15, 45)
        draw.ellipse((cx - r, cy - r, cx + r, cy + r), fill=0)

    return img.filter(ImageFilter.GaussianBlur(1.5))


# =========================================================================
# TYPE 4: THERMAL — straight from edge at 90°, then 1-2 branches
# =========================================================================

def generate_thermal(size, seed):
    """Crack starting perpendicular to edge, straight initially, then branching."""
    random.seed(seed)
    img = Image.new("L", (size, size), 255)
    draw = ImageDraw.Draw(img)

    # Pick edge
    edge = random.choice(["top", "bottom", "left", "right"])
    if edge == "top":
        x0 = random.uniform(size * 0.2, size * 0.8)
        y0 = 0
        angle = math.pi / 2  # perpendicular = straight down
    elif edge == "bottom":
        x0 = random.uniform(size * 0.2, size * 0.8)
        y0 = size
        angle = -math.pi / 2
    elif edge == "left":
        x0 = 0
        y0 = random.uniform(size * 0.2, size * 0.8)
        angle = 0  # perpendicular = straight right
    else:
        x0 = size
        y0 = random.uniform(size * 0.2, size * 0.8)
        angle = math.pi

    # Primary crack — nearly straight, very low jitter
    primary_len = size * random.uniform(0.35, 0.55)
    x1 = x0 + math.cos(angle) * primary_len
    y1 = y0 + math.sin(angle) * primary_len
    _jagged_line(draw, x0, y0, x1, y1,
                 width=random.randint(8, 11), jitter=12, segments=20)

    # Branch point — 60-70% along the primary
    branch_t = random.uniform(0.55, 0.75)
    bx = x0 + (x1 - x0) * branch_t
    by = y0 + (y1 - y0) * branch_t

    # 1-2 branches at 25-45° from primary direction
    n_branches = random.randint(1, 2)
    for _ in range(n_branches):
        branch_angle = angle + random.choice([-1, 1]) * random.uniform(
            math.radians(25), math.radians(45))
        branch_len = size * random.uniform(0.12, 0.25)
        bx1 = bx + math.cos(branch_angle) * branch_len
        by1 = by + math.sin(branch_angle) * branch_len
        _jagged_line(draw, bx, by, bx1, by1,
                     width=random.randint(5, 8), jitter=15, segments=12)

    return img.filter(ImageFilter.GaussianBlur(1.5))


# =========================================================================
# TYPE 5: TEMPERED SHATTER — Voronoi dice/cube fragments
# =========================================================================

def generate_tempered(size, seed):
    """Hundreds of small cube-like fragments via Voronoi diagram."""
    from scipy.spatial import Voronoi

    np.random.seed(seed)
    random.seed(seed)
    img = Image.new("L", (size, size), 255)
    draw = ImageDraw.Draw(img)

    # Generate seed points — mostly uniform with slight clustering near center
    n_points = random.randint(250, 400)
    n_center = n_points // 5
    n_uniform = n_points - n_center

    cx, cy = size / 2, size / 2
    pts_uniform = np.column_stack([
        np.random.uniform(0, size, n_uniform),
        np.random.uniform(0, size, n_uniform),
    ])
    pts_center = np.column_stack([
        np.random.normal(cx, size * 0.15, n_center),
        np.random.normal(cy, size * 0.15, n_center),
    ])
    pts = np.vstack([pts_uniform, pts_center])
    pts[:, 0] = np.clip(pts[:, 0], 5, size - 5)
    pts[:, 1] = np.clip(pts[:, 1], 5, size - 5)

    # Add border points to close Voronoi cells at edges
    border = []
    for v in np.linspace(0, size, 30):
        border.extend([(0, v), (size, v), (v, 0), (v, size)])
    pts = np.vstack([pts, np.array(border)])

    vor = Voronoi(pts)

    # Draw Voronoi edges
    line_width = random.randint(3, 5)
    for ridge in vor.ridge_vertices:
        if -1 in ridge:
            continue
        v0 = vor.vertices[ridge[0]]
        v1 = vor.vertices[ridge[1]]
        # Skip edges far outside the image
        if (v0[0] < -50 or v0[0] > size + 50 or v0[1] < -50 or v0[1] > size + 50 or
            v1[0] < -50 or v1[0] > size + 50 or v1[1] < -50 or v1[1] > size + 50):
            continue
        draw.line([(v0[0], v0[1]), (v1[0], v1[1])], fill=0, width=line_width)

    return img.filter(ImageFilter.GaussianBlur(1.0))


# =========================================================================
# Main
# =========================================================================

GENERATORS = {
    "impact":   generate_impact,
    "stress":   generate_stress,
    "edge":     generate_edge,
    "thermal":  generate_thermal,
    "tempered": generate_tempered,
}


def main():
    args = parse_args()
    types = [t.strip() for t in args.types.split(",") if t.strip()]

    for crack_type in types:
        if crack_type not in GENERATORS:
            print(f"[gen] unknown type: {crack_type}, skipping", flush=True)
            continue

        print(f"[gen] generating {crack_type} (seed={args.seed}, size={args.size})",
              flush=True)
        opacity = GENERATORS[crack_type](args.size, args.seed)

        opacity_path = os.path.join(args.out_dir, f"crack_opacity_{crack_type}.png")
        opacity.save(opacity_path)
        print(f"  saved {os.path.basename(opacity_path)}", flush=True)

        normal = opacity_to_normal(opacity)
        normal_path = os.path.join(args.out_dir, f"crack_normal_{crack_type}.png")
        normal.save(normal_path)
        print(f"  saved {os.path.basename(normal_path)}", flush=True)

    print("[gen] done", flush=True)


if __name__ == "__main__":
    main()
