#!/usr/bin/env python3
"""
Procedurally author the crack normal/opacity textures used by the
cracked-glass dataset pipeline.

Writes six PNGs into the containing directory:

    crack_normal_v1.png / crack_opacity_v1.png   -- radial "impact point" crack
    crack_normal_v2.png / crack_opacity_v2.png   -- spider-web shatter
    crack_normal_v3.png / crack_opacity_v3.png   -- edge chipping / partial breakage

Opacity masks: black = crack (punched through), white = intact glass.
Normal maps: grayscale height → XYZ-encoded normals with a simple Sobel filter.
Tileable at the edges for larger glass panels.

Requires only PIL + numpy. Run on the login node; no GPU needed:

    python3 assets/materials/cracked_glass/generate_textures.py
"""

from __future__ import annotations

import math
import os
import random

import numpy as np
from PIL import Image, ImageDraw, ImageFilter


TEX_SIZE = 2048
OUT_DIR = os.path.dirname(os.path.abspath(__file__))


def _draw_jagged_line(draw: ImageDraw.ImageDraw, x0, y0, x1, y1,
                      width=14, jitter=20, segments=14, color=0):
    """Draw a line from (x0,y0) to (x1,y1) with per-segment jitter."""
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


def _draw_radial_cracks(img: Image.Image, cx, cy, n_spokes, length, jitter=28):
    """Draw radial crack lines emanating from (cx, cy)."""
    draw = ImageDraw.Draw(img)
    for i in range(n_spokes):
        angle = 2 * math.pi * i / n_spokes + random.uniform(-0.2, 0.2)
        lx = cx + math.cos(angle) * length
        ly = cy + math.sin(angle) * length
        _draw_jagged_line(draw, cx, cy, lx, ly,
                          width=random.randint(8, 16), jitter=jitter,
                          segments=16, color=0)


def _draw_rings(img: Image.Image, cx, cy, radii):
    """Draw concentric near-circular cracks."""
    draw = ImageDraw.Draw(img)
    for r in radii:
        # Approximate circle with jagged polyline so it matches the spoke style.
        pts = []
        segs = 40
        for s in range(segs + 1):
            a = 2 * math.pi * s / segs
            rr = r + random.uniform(-12, 12)
            pts.append((cx + math.cos(a) * rr, cy + math.sin(a) * rr))
        draw.line(pts, fill=0, width=random.randint(6, 10))


def _draw_edge_chips(img: Image.Image):
    """Random chipped / missing spots along the panel edge."""
    draw = ImageDraw.Draw(img)
    w, h = img.size
    for _ in range(18):
        side = random.randint(0, 3)
        if side == 0:   # top
            x, y = random.randint(0, w), random.randint(0, 60)
        elif side == 1: # bottom
            x, y = random.randint(0, w), random.randint(h - 60, h)
        elif side == 2: # left
            x, y = random.randint(0, 60), random.randint(0, h)
        else:           # right
            x, y = random.randint(w - 60, w), random.randint(0, h)
        r = random.randint(30, 110)
        draw.ellipse((x - r, y - r, x + r, y + r), fill=0)


def _draw_web(img: Image.Image, cx, cy):
    """Spider-web: radial spokes + rings + cross-connections."""
    _draw_radial_cracks(img, cx, cy, n_spokes=12, length=TEX_SIZE * 0.48, jitter=22)
    _draw_rings(img, cx, cy, radii=[TEX_SIZE * r for r in (0.08, 0.18, 0.30, 0.42)])
    # A handful of branching offshoots
    draw = ImageDraw.Draw(img)
    for _ in range(20):
        angle = random.uniform(0, 2 * math.pi)
        r1 = random.uniform(TEX_SIZE * 0.15, TEX_SIZE * 0.40)
        r2 = r1 + random.uniform(80, 250)
        x0 = cx + math.cos(angle) * r1
        y0 = cy + math.sin(angle) * r1
        x1 = cx + math.cos(angle + random.uniform(-0.6, 0.6)) * r2
        y1 = cy + math.sin(angle + random.uniform(-0.6, 0.6)) * r2
        _draw_jagged_line(draw, x0, y0, x1, y1,
                          width=random.randint(5, 9), jitter=14,
                          segments=8, color=0)


# ----------------------------------------------------------------------------
# Normal map generation
# ----------------------------------------------------------------------------

def height_to_normal(height: np.ndarray, strength: float = 4.0) -> np.ndarray:
    """Convert a grayscale height map (0..1) to an RGB normal map.

    Uses a simple Sobel gradient. Output is in DirectX / Y-down convention
    to match the material's flip_tangent_v setting.
    """
    h = height.astype(np.float32)
    # Sobel gradients
    gx = np.zeros_like(h)
    gy = np.zeros_like(h)
    gx[:, 1:-1] = h[:, 2:] - h[:, :-2]
    gy[1:-1, :] = h[2:, :] - h[:-2, :]
    gx *= strength
    gy *= strength

    nz = np.ones_like(h)
    length = np.sqrt(gx * gx + gy * gy + nz * nz)
    nx = -gx / length
    ny = -gy / length
    nz = nz / length

    rgb = np.zeros(h.shape + (3,), dtype=np.uint8)
    rgb[..., 0] = np.clip((nx * 0.5 + 0.5) * 255, 0, 255).astype(np.uint8)
    rgb[..., 1] = np.clip((ny * 0.5 + 0.5) * 255, 0, 255).astype(np.uint8)
    rgb[..., 2] = np.clip((nz * 0.5 + 0.5) * 255, 0, 255).astype(np.uint8)
    return rgb


def opacity_to_normal(opacity_img: Image.Image, blur_radius: int = 3,
                      strength: float = 18.0) -> Image.Image:
    """Derive a normal map from an opacity / height silhouette.

    Smooth the mask so the cracks get a soft chamfer rather than a hard edge —
    reads as a "ridge" around each crack line in the rendered scene.

    ``blur_radius=0`` keeps edges pixel-crisp (reads as hairline fractures);
    higher blur chamfers the crack into a softer groove. ``strength`` scales
    the normal-gradient magnitude — higher values make the crack read as a
    deeper, more contrast-y scratch through the glass.
    """
    if blur_radius > 0:
        source = opacity_img.filter(ImageFilter.GaussianBlur(blur_radius))
    else:
        source = opacity_img
    arr = np.asarray(source, dtype=np.float32) / 255.0
    # Invert so cracks (dark) are high relief.
    height = 1.0 - arr
    return Image.fromarray(height_to_normal(height, strength=strength))


# ----------------------------------------------------------------------------
# Variant authoring
# ----------------------------------------------------------------------------

def make_variant(name: str, seed: int) -> tuple[Image.Image, Image.Image]:
    """Compose an opacity PNG + derived normal PNG for a single variant."""
    random.seed(seed)
    np.random.seed(seed)

    opacity = Image.new("L", (TEX_SIZE, TEX_SIZE), color=255)

    if name == "v1":
        # Single sharp impact point, strong radial cracks
        cx, cy = TEX_SIZE // 2, TEX_SIZE // 2
        _draw_radial_cracks(opacity, cx, cy, n_spokes=9,
                            length=TEX_SIZE * 0.45, jitter=30)
        _draw_rings(opacity, cx, cy, radii=[TEX_SIZE * r for r in (0.05, 0.14)])
        ImageDraw.Draw(opacity).ellipse(
            (cx - 35, cy - 35, cx + 35, cy + 35), fill=0,
        )

    elif name == "v2":
        # Full shatter web
        cx, cy = TEX_SIZE // 2, TEX_SIZE // 2
        _draw_web(opacity, cx, cy)

    elif name == "v3":
        # Edge chips + a few stray cracks
        _draw_edge_chips(opacity)
        _draw_radial_cracks(opacity, random.randint(400, 1600),
                            random.randint(400, 1600), n_spokes=4,
                            length=TEX_SIZE * 0.25, jitter=18)

    else:
        raise ValueError(f"unknown variant: {name}")

    # Slight blur softens pixelated line endings
    opacity = opacity.filter(ImageFilter.GaussianBlur(1.5))
    normal = opacity_to_normal(opacity)
    return normal, opacity


def main():
    os.makedirs(OUT_DIR, exist_ok=True)
    variants = [("v1", 101), ("v2", 202), ("v3", 303)]
    for name, seed in variants:
        normal, opacity = make_variant(name, seed)
        normal_path = os.path.join(OUT_DIR, f"crack_normal_{name}.png")
        opacity_path = os.path.join(OUT_DIR, f"crack_opacity_{name}.png")
        normal.save(normal_path)
        opacity.save(opacity_path)
        print(f"wrote {normal_path}  ({normal.size})")
        print(f"wrote {opacity_path} ({opacity.size})")


if __name__ == "__main__":
    main()
