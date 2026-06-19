#!/usr/bin/env python3
"""
Generate tempered-glass opacity + normal textures procedurally via Voronoi
tessellation. No photo reference — the dice pattern is synthesized from random
seed points and their Voronoi cells.

Rationale: real tempered glass fractures by radial stress propagation; each
fracture origin "owns" the region closest to it before colliding with another
front. That's literally a Voronoi diagram. Photo-based edge extraction cannot
recover the closed polygons that make the dice pattern read correctly; this
approach produces them by construction.

Seamless tiling:
  Seeds are replicated in a 3x3 grid before the tessellation. The central N
  seeds' cells have boundaries set by their mirrored neighbours, so edges
  near x=W-k in the central tile match edges near x=-k in the right tile.
  When the output PNG is GPU-tiled on a surface, the lines continue across
  tile seams with no visible discontinuity.

Writes:
  crack_opacity_tempered<suffix>.png   (2048x2048)
  crack_normal_tempered<suffix>.png    (2048x2048)

Examples:
  # Medium-density pattern, default looks
  python3 assets/materials/cracked_glass/tempered_voronoi.py

  # Big chunky shards (good for close-up renders)
  python3 tempered_voronoi.py --suffix _v5 --n-seeds 500 --line-width 3

  # Fine dice (closer to real-world tempered density)
  python3 tempered_voronoi.py --suffix _v7 --n-seeds 4000 --line-width 1
"""

from __future__ import annotations

import argparse
import os
import sys

import numpy as np
from PIL import Image, ImageDraw
from scipy.spatial import Voronoi

_HERE = os.path.dirname(os.path.abspath(__file__))
_ASSETS = os.path.abspath(os.path.join(_HERE, "..", "..", "..", "assets", "materials", "cracked_glass"))
sys.path.insert(0, _HERE)
from generate_textures import opacity_to_normal


def parse_args():
    p = argparse.ArgumentParser(description=__doc__.splitlines()[1])
    p.add_argument("--size", type=int, default=2048,
                   help="Output texture side length in px (default 2048)")
    p.add_argument("--n-seeds", type=int, default=1500,
                   help="Seed count across one tile (default 1500). "
                        "~500 = big chunky shards, ~1500 = medium, "
                        "~4000+ = fine dice close to real tempered density")
    p.add_argument("--seed", type=int, default=42,
                   help="RNG seed for reproducibility (default 42)")
    p.add_argument("--line-width", type=int, default=2,
                   help="Crack line thickness in px (default 2)")
    p.add_argument("--edge-jitter", type=float, default=2.0,
                   help="Random midpoint offset in px (default 2.0). "
                        "0 = mathematically-straight Voronoi edges, "
                        "higher = more natural-looking curved crack lines")
    p.add_argument("--normal-blur", type=int, default=2,
                   help="Gaussian blur radius before Sobel in px (default 2)")
    p.add_argument("--suffix", default="_voronoi",
                   help="Output suffix — writes crack_{opacity,normal}_tempered<suffix>.png "
                        "(default '_voronoi')")
    p.add_argument("--debug", action="store_true",
                   help="Also save the raw opacity tile for inspection")
    return p.parse_args()


def tile_seeds_3x3(seeds: np.ndarray, size: int) -> np.ndarray:
    """Replicate seeds in a 3x3 virtual grid around the central [0,size) tile.

    The central tile's cells get their boundaries from neighbours on all sides,
    so the pattern wraps continuously when the output is GPU-tiled.
    """
    tiles = []
    for dx in (-size, 0, size):
        for dy in (-size, 0, size):
            tiles.append(seeds + np.array([dx, dy]))
    return np.vstack(tiles)


def rasterize_voronoi_edges(vor: Voronoi, size: int, line_width: int,
                            edge_jitter: float, rng: np.random.Generator
                            ) -> Image.Image:
    """Draw Voronoi ridges onto a size x size canvas.

    cracks=0 (black), glass=255 (white). PIL clips any line segments that
    extend outside the canvas, which is exactly what we want — edges from
    wrapped seeds pass through the tile boundary naturally.
    """
    img = Image.new("L", (size, size), 255)
    draw = ImageDraw.Draw(img)

    for a, b in vor.ridge_vertices:
        if a < 0 or b < 0:
            continue  # edge to infinity (outer hull of the seed cloud)
        p0 = vor.vertices[a]
        p1 = vor.vertices[b]

        # Skip edges entirely outside the central tile (small optimization —
        # PIL would clip them anyway but the loop cost adds up for large N).
        if (max(p0[0], p1[0]) < -line_width or
                min(p0[0], p1[0]) > size + line_width or
                max(p0[1], p1[1]) < -line_width or
                min(p0[1], p1[1]) > size + line_width):
            continue

        if edge_jitter > 0:
            # Split each ridge at its midpoint, offset that midpoint randomly.
            # Cheap way to make perfectly-straight Voronoi edges look organic.
            mid = (p0 + p1) / 2.0 + rng.normal(0, edge_jitter, 2)
            draw.line([tuple(p0), tuple(mid)], fill=0, width=line_width)
            draw.line([tuple(mid), tuple(p1)], fill=0, width=line_width)
        else:
            draw.line([tuple(p0), tuple(p1)], fill=0, width=line_width)

    return img


def main():
    args = parse_args()

    rng = np.random.default_rng(args.seed)

    print(f"[voronoi] sampling {args.n_seeds} seeds in {args.size}x{args.size}",
          flush=True)
    seeds = rng.uniform(0, args.size, (args.n_seeds, 2))

    tiled = tile_seeds_3x3(seeds, args.size)
    print(f"[voronoi] computing tessellation on {len(tiled)} wrapped seeds",
          flush=True)
    vor = Voronoi(tiled)
    print(f"[voronoi] got {len(vor.ridge_vertices)} ridges, "
          f"{len(vor.vertices)} vertices", flush=True)

    print(f"[voronoi] rasterizing edges (width={args.line_width}, "
          f"jitter={args.edge_jitter})", flush=True)
    opacity = rasterize_voronoi_edges(vor, args.size, args.line_width,
                                      args.edge_jitter, rng)

    crack_pct = 100.0 * np.mean(np.asarray(opacity) == 0)
    print(f"[voronoi] crack coverage: {crack_pct:.1f}%", flush=True)

    out_opacity = os.path.join(_ASSETS, f"crack_opacity_tempered{args.suffix}.png")
    out_normal = os.path.join(_ASSETS, f"crack_normal_tempered{args.suffix}.png")
    out_roughness = os.path.join(_ASSETS, f"crack_roughness_tempered{args.suffix}.png")

    opacity.save(out_opacity)
    print(f"[voronoi] saved {os.path.basename(out_opacity)} ({opacity.size})",
          flush=True)

    # Invert opacity for roughness: cracks become white (rough = frosted
    # scatter in OmniGlass), intact stays black (smooth). Paired with
    # roughness_texture_influence<1 in glass_anomaly.py so intact areas
    # keep the base frosting_roughness rather than dropping to mirror.
    rough = Image.eval(opacity, lambda x: 255 - x)
    rough.save(out_roughness)
    print(f"[voronoi] saved {os.path.basename(out_roughness)} ({rough.size})",
          flush=True)

    if args.debug:
        dbg = os.path.join(_ASSETS, f"_debug_tempered{args.suffix}_raw_opacity.png")
        opacity.save(dbg)

    print(f"[voronoi] deriving normal map (blur={args.normal_blur})", flush=True)
    normal = opacity_to_normal(opacity, blur_radius=args.normal_blur)
    normal.save(out_normal)
    print(f"[voronoi] saved {os.path.basename(out_normal)} ({normal.size})",
          flush=True)

    print("[voronoi] done", flush=True)


if __name__ == "__main__":
    main()
