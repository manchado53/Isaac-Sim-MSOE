#!/usr/bin/env python3
"""
Edge-crack texture generator — biased branching random walk.

Models a crack propagating inward from a panel edge: one root on a boundary,
small angle jitter per step, occasional forking, and line-width taper from
root (thick) to tip (thin) to mimic stress dissipation as the crack advances.

Unlike Voronoi (space-filling cells for tempered shatter), an edge crack is
a TREE — one root, a few branches, fewer branches as depth increases. That's
the natural topology for stress cracks, thermal breaks, and installation
cracks.

Not tileable — edge cracks are anchored to one side by construction. The
texture is UV-mapped 1:1 to the panel.

Writes:
  crack_opacity_<variant>.png   (size x size)
  crack_normal_<variant>.png    (size x size)

Examples:
  python3 edge_crack.py --variant edge_v1 --n-roots 1
  python3 edge_crack.py --variant edge_v2 --n-roots 2 --seed 17
  python3 edge_crack.py --variant edge_v3 --n-roots 3 --max-length 1800
"""

from __future__ import annotations

import argparse
import math
import os
import sys

import numpy as np
from PIL import Image, ImageDraw

_HERE = os.path.dirname(os.path.abspath(__file__))
_ASSETS = os.path.abspath(os.path.join(_HERE, "..", "..", "..", "assets", "materials", "cracked_glass"))
sys.path.insert(0, _HERE)
from generate_textures import opacity_to_normal


def sample_edge_entry(size: int, rng: np.random.Generator):
    """Pick an entry point on one of the 4 edges.

    Uses a Beta(0.7, 0.7) distribution for the along-edge position — it's
    U-shaped, biasing entries toward the two corners of each edge. Real edge
    cracks start at stress concentrations; in curtain-wall panels those are
    where the mullions meet (the corners).
    """
    side = int(rng.integers(0, 4))         # 0=top, 1=right, 2=bottom, 3=left
    t = float(rng.beta(0.7, 0.7))          # 0..1, biased toward 0 and 1
    t_px = int(t * (size - 1))

    if side == 0:
        pt = (t_px, 0)
        base_angle = math.pi / 2           # downward (into panel)
    elif side == 1:
        pt = (size - 1, t_px)
        base_angle = math.pi               # leftward
    elif side == 2:
        pt = (t_px, size - 1)
        base_angle = -math.pi / 2          # upward
    else:
        pt = (0, t_px)
        base_angle = 0.0                   # rightward

    angle = base_angle + rng.uniform(-math.pi / 12, math.pi / 12)  # ±15°
    return pt, angle


def propagate_crack(draw: ImageDraw.ImageDraw, start, angle: float, size: int,
                    rng: np.random.Generator, max_length: int,
                    angle_jitter: float, branch_prob: float,
                    step: int = 4, root_width: int = 12, tip_width: int = 4,
                    depth: int = 0, max_depth: int = 2) -> None:
    """Walk a crack from `start` in direction `angle`, possibly branching.

    Line width tapers from `root_width` at the origin to `tip_width` at the
    tip — mimics how a propagating crack loses stress and narrows with depth.
    Branches inherit a reduced `branch_prob` so forks don't explode.

    Terminates at panel boundaries — real edge cracks die at the opposite
    edge, they don't wrap. With the default low `angle_jitter` (~2.3°/step)
    a crack keeps its dominant direction long enough to span most of the
    panel before dying.
    """
    x, y = start
    path_len = 0
    while path_len < max_length:
        dx = step * math.cos(angle)
        dy = step * math.sin(angle)
        nx, ny = x + dx, y + dy
        if not (0 <= nx < size and 0 <= ny < size):
            break

        t = path_len / max(max_length, 1)   # 0..1 along this branch
        w = max(int(round(root_width * (1 - t) + tip_width * t)), 1)
        draw.line([(x, y), (nx, ny)], fill=0, width=w)

        if depth < max_depth and rng.random() < branch_prob:
            branch_angle = angle + (
                float(rng.choice([-1.0, 1.0]))
                * rng.uniform(math.pi / 6, math.pi / 3)          # 30°..60°
            )
            propagate_crack(
                draw, (nx, ny), branch_angle, size, rng,
                max_length=int((max_length - path_len) * 0.4),   # branches die fast
                angle_jitter=angle_jitter,
                branch_prob=branch_prob * 0.25,                  # forks rare at depth
                step=step,
                root_width=max(w - 1, 1),
                tip_width=tip_width,
                depth=depth + 1,
                max_depth=max_depth,
            )

        x, y = nx, ny
        path_len += step
        angle += rng.normal(0, angle_jitter)


def parse_args():
    p = argparse.ArgumentParser(description=__doc__.splitlines()[1])
    p.add_argument("--variant", default="edge_v1",
                   help="Output variant name — writes crack_{opacity,normal}_<variant>.png "
                        "(default edge_v1)")
    p.add_argument("--size", type=int, default=2048,
                   help="Output texture side length (default 2048)")
    p.add_argument("--n-roots", type=int, default=1,
                   help="How many independent cracks to propagate (default 1)")
    p.add_argument("--max-length", type=int, default=3000,
                   help="Max path length per root in pixels (default 3000). The walk "
                        "reflects off boundaries, so this is total ink-down length, not "
                        "straight-line reach")
    p.add_argument("--branch-prob", type=float, default=0.008,
                   help="Per-step probability of forking (default 0.008 = ~1 branch per "
                        "125 steps = ~6 branches per full 3000/step=4 crack). Real glass "
                        "edge cracks branch sparsely, not fractally")
    p.add_argument("--angle-jitter", type=float, default=0.025,
                   help="Std-dev of angle jitter per step in radians (default 0.025 ≈ 1.4°). "
                        "Low enough that cracks read as mostly-straight with gentle "
                        "curvature, matching real glass fracture propagation")
    p.add_argument("--step", type=int, default=4,
                   help="Step size in px (default 4). Smaller = smoother curves, slower")
    p.add_argument("--root-width", type=int, default=12,
                   help="Line width at crack root in px (default 12). Tuned for the "
                        "cracks to read as unambiguous damage through OmniGlass at "
                        "architectural viewing distances — thin hairlines get lost "
                        "against the refraction noise of a full-size glass panel")
    p.add_argument("--tip-width", type=int, default=4,
                   help="Line width at crack tip in px (default 4)")
    p.add_argument("--seed", type=int, default=42,
                   help="RNG seed for reproducibility")
    p.add_argument("--normal-blur", type=int, default=0,
                   help="Gaussian blur radius before Sobel (default 0 = pixel-crisp "
                        "edges, reads as a hairline fracture rather than a soft smudge)")
    p.add_argument("--normal-strength", type=float, default=40.0,
                   help="Normal-map gradient scale (default 40). Higher = cracks look "
                        "like deeper scratches through the glass. The generate_textures "
                        "default of 18 is tuned for soft diffusion textures — edge "
                        "cracks need more punch to read as fractures through OmniGlass")
    p.add_argument("--debug", action="store_true",
                   help="Also save a debug copy of the raw opacity")
    return p.parse_args()


def main():
    args = parse_args()
    rng = np.random.default_rng(args.seed)

    print(f"[edge] generating {args.n_roots} roots, seed={args.seed}", flush=True)
    img = Image.new("L", (args.size, args.size), 255)
    draw = ImageDraw.Draw(img)

    for i in range(args.n_roots):
        start, angle = sample_edge_entry(args.size, rng)
        print(f"[edge]   root {i}: start={start} angle={math.degrees(angle):+.1f}°",
              flush=True)
        propagate_crack(
            draw, start, angle, args.size, rng,
            max_length=args.max_length,
            angle_jitter=args.angle_jitter,
            branch_prob=args.branch_prob,
            step=args.step,
            root_width=args.root_width,
            tip_width=args.tip_width,
        )

    pct = 100.0 * np.mean(np.asarray(img) == 0)
    print(f"[edge] crack coverage: {pct:.2f}%", flush=True)

    out_op = os.path.join(_ASSETS, f"crack_opacity_{args.variant}.png")
    out_nm = os.path.join(_ASSETS, f"crack_normal_{args.variant}.png")
    out_rh = os.path.join(_ASSETS, f"crack_roughness_{args.variant}.png")
    img.save(out_op)
    print(f"[edge] saved {os.path.basename(out_op)} ({img.size})", flush=True)

    # Invert opacity for a roughness map: cracks (black=0 in opacity) become
    # white (255) = high roughness, so OmniGlass scatters light there and the
    # crack reads as a visibly frosted fracture instead of a faint normal-map
    # perturbation. Intact glass stays black (0) = smooth = clear.
    rough = Image.eval(img, lambda x: 255 - x)
    rough.save(out_rh)
    print(f"[edge] saved {os.path.basename(out_rh)} ({rough.size})", flush=True)

    if args.debug:
        img.save(os.path.join(_ASSETS, f"_debug_{args.variant}_raw_opacity.png"))

    normal = opacity_to_normal(img, blur_radius=args.normal_blur,
                               strength=args.normal_strength)
    normal.save(out_nm)
    print(f"[edge] saved {os.path.basename(out_nm)} ({normal.size})", flush=True)
    print("[edge] done", flush=True)


if __name__ == "__main__":
    main()
