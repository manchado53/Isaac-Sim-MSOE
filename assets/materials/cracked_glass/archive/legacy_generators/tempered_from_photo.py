#!/usr/bin/env python3
"""
Build the tempered-glass opacity + normal textures directly from the reference
photograph — no diffusion, no SDXL. Classical CV only.

Pipeline:
  1. Load references/tempered_safety2.jpg (2448x2448 with foliage top + frame bottom)
  2. Crop to a clean dice region
  3. Extract the thin crack network via Sobel-gradient edge detection
     (NOT by thresholding darkness — shard interiors are dark too and would
     get included; only the transitions between shards are real crack lines)
  4. Mirror-tile the crop to a 2048x2048 seamlessly-tileable opacity map
  5. Derive a normal map via Sobel (reuses generate_textures.opacity_to_normal)

Why mirror-tile instead of offset-blend: the dice pattern is dense and random,
so a 2-fold mirror symmetry is not visually perceptible, and mirror-tiling is
guaranteed seamless at the boundaries (a tile's right edge mirrors its left,
so when GPU-tiled they match exactly).

Writes:
  crack_opacity_tempered.png   (2048x2048, cracks=0, glass=255)
  crack_normal_tempered.png    (2048x2048, tangent-space normal)

Run:  python3 assets/materials/cracked_glass/tempered_from_photo.py
"""

from __future__ import annotations

import argparse
import os
import sys

import numpy as np
from PIL import Image, ImageFilter

_HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, _HERE)
from generate_textures import opacity_to_normal


REF_PATH = os.path.join(_HERE, "references", "tempered_safety2.jpg")


def parse_args():
    p = argparse.ArgumentParser(description=__doc__.splitlines()[1])
    p.add_argument("--crop-x", type=int, default=600,
                   help="Crop origin X in the 2448x2448 reference (default 600)")
    p.add_argument("--crop-y", type=int, default=800,
                   help="Crop origin Y (default 800 — skips top foliage)")
    p.add_argument("--crop-size", type=int, default=1024,
                   help="Crop side length (default 1024). Output is 2x this after mirror-tile")
    p.add_argument("--method", choices=["sobel", "canny"], default="sobel",
                   help="Edge-detection method (default sobel). 'canny' uses scipy and "
                        "gives cleaner connected long-running edges via non-max suppression "
                        "+ hysteresis thresholding")
    p.add_argument("--pre-blur", type=float, default=1.2,
                   help="Gaussian blur radius in px before edge detection (default 1.2). "
                        "Denoises the photo so the detector fires on real crack edges, not grain")
    p.add_argument("--edge-pct", type=float, default=12.0,
                   help="For --method sobel: percentage of pixels to mark as crack (default 12.0). "
                        "Tempered dice networks typically land around 8-15%% coverage")
    p.add_argument("--canny-low", type=float, default=18.0,
                   help="For --method canny: low hysteresis threshold (default 18)")
    p.add_argument("--canny-high", type=float, default=50.0,
                   help="For --method canny: high hysteresis threshold (default 50)")
    p.add_argument("--dilate", type=int, default=1,
                   help="Widen crack lines by this many px (default 1). 0 = raw edges (thin 1-px), "
                        "1-2 = visible as rendered normal-map ridges")
    p.add_argument("--close", type=int, default=0,
                   help="Morphological closing iterations (default 0). Dilate then erode "
                        "to connect nearby edge fragments into closed polygons without "
                        "permanently widening the lines. 1-2 connects small gaps")
    p.add_argument("--normal-blur", type=int, default=1,
                   help="Gaussian blur radius before Sobel, in px (default 1). "
                        "Use 1 for thin dense lines, 3 for thick lines")
    p.add_argument("--suffix", default="",
                   help="Optional output suffix — writes crack_{opacity,normal}_tempered<suffix>.png. "
                        "Useful for saving iterations side-by-side (e.g. --suffix _v1)")
    p.add_argument("--debug", action="store_true",
                   help="Also save intermediate images (crop, raw_opacity)")
    return p.parse_args()


def _sobel_magnitude(gray: np.ndarray) -> np.ndarray:
    """Compute Sobel gradient magnitude with numpy only (PIL+numpy, no scipy)."""
    g = gray.astype(np.float32)
    # 3x3 Sobel via shift-and-subtract (equivalent to convolution with Sobel kernels)
    gx = np.zeros_like(g)
    gy = np.zeros_like(g)
    gx[:, 1:-1] = (g[:, 2:] - g[:, :-2])
    gy[1:-1, :] = (g[2:, :] - g[:-2, :])
    # Smooth orthogonal direction (the ±1/2/1 weighting of a true Sobel)
    gx[1:-1, 1:-1] = (gx[:-2, 1:-1] + 2 * gx[1:-1, 1:-1] + gx[2:, 1:-1]) / 4.0
    gy[1:-1, 1:-1] = (gy[1:-1, :-2] + 2 * gy[1:-1, 1:-1] + gy[1:-1, 2:]) / 4.0
    return np.sqrt(gx * gx + gy * gy)


def _dilate_binary(mask: np.ndarray, iters: int) -> np.ndarray:
    """Binary dilation with a 3x3 cross kernel, `iters` passes. PIL+numpy only."""
    out = mask.copy()
    for _ in range(iters):
        nxt = out.copy()
        nxt[1:, :] |= out[:-1, :]
        nxt[:-1, :] |= out[1:, :]
        nxt[:, 1:] |= out[:, :-1]
        nxt[:, :-1] |= out[:, 1:]
        out = nxt
    return out


def _erode_binary(mask: np.ndarray, iters: int) -> np.ndarray:
    """Binary erosion = dilation of the complement. Cross 3x3, `iters` passes."""
    out = mask.copy()
    for _ in range(iters):
        nxt = out.copy()
        nxt[1:, :] &= out[:-1, :]
        nxt[:-1, :] &= out[1:, :]
        nxt[:, 1:] &= out[:, :-1]
        nxt[:, :-1] &= out[:, 1:]
        out = nxt
    return out


def _morph_close(mask: np.ndarray, iters: int) -> np.ndarray:
    """Morphological closing: dilate then erode. Bridges small gaps between
    nearby edge fragments without permanently thickening the lines."""
    if iters <= 0:
        return mask
    return _erode_binary(_dilate_binary(mask, iters), iters)


def _canny_edges(gray: np.ndarray, sigma: float, low: float, high: float) -> np.ndarray:
    """Canny-lite edge detection via scipy.ndimage.

    Produces thin connected edges rather than thresholded gradient magnitude:
      1. Gaussian blur (sigma)
      2. Sobel gx, gy → magnitude + angle
      3. Non-max suppression along gradient direction (approx via dilation-compare)
      4. Double threshold + hysteresis — strong edges unconditionally, weak
         edges only if they connect to a strong edge.
    """
    from scipy.ndimage import gaussian_filter, sobel, grey_dilation

    g = gaussian_filter(gray.astype(np.float32), sigma=sigma)
    gx = sobel(g, axis=1)
    gy = sobel(g, axis=0)
    mag = np.hypot(gx, gy)

    # Non-max suppression: keep pixels that are local maxima along the gradient
    # direction. Approximated as "keep pixels equal to their 3x3 max" (cheap, not
    # perfectly directional, but enough to thin the edges meaningfully).
    local_max = grey_dilation(mag, size=3)
    nms = np.where(mag >= local_max - 1e-6, mag, 0.0)

    strong = nms > high
    weak = (nms > low) & (~strong)

    # Hysteresis: iteratively grow strong edges through weak neighbours
    out = strong.copy()
    for _ in range(32):
        grown = _dilate_binary(out, 1) & (strong | weak)
        if np.array_equal(grown, out):
            break
        out = grown
    return out


def edges_to_opacity(gray: np.ndarray, method: str, pre_blur: float,
                     edge_pct: float, canny_low: float, canny_high: float,
                     dilate: int, close: int) -> np.ndarray:
    """Extract the crack NETWORK (not shard darkness) via edge detection.

    - method=sobel: percentile threshold on Sobel gradient magnitude.
      Top `edge_pct`% of gradient-magnitude pixels are marked as crack.
      Robust to outlier gradients that would squish a fixed threshold.
    - method=canny: Canny edge detection (scipy) — non-max suppression and
      hysteresis give cleaner, more connected long-running edges.

    Optional morphological closing bridges small gaps between fragments
    into closed polygons. Optional final dilation widens lines for
    rendering visibility.

    Returns uint8 with cracks=0, glass=255.
    """
    if method == "canny":
        is_edge = _canny_edges(gray, sigma=pre_blur, low=canny_low, high=canny_high)
    else:
        blurred = Image.fromarray(gray).filter(ImageFilter.GaussianBlur(pre_blur))
        smoothed = np.asarray(blurred, dtype=np.uint8)
        mag = _sobel_magnitude(smoothed)
        thresh = np.percentile(mag, 100.0 - edge_pct)
        is_edge = mag > thresh

    is_edge = _morph_close(is_edge, close)
    if dilate > 0:
        is_edge = _dilate_binary(is_edge, dilate)
    return np.where(is_edge, 0, 255).astype(np.uint8)


def mirror_tile_2x(arr: np.ndarray) -> np.ndarray:
    """Compose a 2x2 mirror-tiled version. Side length doubles."""
    top = np.hstack([arr, np.fliplr(arr)])
    bot = np.hstack([np.flipud(arr), np.flipud(np.fliplr(arr))])
    return np.vstack([top, bot])


def main():
    args = parse_args()

    if not os.path.isfile(REF_PATH):
        print(f"[tempered] ERROR: reference not found: {REF_PATH}", flush=True)
        sys.exit(1)

    print(f"[tempered] loading {os.path.basename(REF_PATH)}", flush=True)
    ref = Image.open(REF_PATH).convert("L")
    W, H = ref.size
    print(f"[tempered] reference size: {W}x{H}", flush=True)

    x, y, s = args.crop_x, args.crop_y, args.crop_size
    if x + s > W or y + s > H:
        print(f"[tempered] ERROR: crop ({x},{y})+{s} exceeds reference {W}x{H}",
              flush=True)
        sys.exit(1)

    crop = ref.crop((x, y, x + s, y + s))
    print(f"[tempered] crop region: x={x} y={y} size={s}x{s}", flush=True)

    if args.debug:
        crop.save(os.path.join(_HERE, "_debug_tempered_crop.png"))

    crop_arr = np.asarray(crop, dtype=np.uint8)

    if args.method == "canny":
        print(f"[tempered] edge extraction: canny sigma={args.pre_blur} "
              f"low={args.canny_low} high={args.canny_high} "
              f"close={args.close} dilate={args.dilate}", flush=True)
    else:
        print(f"[tempered] edge extraction: sobel pre_blur={args.pre_blur} "
              f"top_pct={args.edge_pct}% close={args.close} dilate={args.dilate}",
              flush=True)
    opacity_small = edges_to_opacity(crop_arr,
                                     method=args.method,
                                     pre_blur=args.pre_blur,
                                     edge_pct=args.edge_pct,
                                     canny_low=args.canny_low,
                                     canny_high=args.canny_high,
                                     dilate=args.dilate,
                                     close=args.close)

    crack_pct = 100.0 * (opacity_small == 0).sum() / opacity_small.size
    print(f"[tempered] crack coverage (single tile): {crack_pct:.1f}%", flush=True)

    if args.debug:
        Image.fromarray(opacity_small).save(
            os.path.join(_HERE, f"_debug_tempered{args.suffix}_raw_opacity.png"))

    print(f"[tempered] mirror-tiling {s}x{s} -> {2*s}x{2*s}", flush=True)
    opacity_full = mirror_tile_2x(opacity_small)

    target = 2048
    if opacity_full.shape[0] != target:
        print(f"[tempered] resizing {opacity_full.shape[0]}x{opacity_full.shape[1]} "
              f"-> {target}x{target}", flush=True)
        opacity_full_img = Image.fromarray(opacity_full).resize(
            (target, target), Image.LANCZOS)
    else:
        opacity_full_img = Image.fromarray(opacity_full)

    out_opacity = os.path.join(_HERE, f"crack_opacity_tempered{args.suffix}.png")
    out_normal = os.path.join(_HERE, f"crack_normal_tempered{args.suffix}.png")

    opacity_full_img.save(out_opacity)
    print(f"[tempered] saved {os.path.basename(out_opacity)} "
          f"({opacity_full_img.size})", flush=True)

    print(f"[tempered] deriving normal map (blur={args.normal_blur})", flush=True)
    normal = opacity_to_normal(opacity_full_img, blur_radius=args.normal_blur)
    normal.save(out_normal)
    print(f"[tempered] saved {os.path.basename(out_normal)} ({normal.size})",
          flush=True)

    print("[tempered] done", flush=True)


if __name__ == "__main__":
    main()
