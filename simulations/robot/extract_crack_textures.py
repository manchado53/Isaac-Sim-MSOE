#!/usr/bin/env python3
"""
extract_crack_textures.py

Pair every cracked synthetic frame with its same-viewpoint intact counterpart
and diff them to recover the crack overlay as a clean alpha PNG per material
variant (edge_v1, v4, tempered_v5). The output library is what Stage 3
composites onto real Diercks Hall photos for sim-to-real glass eval.

Pairing key: (panel, L, arc, d, uv, h) — see data_glass/downsample_glass_val_test.py.
Each bucket has one M-clear intact frame and three cracked frames (one per
material). Running this script walks every cracked frame and emits one PNG
per (cracked frame, intact counterpart) pair.

Usage:
  python3 extract_crack_textures.py \
      --src data_glass/tanner_vlm_dataset_balanced/frames/train \
      --out data/crack_textures \
      --max-per-material 120 \
      --bbox-crop

Output:
  data/crack_textures/edge_v1/*.png   (L mode, alpha-only)
  data/crack_textures/v4/*.png
  data/crack_textures/tempered_v5/*.png
  data/crack_textures/manifest.json   (filenames, bboxes, source pair)
  data/crack_textures/contact_sheet.jpg  (12x3 grid, 4 samples per material)
"""

import argparse
import json
import os
import random
import re
import sys
from collections import defaultdict

import numpy as np
from PIL import Image
from scipy import ndimage

FILENAME_RE = re.compile(
    r"^(?P<label>cracked|intact)_pan(?P<panel>\d+)"
    r"_L-(?P<L>.+?)_M-(?P<M>.+?)"
    r"_arc(?P<arc>[+-]\d+)_d(?P<d>\d+)_uv(?P<uv>\d+)_h(?P<h>[+-]\d+)"
    r"_f(?P<f>\d+)\.jpg$"
)

CRACK_MATERIALS = {"edge_v1", "v4", "tempered_v5"}


def parse(basename: str):
    m = FILENAME_RE.match(basename)
    return m.groupdict() if m else None


def viewpoint_key(meta: dict) -> tuple:
    return (meta["panel"], meta["L"], meta["arc"], meta["d"], meta["uv"], meta["h"])


def index_frames(src_dir: str):
    """Return (intact_map, cracked_list).

    intact_map : viewpoint_key -> list[(abspath, meta)]
        (there can be several intact frames per bucket at different f indices).
    cracked_list : list[(abspath, meta)]
    """
    intact_map: dict = defaultdict(list)
    cracked_list: list = []
    for name in sorted(os.listdir(src_dir)):
        if not name.endswith(".jpg"):
            continue
        meta = parse(name)
        if meta is None:
            continue
        path = os.path.join(src_dir, name)
        if meta["label"] == "intact":
            intact_map[viewpoint_key(meta)].append((path, meta))
        else:
            if meta["M"] in CRACK_MATERIALS:
                cracked_list.append((path, meta))
    return intact_map, cracked_list


def load_rgb(path: str) -> np.ndarray:
    with Image.open(path) as im:
        return np.asarray(im.convert("RGB"), dtype=np.uint8)


def extract_alpha(
    cracked: np.ndarray,
    intact: np.ndarray,
    threshold: int = 18,
    tophat_size: int = 9,
    min_component_px: int = 12,
    roi_blur: int = 25,
    roi_threshold: int = 10,
    roi_dilate: int = 10,
) -> np.ndarray:
    """Return a uint8 alpha map (0..255) isolating the crack overlay.

    The raw |cracked - intact| mixes two signals:
      (a) low-frequency reflection differences on the whole cracked pane (the
          crack material's specular differs from clear glass), and
      (b) high-frequency crack ridges themselves.

    We want only (b). Strategy:
      1. Compute the ROI = blurred delta thresholded, then dilated. This is
         the (whole-pane) region where *any* change happened — but it excludes
         scene-edge noise outside the cracked pane.
      2. Inside the ROI, apply a white top-hat to the delta, which suppresses
         the pane's low-frequency reflection shift and leaves only thin ridges.
      3. Threshold + component filter.
    """
    if cracked.shape != intact.shape:
        raise ValueError(f"shape mismatch: {cracked.shape} vs {intact.shape}")

    delta = np.max(np.abs(cracked.astype(np.int16) - intact.astype(np.int16)), axis=2)
    delta = np.clip(delta, 0, 255).astype(np.uint8)

    # --- ROI gate: where did the pane change? -------------------------------
    blurred = ndimage.gaussian_filter(delta, sigma=roi_blur)
    roi = blurred >= roi_threshold
    if roi_dilate > 0:
        roi = ndimage.binary_dilation(roi, iterations=roi_dilate)
    # Keep only the largest ROI component — real cracked pane is one big region.
    labeled_roi, n_roi = ndimage.label(roi)
    if n_roi > 1:
        sizes = ndimage.sum(roi, labeled_roi, index=np.arange(1, n_roi + 1))
        keep = int(np.argmax(sizes)) + 1
        roi = (labeled_roi == keep)

    # --- Top-hat to extract thin ridges inside ROI --------------------------
    opened = ndimage.grey_opening(delta, size=tophat_size)
    tophat = np.clip(delta.astype(np.int16) - opened.astype(np.int16), 0, 255).astype(np.uint8)
    tophat = np.where(roi, tophat, 0).astype(np.uint8)

    binary = tophat >= threshold

    if min_component_px > 0:
        labeled, n = ndimage.label(binary)
        if n > 0:
            sizes = ndimage.sum(binary, labeled, index=np.arange(1, n + 1))
            too_small = np.zeros(n + 1, dtype=bool)
            too_small[1:] = sizes < min_component_px
            binary[too_small[labeled]] = False

    alpha = np.where(binary, tophat, 0).astype(np.uint8)

    # Normalize so the 99th percentile of non-zero pixels maps to 255, keeping
    # relative intensity differences. Makes downstream compositing
    # intensity-independent of the original scene lighting.
    nz = alpha[alpha > 0]
    if nz.size > 0:
        hi = float(np.percentile(nz, 99))
        if hi > 0:
            scaled = alpha.astype(np.float32) * (255.0 / hi)
            alpha = np.clip(scaled, 0, 255).astype(np.uint8)

    return alpha


def thin_ridge_fraction(alpha: np.ndarray) -> float:
    """Fraction of non-zero pixels that belong to thin ridge components.

    A thin ridge has perimeter ~ area (low-width structures). A blob has
    perimeter << area. We score each component by perimeter/area and the
    overall fraction of "thin" pixels is our quality signal.

    Returns a value in [0, 1]; cracks score high, reflection blobs score low.
    """
    binary = alpha > 0
    if not binary.any():
        return 0.0
    # A pixel is on the edge if any 4-neighbor is background.
    edge = binary & ~(
        ndimage.binary_erosion(binary, structure=np.array([[0, 1, 0],
                                                           [1, 1, 1],
                                                           [0, 1, 0]]))
    )
    # Thin structures have every pixel near an edge. Score = edge_pixels / total.
    return float(edge.sum()) / float(binary.sum())


def crop_to_bbox(alpha: np.ndarray, padding: int = 8):
    """Return (cropped_alpha, (x0, y0, x1, y1)) or (None, None) if empty."""
    ys, xs = np.where(alpha > 0)
    if ys.size == 0:
        return None, None
    h, w = alpha.shape
    y0 = max(0, ys.min() - padding)
    y1 = min(h, ys.max() + padding + 1)
    x0 = max(0, xs.min() - padding)
    x1 = min(w, xs.max() + padding + 1)
    return alpha[y0:y1, x0:x1], (int(x0), int(y0), int(x1), int(y1))


def out_filename(meta: dict) -> str:
    # drop the M- prefix (implied by the output folder) and the label (always cracked)
    return (
        f"pan{meta['panel']}_L-{meta['L']}_arc{meta['arc']}_d{meta['d']}"
        f"_uv{meta['uv']}_h{meta['h']}_f{meta['f']}.png"
    )


def build_contact_sheet(out_root: str, per_material: int = 4):
    """Render a single JPG with `per_material` crack samples per material.

    Each row = one material. Cells are rescaled to a fixed 256x256 so rows line up.
    """
    cell = 256
    cols = per_material
    rows = len(CRACK_MATERIALS)
    mats = sorted(CRACK_MATERIALS)
    sheet = np.full((rows * cell, cols * cell, 3), 32, dtype=np.uint8)  # dark bg
    for r, mat in enumerate(mats):
        mat_dir = os.path.join(out_root, mat)
        if not os.path.isdir(mat_dir):
            continue
        files = sorted(f for f in os.listdir(mat_dir) if f.endswith(".png"))
        random.Random(0).shuffle(files)
        picks = files[:cols]
        for c, fn in enumerate(picks):
            with Image.open(os.path.join(mat_dir, fn)) as im:
                im = im.convert("L").resize((cell, cell), Image.BILINEAR)
                a = np.asarray(im, dtype=np.uint8)
            # Show crack as white-on-dark so it's visible against the gray bg.
            cell_rgb = np.stack([a, a, a], axis=-1)
            sheet[r * cell:(r + 1) * cell, c * cell:(c + 1) * cell] = cell_rgb
    Image.fromarray(sheet).save(os.path.join(out_root, "contact_sheet.jpg"), quality=90)


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--src", required=True, help="Path to synthetic frames/train dir")
    p.add_argument("--out", required=True, help="Output root for texture library")
    p.add_argument("--max-per-material", type=int, default=120,
                   help="Cap textures saved per material (reservoir-sample from full set)")
    p.add_argument("--threshold", type=int, default=18,
                   help="Min top-hat intensity to count as crack pixel")
    p.add_argument("--tophat-size", type=int, default=9,
                   help="Disk size (px) for white top-hat; should exceed crack line width")
    p.add_argument("--min-component-px", type=int, default=12)
    p.add_argument("--min-crack-px", type=int, default=300,
                   help="Reject extractions with fewer non-zero pixels than this")
    p.add_argument("--min-thin-fraction", type=float, default=0.35,
                   help="Reject extractions where thin-ridge fraction < this (blob-heavy)")
    p.add_argument("--roi-blur", type=int, default=25,
                   help="Gaussian sigma for low-freq ROI mask")
    p.add_argument("--roi-threshold", type=int, default=10,
                   help="Blurred-delta threshold to decide cracked pane ROI")
    p.add_argument("--roi-dilate", type=int, default=10,
                   help="Dilation iterations on the ROI (crack ridges near pane edges)")
    p.add_argument("--bbox-crop", action="store_true",
                   help="Save alpha PNG cropped to crack bbox (smaller, retains spatial bbox)")
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--limit", type=int, default=0,
                   help="Stop after processing this many cracked frames (debug); 0 = no limit")
    args = p.parse_args()

    if not os.path.isdir(args.src):
        print(f"ERROR: src not found: {args.src}", file=sys.stderr)
        sys.exit(1)
    os.makedirs(args.out, exist_ok=True)
    for mat in CRACK_MATERIALS:
        os.makedirs(os.path.join(args.out, mat), exist_ok=True)

    print(f"Indexing frames in {args.src}...")
    intact_map, cracked_list = index_frames(args.src)
    print(f"  intact buckets: {len(intact_map)}   cracked frames: {len(cracked_list)}")

    rng = random.Random(args.seed)
    rng.shuffle(cracked_list)
    if args.limit:
        cracked_list = cracked_list[:args.limit]

    kept_per_mat: dict = defaultdict(int)
    manifest = []
    unpaired = 0

    for i, (crack_path, cmeta) in enumerate(cracked_list, 1):
        mat = cmeta["M"]
        if kept_per_mat[mat] >= args.max_per_material:
            # Check whether all materials are full — if so, stop early.
            if all(kept_per_mat[m] >= args.max_per_material for m in CRACK_MATERIALS):
                break
            continue

        intact_candidates = intact_map.get(viewpoint_key(cmeta), [])
        if not intact_candidates:
            unpaired += 1
            continue
        # Pick the intact frame with nearest f index (small drift possible due to
        # sampler frame counters). Doesn't matter much — viewpoint is the same.
        cf = int(cmeta["f"])
        intact_path, imeta = min(
            intact_candidates,
            key=lambda t: abs(int(t[1]["f"]) - cf),
        )

        cracked = load_rgb(crack_path)
        intact = load_rgb(intact_path)
        alpha_full = extract_alpha(
            cracked, intact,
            threshold=args.threshold,
            tophat_size=args.tophat_size,
            min_component_px=args.min_component_px,
            roi_blur=args.roi_blur,
            roi_threshold=args.roi_threshold,
            roi_dilate=args.roi_dilate,
        )

        crack_px = int(np.count_nonzero(alpha_full))
        if crack_px < args.min_crack_px:
            continue
        thin_frac = thin_ridge_fraction(alpha_full)
        if thin_frac < args.min_thin_fraction:
            continue

        if args.bbox_crop:
            alpha, bbox = crop_to_bbox(alpha_full)
            if alpha is None:
                continue
        else:
            alpha = alpha_full
            bbox = (0, 0, alpha.shape[1], alpha.shape[0])

        out_name = out_filename(cmeta)
        out_path = os.path.join(args.out, mat, out_name)
        Image.fromarray(alpha, mode="L").save(out_path, optimize=True)

        kept_per_mat[mat] += 1
        manifest.append({
            "material": mat,
            "texture": f"{mat}/{out_name}",
            "bbox_xyxy": bbox,
            "full_size": [cracked.shape[1], cracked.shape[0]],
            "source_cracked": os.path.basename(crack_path),
            "source_intact": os.path.basename(intact_path),
            "crack_px": int(np.count_nonzero(alpha)),
            "thin_fraction": round(thin_frac, 3),
        })

        if i % 200 == 0:
            msg = "  " + "  ".join(f"{m}:{kept_per_mat[m]}" for m in sorted(CRACK_MATERIALS))
            print(f"[{i}/{len(cracked_list)}]{msg}")

    print(f"\nDone. unpaired cracked frames: {unpaired}")
    for m in sorted(CRACK_MATERIALS):
        print(f"  {m:<12s} kept {kept_per_mat[m]}")

    manifest_path = os.path.join(args.out, "manifest.json")
    with open(manifest_path, "w") as f:
        json.dump({
            "src": os.path.abspath(args.src),
            "threshold": args.threshold,
            "tophat_size": args.tophat_size,
            "min_component_px": args.min_component_px,
            "min_crack_px": args.min_crack_px,
            "min_thin_fraction": args.min_thin_fraction,
            "bbox_crop": args.bbox_crop,
            "entries": manifest,
        }, f, indent=2)
        f.flush()
        os.fsync(f.fileno())
    print(f"manifest: {manifest_path}")

    build_contact_sheet(args.out)
    print(f"contact sheet: {os.path.join(args.out, 'contact_sheet.jpg')}")


if __name__ == "__main__":
    main()
