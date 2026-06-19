#!/usr/bin/env python3
"""
composite_cracked_glass.py

Stage 3 of the DH cracked-glass eval pipeline. For each DH photo × each
synthetic crack-material variant, produce one composite image where the
extracted crack texture is overlaid onto the segmented glass region.

Design:
  - Crack textures from Stage 1 are L-mode alpha PNGs (bbox-cropped).
  - Glass masks from Stage 2 are binary PNGs at full DH resolution.
  - We pick a sub-region of the glass mask (partial shatter — matches synthetic
    training where only part of a pane is cracked), warp/scale the crack
    texture to fit, then composite with luminance-adaptive blending so the
    crack contrasts against whatever is behind the glass.

Output layout (mirrors data_glass/tanner_vlm_dataset_balanced/):
  data/dh_cracked_eval/
    intact/Image_NN.jpg        # copy of the 30 originals (label: GLASS_INTACT)
    cracked/
      Image_NN__edge_v1.jpg
      Image_NN__v4.jpg
      Image_NN__tempered_v5.jpg
    test.jsonl                 # 120 rows: 30 intact + 90 cracked
    README.md
"""

import argparse
import json
import os
import random
import re
import shutil
import sys
from collections import defaultdict

import cv2
import numpy as np
from PIL import Image, ImageOps
from scipy import ndimage

CRACK_MATERIALS = ["edge_v1", "v4", "tempered_v5"]

# Per-material ratio amplification (applied after high-pass). Values >1 pull
# deviations from 1.0 outward before clipping. edge_v1 needs a mild boost so
# hairlines register; pushing it too high amplifies residual training-scene
# silhouettes into implausible black polygons (the cracked/intact pair differs
# globally for edge_v1 due to material BRDF, not only at the crack lines).
MATERIAL_RATIO_BOOST = {
    "edge_v1": 2.2,
    "v4": 1.25,
    "tempered_v5": 1.0,
}

TRAIN_FILENAME_RE = re.compile(
    r"^(?P<label>cracked|intact)_pan(?P<panel>\d+)"
    r"_L-(?P<L>.+?)_M-(?P<M>.+?)"
    r"_arc(?P<arc>[+-]\d+)_d(?P<d>\d+)_uv(?P<uv>\d+)_h(?P<h>[+-]\d+)"
    r"_f(?P<f>\d+)\.jpg$"
)


def load_gray(path: str) -> np.ndarray:
    """Load as grayscale with EXIF orientation applied."""
    with Image.open(path) as im:
        im = ImageOps.exif_transpose(im).convert("L")
        return np.asarray(im, dtype=np.uint8)


def load_rgb(path: str) -> np.ndarray:
    """Load as RGB with EXIF orientation applied (matches segment_dh_panels.py)."""
    with Image.open(path) as im:
        im = ImageOps.exif_transpose(im).convert("RGB")
        return np.asarray(im, dtype=np.uint8)


def sample_sub_region(mask: np.ndarray, rng: random.Random,
                      frac: float = 1.0) -> np.ndarray:
    """Return a boolean sub-mask covering ~`frac` of the glass mask.

    If `frac` >= 0.999 we just return the full mask (training data shows
    whole-pane shatter — partial coverage was a design leftover from the
    early pipeline). Otherwise picks a random axis-aligned rectangle inside
    the glass bounding box.
    """
    mask_bool = mask.astype(bool)
    if frac >= 0.999:
        return mask_bool
    ys, xs = np.where(mask_bool)
    if ys.size == 0:
        return np.zeros_like(mask_bool)
    y0, y1 = ys.min(), ys.max()
    x0, x1 = xs.min(), xs.max()
    bh = y1 - y0 + 1
    bw = x1 - x0 + 1
    side_frac = float(np.sqrt(frac))
    rh = max(1, int(bh * side_frac))
    rw = max(1, int(bw * side_frac))
    ry0 = rng.randint(y0, y1 - rh + 1)
    rx0 = rng.randint(x0, x1 - rw + 1)
    rect = np.zeros_like(mask_bool)
    rect[ry0:ry0 + rh, rx0:rx0 + rw] = True
    return rect & mask_bool


def rotate_crack(crack: np.ndarray, angle_deg: float) -> np.ndarray:
    # ndimage.rotate with order=1 (bilinear) on the L-mode alpha.
    out = ndimage.rotate(crack, angle_deg, reshape=True, order=1, cval=0)
    return np.clip(out, 0, 255).astype(np.uint8)


def crop_to_alpha_bbox(crack: np.ndarray, alpha_thresh: int = 16) -> np.ndarray:
    """Crop the crack texture to the tight bbox of its lit pixels.

    Removes the transparent padding that some extracted textures carry, so
    the tile density is actually crack-content instead of empty border.
    """
    ys, xs = np.where(crack > alpha_thresh)
    if ys.size == 0:
        return crack
    return crack[ys.min():ys.max() + 1, xs.min():xs.max() + 1]


def fill_holes(alpha: np.ndarray, max_hole_px: int) -> np.ndarray:
    """Close small zero-alpha holes inside a tile so cracks read as continuous.

    The raw |cracked - intact| diff has dark Voronoi cell interiors that
    register as zero alpha, leaving gaps once we tile + blend. A grayscale
    closing with a disk kernel of ~max_hole_px fills these in while keeping
    the outer silhouette.
    """
    if max_hole_px <= 0:
        return alpha
    return ndimage.grey_closing(alpha, size=max_hole_px).astype(np.uint8)


def tile_crack_over_region(crack: np.ndarray, region: np.ndarray,
                           rng: random.Random,
                           tiles_per_short_side: float = 2.0,
                           alpha_thresh: int = 16,
                           close_size: int = 9) -> np.ndarray:
    """Tile the crack texture at near-native resolution across the region bbox.

    This replaces the prior "stretch one crack to fit the whole bbox" logic,
    which blurred detail and propagated internal black holes across the whole
    pane. Tiling keeps each Voronoi/web pattern at a reasonable scale and
    guarantees the full masked region is covered.

    Steps:
      1. Crop the crack texture to its lit bbox (kill transparent padding).
      2. Close small alpha holes so each tile reads as continuous shatter.
      3. Rotate + rescale the tile so its short side is ~(bbox_short / N).
      4. Tile across the region bbox.
      5. Clip to the mask so cracks never spill onto frames/mullions.
    """
    ys, xs = np.where(region)
    if ys.size == 0:
        return np.zeros(region.shape, dtype=np.uint8)
    y0, y1 = ys.min(), ys.max() + 1
    x0, x1 = xs.min(), xs.max() + 1
    rh, rw = y1 - y0, x1 - x0

    tile = crop_to_alpha_bbox(crack, alpha_thresh=alpha_thresh)
    tile = fill_holes(tile, close_size)

    # Rotate the tile (±15°) so tiling seams don't line up with image edges.
    angle = rng.uniform(-15, 15)
    tile = rotate_crack(tile, angle)

    # Scale so the tile's short side ~= bbox_short / tiles_per_short_side.
    th, tw = tile.shape
    bbox_short = min(rh, rw)
    tile_short = max(1, int(round(bbox_short / tiles_per_short_side)))
    scale = tile_short / float(min(th, tw))
    new_w = max(1, int(round(tw * scale)))
    new_h = max(1, int(round(th * scale)))
    tile = np.asarray(
        Image.fromarray(tile, mode="L").resize((new_w, new_h), Image.BILINEAR),
        dtype=np.uint8,
    )
    th, tw = tile.shape

    # Build a canvas sized to the bbox; tile across with a random phase.
    canvas = np.zeros((rh, rw), dtype=np.uint8)
    phase_y = rng.randint(0, th - 1)
    phase_x = rng.randint(0, tw - 1)
    for dy in range(-phase_y, rh, th):
        for dx in range(-phase_x, rw, tw):
            y_lo = max(0, dy)
            x_lo = max(0, dx)
            y_hi = min(rh, dy + th)
            x_hi = min(rw, dx + tw)
            src_y = y_lo - dy
            src_x = x_lo - dx
            canvas[y_lo:y_hi, x_lo:x_hi] = np.maximum(
                canvas[y_lo:y_hi, x_lo:x_hi],
                tile[src_y:src_y + (y_hi - y_lo), src_x:src_x + (x_hi - x_lo)],
            )

    out = np.zeros(region.shape, dtype=np.uint8)
    out[y0:y1, x0:x1] = canvas
    return np.where(region, out, 0).astype(np.uint8)


def luminance_blend(image: np.ndarray, alpha_u8: np.ndarray,
                    strength: float = 1.0, blur_sigma: float = 15.0,
                    alpha_gamma: float = 1.0) -> np.ndarray:
    """Blend crack onto image: bright on dark pixels, dark on bright pixels.

    crack_color(per pixel) = 255 - local_mean_luma (invert local luma so the
    crack always contrasts). We use a blurred luma (low-freq) so we don't
    chase tiny pixel variations.

    `alpha_gamma` < 1.0 boosts thin/faint alpha values so hairline cracks stay
    visible after upscaling (0.5 roughly doubles the perceived coverage).

    out = image * (1 - a) + crack_color * a, where a = (alpha_u8/255)^gamma * strength.
    """
    luma = (0.299 * image[:, :, 0] + 0.587 * image[:, :, 1] + 0.114 * image[:, :, 2])
    luma_lf = ndimage.gaussian_filter(luma.astype(np.float32), sigma=blur_sigma)
    crack_luma = 255.0 - luma_lf
    crack_rgb = np.stack([crack_luma, crack_luma, crack_luma], axis=-1)

    a = (alpha_u8.astype(np.float32) / 255.0)
    if alpha_gamma != 1.0:
        a = np.power(a, alpha_gamma)
    a = a * strength
    a = np.clip(a, 0.0, 1.0)[:, :, None]

    out = image.astype(np.float32) * (1.0 - a) + crack_rgb * a
    return np.clip(out, 0, 255).astype(np.uint8)


def composite_one(image: np.ndarray, mask: np.ndarray, crack: np.ndarray,
                  rng: random.Random,
                  sub_region_frac: float,
                  blend_strength: float,
                  alpha_gamma: float,
                  tiles_per_short_side: float,
                  close_size: int) -> np.ndarray:
    region = sample_sub_region(mask, rng, frac=sub_region_frac)
    alpha = tile_crack_over_region(
        crack, region, rng,
        tiles_per_short_side=tiles_per_short_side,
        close_size=close_size,
    )
    return luminance_blend(image, alpha, strength=blend_strength,
                           alpha_gamma=alpha_gamma)


def _order_quad(pts: np.ndarray) -> np.ndarray:
    """Order 4 (x, y) points as [tl, tr, br, bl] using sum/diff heuristics.

    Works for any quad where the four corners map cleanly to top-left,
    top-right, bottom-right, bottom-left. Robust to mild skew / perspective.
    """
    pts = np.asarray(pts, dtype=np.float32).reshape(-1, 2)
    s = pts.sum(axis=1)
    d = pts[:, 0] - pts[:, 1]
    tl = pts[int(np.argmin(s))]
    br = pts[int(np.argmax(s))]
    tr = pts[int(np.argmax(d))]
    bl = pts[int(np.argmin(d))]
    return np.stack([tl, tr, br, bl], axis=0).astype(np.float32)


def quad_from_binary(mask: np.ndarray) -> np.ndarray | None:
    """Approximate a binary blob's contour to a 4-corner quadrilateral.

    Returns a (4, 2) float32 array ordered tl, tr, br, bl, or None if the mask
    is empty. Tries progressively looser polygon approximations, falling back
    to `minAreaRect` if no 4-corner approximation is found.
    """
    if mask.dtype != np.uint8:
        mask_u8 = (mask.astype(bool).astype(np.uint8)) * 255
    else:
        mask_u8 = (mask > 0).astype(np.uint8) * 255
    contours, _ = cv2.findContours(mask_u8, cv2.RETR_EXTERNAL,
                                   cv2.CHAIN_APPROX_SIMPLE)
    if not contours:
        return None
    c = max(contours, key=cv2.contourArea)
    if cv2.contourArea(c) < 4:
        return None
    peri = cv2.arcLength(c, True)
    for eps_mul in (0.01, 0.02, 0.03, 0.05, 0.08, 0.12, 0.18):
        approx = cv2.approxPolyDP(c, eps_mul * peri, True)
        if len(approx) == 4:
            return _order_quad(approx.reshape(-1, 2))
    rect = cv2.minAreaRect(c)
    box = cv2.boxPoints(rect)
    return _order_quad(box)


def highpass_ratio(ratio: np.ndarray, roi: np.ndarray,
                   blur_sigma: float = 40.0) -> np.ndarray:
    """Damp slow (glare-scale) variations in the ratio toward 1.0.

    Synthetic training frames carry bright polygonal glare that leaks into
    cracked / intact on the low-frequency band. Dividing the ratio by its
    own Gaussian-blurred version normalizes that drift away while preserving
    the high-frequency crack edges. Outside the ROI ratio stays at 1.0.
    """
    out = ratio.copy()
    roi_f = roi.astype(np.float32)
    for c in range(ratio.shape[2]):
        ch = ratio[:, :, c].astype(np.float32)
        # Blur ratio weighted by the ROI so identity pixels outside the ROI
        # don't pull the low-freq estimate toward 1.0 near the ROI boundary.
        num = ndimage.gaussian_filter(ch * roi_f, sigma=blur_sigma)
        den = ndimage.gaussian_filter(roi_f, sigma=blur_sigma)
        lf = np.where(den > 1e-3, num / np.clip(den, 1e-3, None), 1.0)
        lf = np.clip(lf, 0.25, 4.0)
        hp = ch / lf
        out[:, :, c] = np.where(roi, hp, 1.0)
    return out


def load_pair_to_ratio(cracked_path: str, intact_path: str,
                       material: str = "",
                       ratio_floor: float = 0.25,
                       ratio_ceil: float = 4.0,
                       identity_eps: float = 14.0,
                       roi_blur_sigma: float = 6.0,
                       roi_delta_thresh: float = 8.0,
                       roi_dilate_iter: int = 3,
                       highpass_sigma: float = 40.0) -> tuple:
    """Return (ratio_cropped, src_quad_in_cropped_coords).

    Steps:
      1. ratio[y,x,c] = cracked[y,x,c] / intact[y,x,c]  (per channel)
      2. Build an ROI mask of the cracked pane: blur |delta| and threshold to
         find where the synthetic crack actually changed the image.
      3. Keep the largest connected ROI component (the cracked pane itself;
         discards scattered compression-artifact noise outside it).
      4. High-pass the ratio inside the ROI to damp slow glare/lighting drift.
      5. Amplify deviations from 1.0 by the per-material boost (edge_v1 etc.).
      6. Crop ratio + ROI to that component's bbox and derive a 4-corner
         quadrilateral of the ROI in the cropped coord frame.

    The quad is what the downstream compositor warps onto each DH pane quad,
    so cracks land on the actual slanted / foreshortened glass surface
    instead of the mask's axis-aligned bounding box.
    """
    cracked = np.asarray(Image.open(cracked_path).convert("RGB"), dtype=np.float32)
    intact = np.asarray(Image.open(intact_path).convert("RGB"), dtype=np.float32)
    if cracked.shape != intact.shape:
        raise ValueError(f"pair shape mismatch: {cracked.shape} vs {intact.shape}")

    delta = np.max(np.abs(cracked - intact), axis=2)
    blurred = ndimage.gaussian_filter(delta, sigma=roi_blur_sigma)
    roi = blurred > roi_delta_thresh
    if roi_dilate_iter > 0:
        roi = ndimage.binary_dilation(roi, iterations=roi_dilate_iter)
    labeled, n_comp = ndimage.label(roi)
    if n_comp == 0:
        return np.ones((1, 1, 3), dtype=np.float32), None
    sizes = ndimage.sum(roi, labeled, index=np.arange(1, n_comp + 1))
    keep = int(np.argmax(sizes)) + 1
    roi = (labeled == keep)

    # Thin-feature gate. Morphological white tophat keeps features smaller than
    # the kernel and suppresses broad smooth regions. Synthetic training pairs
    # differ globally (material BRDF shifts, displaced specular highlights) —
    # that shows up as large smooth-delta patches that previously leaked into
    # the composite as black scene silhouettes. Cracks are thin linear
    # structures that survive tophat; scene silhouettes do not.
    delta_u8 = np.clip(delta, 0, 255).astype(np.uint8)
    tophat_kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (25, 25))
    thin_top = cv2.morphologyEx(delta_u8, cv2.MORPH_TOPHAT, tophat_kernel)
    thin_mask = thin_top > 10
    thin_mask = ndimage.binary_dilation(thin_mask, iterations=2)

    safe_intact = np.clip(intact, 1.0, None)
    ratio = (cracked / safe_intact).astype(np.float32)
    ratio = np.clip(ratio, ratio_floor, ratio_ceil)

    # Zero-out tiny per-pixel differences inside the ROI (compression grain).
    pixelwise_changed = np.max(np.abs(cracked - intact), axis=2, keepdims=True) >= identity_eps
    ratio = np.where(pixelwise_changed, ratio, 1.0).astype(np.float32)
    # Hard-force identity outside the ROI or outside thin-crack features.
    apply_mask = roi & thin_mask
    ratio = np.where(apply_mask[:, :, None], ratio, 1.0).astype(np.float32)

    if highpass_sigma > 0:
        ratio = highpass_ratio(ratio, roi, blur_sigma=highpass_sigma)

    boost = MATERIAL_RATIO_BOOST.get(material, 1.0)
    if boost != 1.0:
        ratio = 1.0 + (ratio - 1.0) * boost
    ratio = np.clip(ratio, ratio_floor, ratio_ceil).astype(np.float32)
    # Re-force identity outside ROI (boost moves 1.0 to 1.0, but clip may not).
    ratio = np.where(roi[:, :, None], ratio, 1.0).astype(np.float32)

    # Derive the ROI's 4-corner quad before cropping.
    src_quad_full = quad_from_binary(roi)

    ys, xs = np.where(roi)
    y0, y1 = ys.min(), ys.max() + 1
    x0, x1 = xs.min(), xs.max() + 1
    ratio_cropped = ratio[y0:y1, x0:x1]
    if src_quad_full is not None:
        src_quad = src_quad_full - np.array([x0, y0], dtype=np.float32)
    else:
        # Fallback: bbox corners.
        rh, rw = ratio_cropped.shape[:2]
        src_quad = np.array([[0, 0], [rw - 1, 0],
                             [rw - 1, rh - 1], [0, rh - 1]], dtype=np.float32)
    return ratio_cropped, src_quad


def composite_ratio(image: np.ndarray, mask: np.ndarray, ratio_rgb: np.ndarray,
                    src_quad: np.ndarray, rng: random.Random,
                    strength: float = 1.0) -> np.ndarray:
    """Warp a training-derived ratio map onto the DH pane's quadrilateral.

    Derives a 4-corner quad from the DH mask and computes the perspective
    transform that maps the training-ROI quad onto it. This lines cracks up
    with the actual pane geometry (slanted, foreshortened, skewed) instead of
    pasting an axis-aligned rectangle onto it. The ratio is applied only
    inside the DH mask so nothing spills onto the frame/mullions.
    """
    H, W = image.shape[:2]
    dst_quad = quad_from_binary(mask)
    if dst_quad is None:
        return image

    rh, rw = ratio_rgb.shape[:2]
    if src_quad is None:
        src_quad = np.array([[0, 0], [rw - 1, 0],
                             [rw - 1, rh - 1], [0, rh - 1]], dtype=np.float32)

    # Random horizontal/vertical flip of the src quad so identical pairs on
    # many DH photos don't all show the same crack orientation.
    flip_h = rng.random() < 0.5
    flip_v = rng.random() < 0.5
    src = src_quad.copy()
    if flip_h:
        src = src[[1, 0, 3, 2]]
    if flip_v:
        src = src[[3, 2, 1, 0]]

    M = cv2.getPerspectiveTransform(src.astype(np.float32),
                                    dst_quad.astype(np.float32))
    ratio_warped = cv2.warpPerspective(
        ratio_rgb.astype(np.float32), M, (W, H),
        flags=cv2.INTER_LINEAR,
        borderMode=cv2.BORDER_CONSTANT,
        borderValue=(1.0, 1.0, 1.0),
    )

    if strength != 1.0:
        ratio_warped = 1.0 + (ratio_warped - 1.0) * strength

    mask3 = mask[:, :, None]
    out = image.astype(np.float32)
    modulated = out * ratio_warped
    out = np.where(mask3, modulated, out)
    return np.clip(out, 0, 255).astype(np.uint8)


def index_training_pairs(train_dir: str):
    """Map material -> list of (cracked_path, intact_path) pairs from training data."""
    if not os.path.isdir(train_dir):
        return {}
    intact_by_vp: dict = defaultdict(list)
    cracked_by_mat: dict = defaultdict(list)
    for name in os.listdir(train_dir):
        m = TRAIN_FILENAME_RE.match(name)
        if not m:
            continue
        g = m.groupdict()
        path = os.path.join(train_dir, name)
        vp = (g["panel"], g["L"], g["arc"], g["d"], g["uv"], g["h"])
        if g["label"] == "intact":
            intact_by_vp[vp].append(path)
        elif g["M"] in CRACK_MATERIALS:
            cracked_by_mat[g["M"]].append((path, vp))

    out: dict = {mat: [] for mat in CRACK_MATERIALS}
    for mat, items in cracked_by_mat.items():
        for cracked_path, vp in items:
            if vp in intact_by_vp:
                out[mat].append((cracked_path, intact_by_vp[vp][0]))
    return out


PROMPT = ("<|vision_start|><|image_pad|><|vision_end|>"
          "What is shown in this image? "
          "Options: door open, door closed, glass cracked, glass intact.")


def jsonl_row(image_rel: str, label: str) -> dict:
    state = "GLASS_CRACKED" if label == "cracked" else "GLASS_INTACT"
    return {
        "conversations": [
            {"role": "user", "content": PROMPT},
            {"role": "assistant", "content": f"State: [{state}]."},
        ],
        "image": image_rel,
    }


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--src-images", default="dh_glass_panels")
    ap.add_argument("--masks", default="data/dh_cracked_eval/masks")
    ap.add_argument("--crack-lib", default="data/crack_textures")
    ap.add_argument("--out", default="data/dh_cracked_eval")
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--sub-region-frac", type=float, default=1.0,
                    help="Fraction of glass bbox to cover with cracks "
                         "(1.0 = full pane, matches synthetic training)")
    ap.add_argument("--blend-strength", type=float, default=1.0,
                    help="0..1 — intensity of the crack overlay")
    ap.add_argument("--alpha-gamma", type=float, default=0.55,
                    help="Gamma on crack alpha before blending (<1 boosts "
                         "faint pixels — counteracts bilinear-upscale blur)")
    ap.add_argument("--dh-max-side", type=int, default=1600,
                    help="Downscale DH photos + masks so max(H,W) == this "
                         "before compositing (matches 1280x720 synthetic "
                         "training scale, keeps cracks sharp). 0 disables.")
    ap.add_argument("--tiles-per-short-side", type=float, default=2.0,
                    help="How many crack-texture tiles span the mask's short "
                         "edge. Higher = finer cracks. Keeps textures near "
                         "native resolution instead of upscaling to fit.")
    ap.add_argument("--close-size", type=int, default=9,
                    help="Morphological-close size on the crack tile before "
                         "tiling (fills dark internal holes from the |diff| "
                         "extraction, so shatter reads as continuous).")
    ap.add_argument("--mode", choices=["ratio", "alpha"], default="ratio",
                    help="ratio = per-pixel per-channel modulation from a "
                         "training (cracked,intact) pair (matches training "
                         "distribution exactly). alpha = legacy tile-blend "
                         "using extracted crack_textures/.")
    ap.add_argument("--train-dir", default="data_glass/tanner_vlm_dataset_balanced/frames/train",
                    help="Directory with paired cracked_*/intact_* synthetic "
                         "training frames (used by --mode ratio).")
    ap.add_argument("--ratio-strength", type=float, default=1.0,
                    help="0..1 — blend toward identity (1.0 = full training "
                         "effect, 0.0 = no change). Used only in --mode ratio.")
    ap.add_argument("--jpeg-quality", type=int, default=92)
    args = ap.parse_args()

    intact_out = os.path.join(args.out, "intact")
    cracked_out = os.path.join(args.out, "cracked")
    os.makedirs(intact_out, exist_ok=True)
    os.makedirs(cracked_out, exist_ok=True)

    # Source data for the two modes.
    crack_paths = {}
    training_pairs = {}
    if args.mode == "ratio":
        training_pairs = index_training_pairs(args.train_dir)
        missing = [m for m in CRACK_MATERIALS if not training_pairs.get(m)]
        if missing:
            print(f"ERROR: no training pairs found for materials: {missing} "
                  f"under {args.train_dir}", file=sys.stderr)
            sys.exit(1)
        for mat in CRACK_MATERIALS:
            print(f"training pairs [{mat}]: {len(training_pairs[mat])}")
    else:
        for mat in CRACK_MATERIALS:
            d = os.path.join(args.crack_lib, mat)
            if not os.path.isdir(d):
                print(f"ERROR: missing crack lib dir: {d}", file=sys.stderr)
                sys.exit(1)
            crack_paths[mat] = sorted(
                os.path.join(d, f) for f in os.listdir(d) if f.endswith(".png")
            )
            if not crack_paths[mat]:
                print(f"ERROR: no PNGs under {d}", file=sys.stderr)
                sys.exit(1)

    # Discover (image, mask) pairs.
    image_files = sorted(
        f for f in os.listdir(args.src_images)
        if f.lower().endswith((".jpg", ".jpeg", ".png"))
    )
    rows = []
    missing_masks = []

    for idx, fname in enumerate(image_files):
        stem = os.path.splitext(fname)[0]
        safe_stem = stem.replace(" ", "_").replace("(", "").replace(")", "")
        src = os.path.join(args.src_images, fname)
        mask_path = os.path.join(args.masks, f"{stem}.png")
        if not os.path.exists(mask_path):
            # Also try the safe-stem variant in case the mask was named that way.
            alt = os.path.join(args.masks, f"{safe_stem}.png")
            if os.path.exists(alt):
                mask_path = alt
            else:
                missing_masks.append(stem)
                continue

        image = load_rgb(src)
        mask = load_gray(mask_path) > 127

        # Downscale DH photo + mask to match synthetic training resolution.
        # This is the single biggest lever for crack visibility — without it,
        # hairline textures get bilinear-blurred into invisible smudges when
        # stretched across a 5712x4284 region.
        if args.dh_max_side > 0:
            h, w = image.shape[:2]
            if max(h, w) > args.dh_max_side:
                scale = args.dh_max_side / max(h, w)
                new_w = int(round(w * scale))
                new_h = int(round(h * scale))
                image = np.asarray(
                    Image.fromarray(image).resize((new_w, new_h), Image.LANCZOS),
                    dtype=np.uint8,
                )
                mask = np.asarray(
                    Image.fromarray(mask.astype(np.uint8) * 255, mode="L")
                        .resize((new_w, new_h), Image.NEAREST),
                    dtype=np.uint8,
                ) > 127

        # Intact copy (under a safe, whitespace-free name).
        intact_dst = os.path.join(intact_out, f"{safe_stem}.jpg")
        Image.fromarray(image).save(intact_dst, quality=args.jpeg_quality)
        rows.append(jsonl_row(os.path.relpath(intact_dst, args.out), "intact"))

        for mi, mat in enumerate(CRACK_MATERIALS):
            # Deterministic seed so regenerations are reproducible.
            rng = random.Random(args.seed * 1000 + idx * 10 + mi)
            if args.mode == "ratio":
                cracked_path_tr, intact_path_tr = rng.choice(training_pairs[mat])
                ratio_rgb, src_quad = load_pair_to_ratio(
                    cracked_path_tr, intact_path_tr, material=mat,
                )
                composite = composite_ratio(
                    image, mask, ratio_rgb, src_quad, rng,
                    strength=args.ratio_strength,
                )
            else:
                crack_pick = rng.choice(crack_paths[mat])
                crack = load_gray(crack_pick)
                composite = composite_one(
                    image, mask, crack, rng,
                    sub_region_frac=args.sub_region_frac,
                    blend_strength=args.blend_strength,
                    alpha_gamma=args.alpha_gamma,
                    tiles_per_short_side=args.tiles_per_short_side,
                    close_size=args.close_size,
                )
            cracked_name = f"{safe_stem}__{mat}.jpg"
            cracked_path = os.path.join(cracked_out, cracked_name)
            Image.fromarray(composite).save(cracked_path, quality=args.jpeg_quality)
            rows.append(jsonl_row(os.path.relpath(cracked_path, args.out), "cracked"))

        print(f"[{idx+1}/{len(image_files)}] {fname}  ->  3 cracked composites")

    # Write test.jsonl.
    out_jsonl = os.path.join(args.out, "test.jsonl")
    with open(out_jsonl, "w") as f:
        for row in rows:
            f.write(json.dumps(row) + "\n")
        f.flush()
        os.fsync(f.fileno())

    if missing_masks:
        print(f"\nWARNING: {len(missing_masks)} images have no mask, skipped:")
        for s in missing_masks:
            print(f"   {s}")

    # README.
    n_intact = sum(1 for r in rows if "GLASS_INTACT" in r["conversations"][1]["content"])
    n_cracked = sum(1 for r in rows if "GLASS_CRACKED" in r["conversations"][1]["content"])
    readme = (
        "# DH cracked glass eval set\n\n"
        f"- Source photos: `{args.src_images}` ({len(image_files)} images)\n"
        f"- Glass masks: `{args.masks}`\n"
        f"- Crack textures: `{args.crack_lib}` (edge_v1, v4, tempered_v5)\n\n"
        f"## Contents\n- intact: {n_intact}\n- cracked: {n_cracked} "
        f"({n_cracked // 3} photos × 3 materials)\n\n"
        f"Prompt: `{PROMPT}`\n"
    )
    with open(os.path.join(args.out, "README.md"), "w") as f:
        f.write(readme)

    print(f"\nDone. {n_intact} intact + {n_cracked} cracked -> {out_jsonl}")


if __name__ == "__main__":
    main()
