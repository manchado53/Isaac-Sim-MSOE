#!/usr/bin/env python3
"""
segment_dh_panels.py

Segment the glass regions of Diercks Hall photos so Stage 3 can composite
synthetic crack textures onto glass only (not frames, mullions, sky).

Two modes:
  --propose   Runs SAM2 automatic mask generation on GPU, ranks proposals by a
              "glass-like-region" heuristic, and writes the top-K per image as
              a preview grid + a compact NPZ. Intended for SLURM GPU jobs.
  --review    Opens a Matplotlib UI (one figure per image) showing the top
              candidates from --propose. Keyboard keys:
                1..9 toggle candidate N into the accepted set
                space / enter accept current selection + next image
                d    draw a polygon manually (click vertices, Enter to close)
                x    clear current selection (forces re-choice)
                s    skip (no mask for this image)
                q    quit (save progress and exit)
              Accepted masks are written as data/dh_cracked_eval/masks/<stem>.png.

Typical usage:
  # 1) Propose candidates on a GPU node (SLURM):
  sbatch submit_sam2_propose.sh

  # 2) Review locally with X-forwarded matplotlib:
  python3 simulations/robot/segment_dh_panels.py --review \\
      --proposals data/dh_cracked_eval/proposals \\
      --out-masks data/dh_cracked_eval/masks
"""

import argparse
import json
import os
import sys
from dataclasses import dataclass
from typing import List, Optional

import numpy as np
from PIL import Image, ImageOps


def _load_rgb(path: str) -> np.ndarray:
    """Load an image as RGB with EXIF orientation applied.

    DH phone photos carry an EXIF Orientation tag (typically 6 = rotate 270 CW);
    without exif_transpose, SAM2 sees them sideways and masks end up mismatched
    with any viewer that honors the EXIF tag.
    """
    with Image.open(path) as im:
        im = ImageOps.exif_transpose(im).convert("RGB")
        return np.asarray(im, dtype=np.uint8)


# ---------- Glass-like region heuristic ------------------------------------

def glass_score(mask: np.ndarray, image_rgb: np.ndarray,
                sam_iou: float = 0.0, sam_stab: float = 0.0) -> dict:
    """Rank a candidate mask on how glass-pane-like it looks.

    Larger score = more glass-like. Glass in real photos is ambiguous — our
    DH shots range from close-ups where a single pane fills the frame to
    wider shots where glass is one of several rectangular regions. The only
    feature that is *robust* across both is "SAM2 is confident and the mask
    has a rectangular bounding box". Saturation / color uniformity is not
    reliable because glass reflects whatever is behind it (bright bikes,
    colored walls, etc.), which wrecks any color-based heuristic.
    """
    h, w = mask.shape
    area = int(mask.sum())
    img_area = h * w
    area_frac = area / img_area

    if area == 0:
        return dict(area_frac=0.0, bbox_fill=0.0, sam_iou=round(float(sam_iou), 4),
                    sam_stab=round(float(sam_stab), 4), score=-1.0)

    ys, xs = np.where(mask)
    y0, y1 = int(ys.min()), int(ys.max())
    x0, x1 = int(xs.min()), int(xs.max())
    bbox_area = max(1, (y1 - y0 + 1) * (x1 - x0 + 1))
    bbox_fill = area / bbox_area  # 1 = rectangular

    # Only penalize the near-full-frame "everything" mask.
    area_term = 1.0 if 0.05 <= area_frac <= 0.95 else max(0.0, 1.0 - (area_frac - 0.95) * 10.0)
    score = (
        0.6 * float(sam_iou)
        + 0.3 * float(sam_stab)
        + 0.4 * bbox_fill
        + 0.3 * area_term
    )
    return dict(
        area_frac=round(area_frac, 4),
        bbox_fill=round(bbox_fill, 4),
        sam_iou=round(float(sam_iou), 4),
        sam_stab=round(float(sam_stab), 4),
        score=round(float(score), 4),
        bbox_xyxy=(x0, y0, x1 + 1, y1 + 1),
    )


# ---------- Proposal stage (GPU) -------------------------------------------

def _lazy_sam2():
    import torch
    from sam2.build_sam import build_sam2
    from sam2.automatic_mask_generator import SAM2AutomaticMaskGenerator
    return torch, build_sam2, SAM2AutomaticMaskGenerator


def build_mask_generator(checkpoint: str, cfg: str, device: str,
                         points_per_side: int = 24,
                         points_per_batch: int = 16):
    torch, build_sam2, SAM2AutomaticMaskGenerator = _lazy_sam2()
    model = build_sam2(cfg, checkpoint, device=device, apply_postprocessing=True)
    mg = SAM2AutomaticMaskGenerator(
        model=model,
        points_per_side=points_per_side,
        points_per_batch=points_per_batch,
        pred_iou_thresh=0.7,
        stability_score_thresh=0.9,
        box_nms_thresh=0.7,
        crop_n_layers=0,
        min_mask_region_area=2000,
    )
    return mg


def _resize_to_max_side(image: np.ndarray, max_side: int) -> tuple:
    """Return (resized_image, scale) where resized longest-side <= max_side.

    scale is the factor used (new = old * scale); pass <1 to enlarge back.
    """
    h, w = image.shape[:2]
    long = max(h, w)
    if long <= max_side:
        return image, 1.0
    scale = max_side / long
    new_w = int(round(w * scale))
    new_h = int(round(h * scale))
    pil = Image.fromarray(image).resize((new_w, new_h), Image.LANCZOS)
    return np.asarray(pil, dtype=np.uint8), scale


def run_propose(args):
    torch, _, _ = _lazy_sam2()
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"[propose] device={device}  checkpoint={args.checkpoint}  cfg={args.cfg}"
          f"  max_side={args.max_side}  points_per_side={args.points_per_side}"
          f"  points_per_batch={args.points_per_batch}")
    mg = build_mask_generator(
        args.checkpoint, args.cfg, device,
        points_per_side=args.points_per_side,
        points_per_batch=args.points_per_batch,
    )

    src_paths = sorted(
        os.path.join(args.src, f) for f in os.listdir(args.src)
        if f.lower().endswith((".jpg", ".jpeg", ".png"))
    )
    if not src_paths:
        print(f"[propose] no images in {args.src}", file=sys.stderr)
        sys.exit(1)

    os.makedirs(args.out, exist_ok=True)
    for i, path in enumerate(src_paths, 1):
        stem = os.path.splitext(os.path.basename(path))[0]
        out_npz = os.path.join(args.out, f"{stem}.npz")
        out_preview = os.path.join(args.out, f"{stem}_preview.jpg")
        if os.path.exists(out_npz) and not args.overwrite:
            print(f"[propose][{i}/{len(src_paths)}] skip (exists): {stem}")
            continue

        full = _load_rgb(path)
        small, scale = _resize_to_max_side(full, args.max_side)
        masks = mg.generate(small)  # list of dicts at small resolution
        # Rank and keep top-K (score at small res for speed; bbox is rescaled).
        scored = []
        for m in masks:
            seg = m["segmentation"].astype(bool)
            sam_iou = float(m.get("predicted_iou", 0.0))
            sam_stab = float(m.get("stability_score", 0.0))
            s = glass_score(seg, small, sam_iou=sam_iou, sam_stab=sam_stab)
            scored.append((s, seg, sam_iou, sam_stab))
        scored.sort(key=lambda t: t[0]["score"], reverse=True)
        top = scored[: args.top_k]

        if not top:
            print(f"[propose][{i}/{len(src_paths)}] NO MASKS: {stem}")
            torch.cuda.empty_cache() if device == "cuda" else None
            continue

        # Rescale masks back up to full resolution via nearest-neighbor.
        full_h, full_w = full.shape[:2]
        rescaled = []
        sam_scores = []
        for _, seg, sam_iou, sam_stab in top:
            m_pil = Image.fromarray((seg.astype(np.uint8) * 255), mode="L")
            m_pil = m_pil.resize((full_w, full_h), Image.NEAREST)
            rescaled.append(np.asarray(m_pil, dtype=np.uint8) > 0)
            sam_scores.append((sam_iou, sam_stab))
        stack = np.stack([r.astype(np.uint8) for r in rescaled], axis=0)

        # Recompute meta (score, bbox) at full resolution for the UI labels.
        full_meta = [
            glass_score(r, full, sam_iou=si, sam_stab=ss)
            for r, (si, ss) in zip(rescaled, sam_scores)
        ]

        np.savez_compressed(out_npz, masks=stack,
                            meta=np.array(json.dumps(full_meta), dtype="U"),
                            image_shape=np.array(full.shape, dtype=np.int32),
                            scale=np.float32(scale))
        _save_preview(full, stack, full_meta, out_preview)
        print(f"[propose][{i}/{len(src_paths)}] {stem}  kept {len(top)} candidates"
              f"  top score={full_meta[0]['score']:.3f}  scale={scale:.3f}")

        if device == "cuda":
            torch.cuda.empty_cache()

    print(f"[propose] done. proposals in {args.out}")


def _save_preview(image: np.ndarray, masks: np.ndarray, meta: list, out: str):
    """Save a grid preview JPG: original + each candidate overlaid."""
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    k = masks.shape[0]
    cols = min(4, k + 1)
    rows = (k + 1 + cols - 1) // cols
    fig, axes = plt.subplots(rows, cols, figsize=(cols * 4, rows * 3))
    axes = np.atleast_2d(axes).ravel()

    axes[0].imshow(image)
    axes[0].set_title("source")
    axes[0].axis("off")

    for i in range(k):
        overlay = image.copy()
        m = masks[i].astype(bool)
        overlay[m] = (overlay[m] * 0.4 + np.array([255, 60, 60]) * 0.6).astype(np.uint8)
        axes[i + 1].imshow(overlay)
        s = meta[i]
        axes[i + 1].set_title(
            f"{i+1}. score {s['score']:.2f}\narea {s['area_frac']*100:.1f}%  "
            f"bbox {s['bbox_fill']:.2f}  iou {s.get('sam_iou', 0):.2f}  "
            f"stab {s.get('sam_stab', 0):.2f}",
            fontsize=8,
        )
        axes[i + 1].axis("off")
    for j in range(k + 1, len(axes)):
        axes[j].axis("off")
    fig.tight_layout()
    fig.savefig(out, dpi=90, bbox_inches="tight")
    plt.close(fig)


# ---------- Review stage (interactive) -------------------------------------

@dataclass
class ReviewState:
    img_path: str
    image: np.ndarray
    cands: np.ndarray  # (k, h, w) uint8
    meta: list
    selected: List[int]  # indices into cands currently toggled on
    manual_polys: List[list]  # list of [(x,y), ...]


def run_review(args):
    import matplotlib
    import matplotlib.pyplot as plt
    from matplotlib.widgets import PolygonSelector
    from matplotlib.path import Path as MplPath

    os.makedirs(args.out_masks, exist_ok=True)

    proposal_files = sorted(
        f for f in os.listdir(args.proposals) if f.endswith(".npz")
    )
    if not proposal_files:
        print(f"[review] no .npz proposals in {args.proposals}", file=sys.stderr)
        sys.exit(1)

    # Map stem -> source image path (for display quality).
    def find_src(stem: str) -> Optional[str]:
        for ext in (".jpg", ".JPG", ".jpeg", ".png"):
            p = os.path.join(args.src, stem + ext)
            if os.path.isfile(p):
                return p
        return None

    progress_path = os.path.join(args.out_masks, "_progress.json")
    progress = {}
    if os.path.isfile(progress_path):
        with open(progress_path) as f:
            progress = json.load(f)

    for f in proposal_files:
        stem = f[: -len(".npz")]
        out_mask = os.path.join(args.out_masks, f"{stem}.png")
        if os.path.exists(out_mask) and not args.re_review:
            print(f"[review] skip (already reviewed): {stem}")
            continue

        src_path = find_src(stem)
        if src_path is None:
            print(f"[review] missing source for {stem}, skipping")
            continue

        data = np.load(os.path.join(args.proposals, f), allow_pickle=True)
        cands = data["masks"]
        meta = json.loads(str(data["meta"]))
        image = _load_rgb(src_path)

        state = ReviewState(
            img_path=src_path, image=image, cands=cands, meta=meta,
            selected=[0] if len(cands) > 0 else [],  # top-ranked preselected
            manual_polys=[],
        )

        action = _review_one(state, plt, MplPath)
        if action == "quit":
            _write_progress(progress_path, progress)
            print(f"[review] saved progress, exiting at {stem}")
            return

        if action == "skip":
            progress[stem] = {"skipped": True}
            continue

        mask = _compose_final_mask(state)
        if mask.sum() == 0:
            print(f"[review] empty mask for {stem}, skipping save")
            progress[stem] = {"empty": True}
            continue

        Image.fromarray((mask * 255).astype(np.uint8), mode="L").save(out_mask)
        progress[stem] = {
            "selected": state.selected,
            "manual_polys": len(state.manual_polys),
            "mask_px": int(mask.sum()),
        }
        _write_progress(progress_path, progress)
        print(f"[review] saved {out_mask}  ({int(mask.sum())} px)")

    _write_progress(progress_path, progress)
    print(f"[review] done. masks in {args.out_masks}")


def _write_progress(path: str, progress: dict):
    with open(path, "w") as f:
        json.dump(progress, f, indent=2)


def _compose_final_mask(state: ReviewState) -> np.ndarray:
    h, w = state.image.shape[:2]
    out = np.zeros((h, w), dtype=bool)
    for idx in state.selected:
        out |= state.cands[idx].astype(bool)
    for poly in state.manual_polys:
        if len(poly) < 3:
            continue
        from matplotlib.path import Path as MplPath
        yy, xx = np.mgrid[:h, :w]
        points = np.column_stack((xx.ravel(), yy.ravel()))
        inside = MplPath(poly).contains_points(points).reshape(h, w)
        out |= inside
    return out


def _review_one(state: ReviewState, plt, MplPath) -> str:
    """Show the figure and block until user presses an action key.

    Returns one of: "next", "skip", "quit".
    """
    from matplotlib.widgets import PolygonSelector

    fig, ax = plt.subplots(figsize=(10, 7))

    def redraw():
        ax.clear()
        overlay = state.image.copy()
        # Blend all currently-selected candidates in green.
        union = np.zeros(state.image.shape[:2], dtype=bool)
        for idx in state.selected:
            union |= state.cands[idx].astype(bool)
        for poly in state.manual_polys:
            if len(poly) >= 3:
                yy, xx = np.mgrid[:overlay.shape[0], :overlay.shape[1]]
                points = np.column_stack((xx.ravel(), yy.ravel()))
                inside = MplPath(poly).contains_points(points).reshape(overlay.shape[:2])
                union |= inside
        overlay[union] = (overlay[union] * 0.45 + np.array([40, 220, 40]) * 0.55).astype(np.uint8)
        ax.imshow(overlay)
        stem = os.path.splitext(os.path.basename(state.img_path))[0]
        sel_str = ",".join(str(i + 1) for i in state.selected) if state.selected else "none"
        ax.set_title(
            f"{stem}   selected={sel_str}   manual_polys={len(state.manual_polys)}\n"
            f"Keys: [1-9] toggle cand  [d] draw poly  [x] clear  [space] next  "
            f"[s] skip  [q] quit",
            fontsize=10,
        )
        ax.axis("off")
        fig.canvas.draw_idle()

    action = {"value": "next"}

    def on_key(event):
        key = event.key
        if key in {"1", "2", "3", "4", "5", "6", "7", "8", "9"}:
            i = int(key) - 1
            if i < len(state.cands):
                if i in state.selected:
                    state.selected.remove(i)
                else:
                    state.selected.append(i)
                redraw()
        elif key == "x":
            state.selected.clear()
            state.manual_polys.clear()
            redraw()
        elif key == "d":
            # Start polygon drawing: user clicks vertices, Enter finishes.
            ax.set_title("Drawing polygon: click vertices, Esc cancels. Dbl-click or Enter to finish.",
                         fontsize=10)
            fig.canvas.draw_idle()
            poly = []

            def on_select(verts):
                poly.extend([(float(x), float(y)) for x, y in verts])

            selector = PolygonSelector(ax, on_select, useblit=True)
            # Block until the selector completes — simple synchronous loop.
            while not poly and plt.fignum_exists(fig.number):
                plt.pause(0.1)
            try:
                selector.disconnect_events()
            except Exception:
                pass
            if poly:
                state.manual_polys.append(poly)
            redraw()
        elif key in {" ", "enter"}:
            action["value"] = "next"
            plt.close(fig)
        elif key == "s":
            action["value"] = "skip"
            plt.close(fig)
        elif key == "q":
            action["value"] = "quit"
            plt.close(fig)

    fig.canvas.mpl_connect("key_press_event", on_key)
    redraw()
    plt.show()
    return action["value"]


# ---------- Export / Finalize (no-UI manual review) ------------------------

def _overlay_image(image: np.ndarray, mask: np.ndarray,
                   rgb=(40, 220, 40), alpha=0.55) -> np.ndarray:
    out = image.copy()
    m = mask.astype(bool)
    out[m] = (out[m] * (1 - alpha) + np.array(rgb, dtype=np.float32) * alpha).astype(np.uint8)
    return out


def run_export(args):
    """Dump per-candidate overlay JPGs to review/<stem>/cand_XX.jpg.

    Workflow:
      1. Run this once -> populates review/<stem>/ with up to 20 cand_XX.jpg per image
         plus _source.jpg for reference.
      2. Browse each folder, DELETE the overlays that don't cover the glass correctly.
      3. Run `finalize` -> OR the surviving candidates into data/dh_cracked_eval/masks/<stem>.png.

    If NO overlay covers the glass for an image, leave the folder with only _source.jpg
    (delete every cand_*). Finalize will skip empty folders; paint a mask by hand with
    an image editor and drop <stem>.png into the masks/ dir.
    """
    proposal_files = sorted(
        f for f in os.listdir(args.proposals) if f.endswith(".npz")
    )
    if not proposal_files:
        print(f"[export] no .npz proposals in {args.proposals}", file=sys.stderr)
        sys.exit(1)
    os.makedirs(args.out, exist_ok=True)

    def find_src(stem: str) -> Optional[str]:
        for ext in (".jpg", ".JPG", ".jpeg", ".png"):
            p = os.path.join(args.src, stem + ext)
            if os.path.isfile(p):
                return p
        return None

    for fn in proposal_files:
        stem = fn[: -len(".npz")]
        src_path = find_src(stem)
        if src_path is None:
            print(f"[export] missing source for {stem}, skipping")
            continue

        data = np.load(os.path.join(args.proposals, fn), allow_pickle=True)
        cands = data["masks"]  # (k, H, W) uint8
        meta = json.loads(str(data["meta"]))
        image = _load_rgb(src_path)

        img_dir = os.path.join(args.out, stem)
        os.makedirs(img_dir, exist_ok=True)

        # Reference source copy.
        Image.fromarray(image).save(os.path.join(img_dir, "_source.jpg"), quality=88)

        for i in range(cands.shape[0]):
            overlay = _overlay_image(image, cands[i])
            s = meta[i] if i < len(meta) else {}
            score = s.get("score", 0.0)
            area = s.get("area_frac", 0.0)
            name = f"cand_{i+1:02d}__score{score:.2f}_area{area*100:.1f}pct.jpg"
            Image.fromarray(overlay).save(os.path.join(img_dir, name), quality=85)
        print(f"[export] {stem}  -> {img_dir} ({cands.shape[0]} candidates)")

    print(f"\n[export] done. Review manually:")
    print(f"  1. Browse {args.out}/<stem>/ and DELETE the candidate files whose green")
    print(f"     overlay does NOT match the glass pane.")
    print(f"  2. Run: python3 {os.path.basename(__file__)} finalize \\")
    print(f"            --proposals {args.proposals} \\")
    print(f"            --review {args.out} \\")
    print(f"            --out-masks data/dh_cracked_eval/masks")


_CAND_RE = __import__("re").compile(r"^cand_(\d{2})(?:__|\.)")


def run_finalize(args):
    """Scan review/<stem>/cand_XX*.jpg -> OR kept masks into out-masks/<stem>.png."""
    if not os.path.isdir(args.review):
        print(f"[finalize] review dir not found: {args.review}", file=sys.stderr)
        sys.exit(1)
    os.makedirs(args.out_masks, exist_ok=True)

    stems = sorted(
        d for d in os.listdir(args.review)
        if os.path.isdir(os.path.join(args.review, d))
    )
    if not stems:
        print(f"[finalize] no per-image folders in {args.review}", file=sys.stderr)
        sys.exit(1)

    summary = {"saved": [], "empty": [], "missing_proposal": []}
    for stem in stems:
        folder = os.path.join(args.review, stem)
        kept_indices = []
        for f in sorted(os.listdir(folder)):
            m = _CAND_RE.match(f)
            if m:
                kept_indices.append(int(m.group(1)) - 1)  # 1-based filenames -> 0-based idx

        npz_path = os.path.join(args.proposals, f"{stem}.npz")
        if not os.path.isfile(npz_path):
            print(f"[finalize] no proposals NPZ for {stem}, skipping")
            summary["missing_proposal"].append(stem)
            continue

        if not kept_indices:
            print(f"[finalize] {stem}: no surviving candidates (paint mask by hand if needed)")
            summary["empty"].append(stem)
            continue

        data = np.load(npz_path, allow_pickle=True)
        cands = data["masks"]
        h, w = cands.shape[1], cands.shape[2]
        union = np.zeros((h, w), dtype=bool)
        valid = []
        for idx in kept_indices:
            if 0 <= idx < cands.shape[0]:
                union |= cands[idx].astype(bool)
                valid.append(idx + 1)
        if union.sum() == 0:
            print(f"[finalize] {stem}: union is empty, skipping")
            summary["empty"].append(stem)
            continue

        out_path = os.path.join(args.out_masks, f"{stem}.png")
        Image.fromarray((union * 255).astype(np.uint8), mode="L").save(out_path)
        print(f"[finalize] {stem}: kept {valid} -> {out_path} ({int(union.sum())} px)")
        summary["saved"].append(stem)

    summary_path = os.path.join(args.out_masks, "_finalize_summary.json")
    with open(summary_path, "w") as f:
        json.dump(summary, f, indent=2)
    print(f"\n[finalize] {len(summary['saved'])} masks saved, "
          f"{len(summary['empty'])} empty, "
          f"{len(summary['missing_proposal'])} missing proposals")
    print(f"[finalize] summary -> {summary_path}")


# ---------- main -----------------------------------------------------------

def main():
    ap = argparse.ArgumentParser()
    sub = ap.add_subparsers(dest="mode", required=True)

    pp = sub.add_parser("propose", help="Run SAM2 to generate candidate masks (GPU)")
    pp.add_argument("--src", default="dh_glass_panels",
                    help="Directory of DH panel JPGs")
    pp.add_argument("--out", default="data/dh_cracked_eval/proposals",
                    help="Output dir for per-image proposal NPZs + preview JPGs")
    pp.add_argument("--checkpoint", default="data/sam2_checkpoints/sam2.1_hiera_large.pt")
    pp.add_argument("--cfg", default="configs/sam2.1/sam2.1_hiera_l.yaml",
                    help="SAM2 config path (relative to sam2 package)")
    pp.add_argument("--top-k", type=int, default=20)
    pp.add_argument("--max-side", type=int, default=1280,
                    help="Resize input to this longest-side before SAM2 (VRAM-bound on T4)")
    pp.add_argument("--points-per-side", type=int, default=24)
    pp.add_argument("--points-per-batch", type=int, default=16)
    pp.add_argument("--overwrite", action="store_true")

    rv = sub.add_parser("review", help="Interactive accept/reject of proposals")
    rv.add_argument("--src", default="dh_glass_panels")
    rv.add_argument("--proposals", default="data/dh_cracked_eval/proposals")
    rv.add_argument("--out-masks", default="data/dh_cracked_eval/masks")
    rv.add_argument("--re-review", action="store_true",
                    help="Re-open masks that already exist")

    ex = sub.add_parser("export", help="Dump per-candidate overlays for manual review (no UI)")
    ex.add_argument("--src", default="dh_glass_panels")
    ex.add_argument("--proposals", default="data/dh_cracked_eval/proposals")
    ex.add_argument("--out", default="data/dh_cracked_eval/review",
                    help="review/<stem>/cand_XX.jpg - delete bad candidates, keep good ones")

    fz = sub.add_parser("finalize", help="Read surviving overlays in review/<stem>/ -> write masks")
    fz.add_argument("--proposals", default="data/dh_cracked_eval/proposals")
    fz.add_argument("--review", default="data/dh_cracked_eval/review")
    fz.add_argument("--out-masks", default="data/dh_cracked_eval/masks")

    args = ap.parse_args()
    if args.mode == "propose":
        run_propose(args)
    elif args.mode == "review":
        run_review(args)
    elif args.mode == "export":
        run_export(args)
    elif args.mode == "finalize":
        run_finalize(args)
    else:
        ap.error(f"unknown mode: {args.mode}")


if __name__ == "__main__":
    main()
