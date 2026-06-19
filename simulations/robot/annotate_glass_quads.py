#!/usr/bin/env python3
"""
annotate_glass_quads.py

Quick manual annotator for DH glass panels. Replaces the SAM2-then-review
pipeline with direct human clicks — SAM2 can't reliably segment transparent
surfaces, and for ~30 photos a human with a mouse is faster and strictly
better.

Workflow (run on a node with a display — Rosie OOD Desktop):

    python3 simulations/robot/annotate_glass_quads.py \\
        --src dh_glass_panels \\
        --out data/dh_cracked_eval/masks

For each photo, left-click 4 corners of one glass pane (in any order — the
tool sorts them into a convex quad before rasterizing). Keys:

    space / enter  save the current quad, move on to the next image
                   (if you added multiple quads with `n`, all of them are
                   saved into one binary mask)
    n              start another quad on the same image (multi-pane shots)
    u              undo the last clicked point
    c              clear all points + unsaved quads on this image
    s              skip this image (no mask written)
    q              quit, save progress, can be resumed later

Output: <out>/<stem>.png — binary L-mode PNG at full EXIF-oriented resolution
(255 where glass, 0 elsewhere). Matches what Stage 3 (composite_cracked_glass.py)
expects.
"""

import argparse
import json
import os
import sys

import numpy as np
from PIL import Image, ImageDraw, ImageOps


def load_rgb(path: str) -> np.ndarray:
    """EXIF-oriented RGB load — must match composite_cracked_glass.py."""
    with Image.open(path) as im:
        im = ImageOps.exif_transpose(im).convert("RGB")
        return np.asarray(im, dtype=np.uint8)


def sort_quad_vertices(pts):
    """Sort 4 (x, y) points into top-left, top-right, bottom-right, bottom-left order.

    Using angle-from-centroid sort so the polygon is always drawn as a
    non-self-intersecting convex quad regardless of click order.
    """
    pts = np.asarray(pts, dtype=np.float32)
    cx, cy = pts.mean(axis=0)
    angles = np.arctan2(pts[:, 1] - cy, pts[:, 0] - cx)
    return [tuple(p) for p in pts[np.argsort(angles)]]


def rasterize_quads(h: int, w: int, quads):
    """Fill each quad into a binary PIL L-mode image (255 inside, 0 outside)."""
    mask = Image.new("L", (w, h), 0)
    draw = ImageDraw.Draw(mask)
    for q in quads:
        if len(q) < 3:
            continue
        pts = sort_quad_vertices(q) if len(q) == 4 else list(q)
        draw.polygon(pts, fill=255)
    return np.asarray(mask, dtype=np.uint8)


def annotate_one(path: str, save_path: str, plt) -> str:
    """Return one of: "saved", "skipped", "quit"."""
    image = load_rgb(path)
    h, w = image.shape[:2]

    fig, ax = plt.subplots(figsize=(12, 8))
    ax.imshow(image)
    stem = os.path.splitext(os.path.basename(path))[0]

    state = {
        "points": [],        # current unfinished quad
        "saved_quads": [],   # completed quads (list of 4-tuple lists)
        "action": None,
    }

    scatter = ax.scatter([], [], s=80, c="lime", edgecolors="black", linewidths=1.5, zorder=5)
    poly_artists = []

    def redraw_title():
        n_points = len(state["points"])
        n_saved = len(state["saved_quads"])
        ax.set_title(
            f"{stem}   [{n_saved} saved quads, {n_points}/4 points on current]\n"
            f"Click 4 corners of a glass pane. "
            f"[n] new quad  [u] undo  [c] clear  [space] save+next  [s] skip  [q] quit",
            fontsize=10,
        )

    def refresh():
        # Update the scatter for the in-progress points.
        if state["points"]:
            xs, ys = zip(*state["points"])
            scatter.set_offsets(np.column_stack([xs, ys]))
        else:
            scatter.set_offsets(np.empty((0, 2)))
        # Redraw saved quads in translucent green.
        for a in poly_artists:
            a.remove()
        poly_artists.clear()
        for q in state["saved_quads"]:
            verts = sort_quad_vertices(q) + [sort_quad_vertices(q)[0]]
            xs = [v[0] for v in verts]
            ys = [v[1] for v in verts]
            line, = ax.plot(xs, ys, "-", color="lime", linewidth=2, alpha=0.9)
            fill = ax.fill(xs, ys, color="lime", alpha=0.25)[0]
            poly_artists.extend([line, fill])
        redraw_title()
        fig.canvas.draw_idle()

    def on_click(event):
        if event.inaxes != ax or event.button != 1:
            return
        if event.xdata is None or event.ydata is None:
            return
        state["points"].append((float(event.xdata), float(event.ydata)))
        if len(state["points"]) == 4:
            state["saved_quads"].append(state["points"])
            state["points"] = []
        refresh()

    def on_key(event):
        key = event.key
        if key == "u":
            if state["points"]:
                state["points"].pop()
            elif state["saved_quads"]:
                # Pull the last complete quad back into in-progress points so
                # the user can tweak one corner.
                state["points"] = list(state["saved_quads"].pop())
            refresh()
        elif key == "c":
            state["points"] = []
            state["saved_quads"] = []
            refresh()
        elif key == "n":
            # Force-finish the current quad even if it has <4 points
            # (triangle is OK — polygon fill still works).
            if len(state["points"]) >= 3:
                state["saved_quads"].append(state["points"])
                state["points"] = []
                refresh()
        elif key == "s":
            state["action"] = "skipped"
            plt.close(fig)
        elif key == "q":
            state["action"] = "quit"
            plt.close(fig)
        elif key in {" ", "enter"}:
            # Also accept a partially-finished in-progress quad (3+ points).
            if len(state["points"]) >= 3:
                state["saved_quads"].append(state["points"])
                state["points"] = []
            if not state["saved_quads"]:
                # No quads drawn — treat as skip.
                state["action"] = "skipped"
            else:
                state["action"] = "saved"
            plt.close(fig)

    fig.canvas.mpl_connect("button_press_event", on_click)
    fig.canvas.mpl_connect("key_press_event", on_key)
    refresh()
    plt.show()

    if state["action"] == "saved":
        mask = rasterize_quads(h, w, state["saved_quads"])
        Image.fromarray(mask, mode="L").save(save_path)
        px = int((mask > 0).sum())
        print(f"[saved] {stem}: {len(state['saved_quads'])} quads -> {save_path} ({px} px)")
        return "saved"
    if state["action"] == "skipped":
        print(f"[skip]  {stem}")
        return "skipped"
    return "quit"


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--src", default="dh_glass_panels")
    ap.add_argument("--out", default="data/dh_cracked_eval/masks")
    ap.add_argument("--redo", action="store_true",
                    help="Re-open images whose masks already exist")
    args = ap.parse_args()

    os.makedirs(args.out, exist_ok=True)
    paths = sorted(
        os.path.join(args.src, f) for f in os.listdir(args.src)
        if f.lower().endswith((".jpg", ".jpeg", ".png"))
    )
    if not paths:
        print(f"no images in {args.src}", file=sys.stderr)
        sys.exit(1)

    # Guard against silent headless-backend pass-through (plt.show() returns
    # immediately with no display). Only reject genuinely non-interactive
    # backends — TkAgg / QtAgg etc. all legitimately contain "agg" in the name
    # but ARE interactive.
    import matplotlib
    backend = matplotlib.get_backend().lower()
    NON_INTERACTIVE = {"agg", "cairo", "pdf", "pgf", "ps", "svg", "template"}
    if backend in NON_INTERACTIVE:
        print(f"ERROR: matplotlib backend is {backend!r} (non-interactive). "
              f"Run this on a node with a display (Rosie OOD Desktop).",
              file=sys.stderr)
        sys.exit(2)
    if not os.environ.get("DISPLAY") and sys.platform != "darwin":
        print(f"ERROR: $DISPLAY is not set. Run this on a node with a display.",
              file=sys.stderr)
        sys.exit(2)
    import matplotlib.pyplot as plt

    progress_path = os.path.join(args.out, "_annotate_progress.json")
    progress = {}
    if os.path.isfile(progress_path):
        with open(progress_path) as f:
            progress = json.load(f)

    total = len(paths)
    done = 0
    skipped = 0
    for i, path in enumerate(paths, 1):
        stem = os.path.splitext(os.path.basename(path))[0]
        save_path = os.path.join(args.out, f"{stem}.png")
        if os.path.exists(save_path) and not args.redo:
            print(f"[exists] {stem}  (skip — pass --redo to re-open)")
            done += 1
            continue

        print(f"--- [{i}/{total}] {stem} ---")
        result = annotate_one(path, save_path, plt)
        progress[stem] = {"result": result}
        with open(progress_path, "w") as f:
            json.dump(progress, f, indent=2)
            f.flush()
            os.fsync(f.fileno())

        if result == "saved":
            done += 1
        elif result == "skipped":
            skipped += 1
        elif result == "quit":
            print(f"\n[quit] {done} saved, {skipped} skipped, "
                  f"{total - done - skipped} remaining.")
            return

    print(f"\n[done] {done} masks in {args.out}  ({skipped} skipped)")


if __name__ == "__main__":
    main()
