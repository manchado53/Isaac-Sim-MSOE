#!/usr/bin/env python3
"""Build an MP4 of every dh_doors/ image with its CLOSED/OPEN label overlaid.

Used to visually spot-check the manual ground-truth labels before plugging
the test set into the sim-to-real eval.
"""

import argparse
import os
import re

import cv2

DOORS_DIR  = "data/dh_doors"
OUT_PATH   = "data/dh_doors_labels_check.mp4"
FPS        = 1            # 1 frame/sec — enough time to read each label
CANVAS_W   = 1024
CANVAS_H   = 1024

LABEL_RE = re.compile(r"IMG_(\d+)_(CLOSED|OPEN)\.jpg$")


def fit_to_canvas(img, w, h):
    ih, iw = img.shape[:2]
    scale = min(w / iw, h / ih)
    new_w, new_h = int(round(iw * scale)), int(round(ih * scale))
    resized = cv2.resize(img, (new_w, new_h), interpolation=cv2.INTER_AREA)
    canvas = cv2.copyMakeBorder(
        resized,
        (h - new_h) // 2, h - new_h - (h - new_h) // 2,
        (w - new_w) // 2, w - new_w - (w - new_w) // 2,
        cv2.BORDER_CONSTANT, value=(0, 0, 0),
    )
    return canvas


def draw_label(img, name, label):
    color = (0, 200, 0) if label == "OPEN" else (0, 80, 220)  # BGR
    cv2.rectangle(img, (0, 0), (img.shape[1], 90), (0, 0, 0), thickness=-1)
    cv2.putText(img, f"{name}   [{label}]", (20, 60),
                cv2.FONT_HERSHEY_SIMPLEX, 1.4, color, 3, cv2.LINE_AA)
    return img


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--doors_dir", default=DOORS_DIR)
    parser.add_argument("--out",       default=OUT_PATH)
    parser.add_argument("--fps",       type=float, default=FPS)
    args = parser.parse_args()

    files = sorted(f for f in os.listdir(args.doors_dir) if LABEL_RE.match(f))
    print(f"Found {len(files)} labeled frames in {args.doors_dir}")
    if not files:
        raise SystemExit("Nothing to render.")

    fourcc = cv2.VideoWriter_fourcc(*"mp4v")
    writer = cv2.VideoWriter(args.out, fourcc, args.fps, (CANVAS_W, CANVAS_H))

    n_open = n_closed = 0
    for fname in files:
        m = LABEL_RE.match(fname)
        idx, label = m.group(1), m.group(2)
        img = cv2.imread(os.path.join(args.doors_dir, fname))
        if img is None:
            print(f"  WARN: could not read {fname}")
            continue
        canvas = fit_to_canvas(img, CANVAS_W, CANVAS_H)
        canvas = draw_label(canvas, f"IMG_{idx}", label)
        writer.write(canvas)
        if label == "OPEN":
            n_open += 1
        else:
            n_closed += 1

    writer.release()
    print(f"Wrote {args.out}  ({n_open} OPEN, {n_closed} CLOSED, fps={args.fps})")


if __name__ == "__main__":
    main()
