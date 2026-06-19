#!/usr/bin/env python3
"""Render a 4x4 contact-sheet of the DH door dataset for slides.

Hand-picked 16 frames covering:
  - 5 different doors (Jensen 444, Cohen 327, Johnson 442, Cohen 323, Room 443/453, North Stair)
  - both states (8 CLOSED, 8 OPEN)
  - varied open angles (30°, 45°, 90° edge view, full swing)
"""

import os
import re

import cv2
import numpy as np

DOORS_DIR = "data/dh_doors"
OUT_PATH  = "data/dh_doors_grid_4x4.png"

# 4 rows x 4 cols, top-to-bottom left-to-right
SELECTION = [
    # Row 1: CLOSED — flush head-on, 4 different doors
    "IMG_5475_CLOSED.jpg",   # Jensen 444 (cream wall, gold plaque)
    "IMG_5495_CLOSED.jpg",   # Cohen 327 (blue wall)
    "IMG_5513_CLOSED.jpg",   # Johnson Controls 442 (white wall)
    "IMG_5545_CLOSED.jpg",   # Cohen 323 wooden door
    # Row 2: OPEN — partial swings, doors from row 1
    "IMG_5478_OPEN.jpg",     # Jensen 444 ~30° open, person at desk
    "IMG_5500_OPEN.jpg",     # Cohen 327 ~70° open, hallway
    "IMG_5519_OPEN.jpg",     # Johnson 442 open, classroom inside
    "IMG_5535_OPEN.jpg",     # Cohen 323 ~45° open
    # Row 3: CLOSED — different door variants
    "IMG_5546_CLOSED.jpg",   # Room 443 (yellow accent wall)
    "IMG_5557_CLOSED.jpg",   # Room 453 (white wall)
    "IMG_5561_CLOSED.jpg",   # North Stair fire-exit (EXIT sign, push bar)
    "IMG_5566_CLOSED.jpg",   # North Stair fire-exit, off-angle
    # Row 4: OPEN — wider swings
    "IMG_5491_OPEN.jpg",     # Jensen 444 wide open, two students
    "IMG_5537_OPEN.jpg",     # Cohen 323 ~90° edge view
    "IMG_5567_OPEN.jpg",     # Cohen 323 wide open, meeting room
    "IMG_5577_OPEN.jpg",     # Cohen 323 ~75° angle
]

ROWS, COLS = 4, 4
TILE_W, TILE_H = 480, 640        # portrait orientation matches iPhone source
PAD = 8                          # gap between tiles
BG = (20, 20, 20)                # near-black background


def fit(img, w, h):
    ih, iw = img.shape[:2]
    s = min(w / iw, h / ih)
    nw, nh = int(round(iw * s)), int(round(ih * s))
    r = cv2.resize(img, (nw, nh), interpolation=cv2.INTER_AREA)
    canvas = np.full((h, w, 3), BG, dtype=np.uint8)
    y0, x0 = (h - nh) // 2, (w - nw) // 2
    canvas[y0:y0 + nh, x0:x0 + nw] = r
    return canvas


def label_badge(tile, label):
    color = (0, 200, 0) if label == "OPEN" else (0, 60, 220)  # BGR
    h, w = tile.shape[:2]
    box_w, box_h = 130, 36
    x0, y0 = w - box_w - 8, 8
    overlay = tile.copy()
    cv2.rectangle(overlay, (x0, y0), (x0 + box_w, y0 + box_h), color, thickness=-1)
    cv2.addWeighted(overlay, 0.85, tile, 0.15, 0, tile)
    cv2.putText(tile, label, (x0 + 12, y0 + 26),
                cv2.FONT_HERSHEY_SIMPLEX, 0.75, (255, 255, 255), 2, cv2.LINE_AA)
    return tile


def main():
    grid_w = COLS * TILE_W + (COLS + 1) * PAD
    grid_h = ROWS * TILE_H + (ROWS + 1) * PAD
    grid = np.full((grid_h, grid_w, 3), BG, dtype=np.uint8)

    label_re = re.compile(r"_(CLOSED|OPEN)\.jpg$")

    for i, fname in enumerate(SELECTION):
        path = os.path.join(DOORS_DIR, fname)
        img = cv2.imread(path)
        if img is None:
            raise SystemExit(f"missing {path}")
        m = label_re.search(fname)
        label = m.group(1) if m else "?"
        tile = fit(img, TILE_W, TILE_H)
        tile = label_badge(tile, label)
        r, c = i // COLS, i % COLS
        y = PAD + r * (TILE_H + PAD)
        x = PAD + c * (TILE_W + PAD)
        grid[y:y + TILE_H, x:x + TILE_W] = tile

    cv2.imwrite(OUT_PATH, grid, [cv2.IMWRITE_PNG_COMPRESSION, 6])
    print(f"Wrote {OUT_PATH}  ({grid_w}x{grid_h}, {len(SELECTION)} tiles)")


if __name__ == "__main__":
    main()
