#!/usr/bin/env python3
from PIL import Image, ImageDraw, ImageFont
from pathlib import Path

ROOT = Path("/home/ad.msoe.edu/celliot/Digital-twin-v2/isaac-sim-msoe/data/dh_cracked_eval/balanced_100_labeled")
OUT = Path("/home/ad.msoe.edu/celliot/Digital-twin-v2/isaac-sim-msoe/glass_grid.jpg")

# 2x2: each row is the same scene, broken (left) vs intact (right)
selected = [
    ("BROKEN", "cracked__Image_31__v4.jpg"), ("INTACT", "intact__Image_31.jpg"),
    ("BROKEN", "cracked__Image_26__tempered_v5.jpg"), ("INTACT", "intact__Image_26.jpg"),
]

CELL_W, CELL_H = 700, 933   # portrait cells (images are ~3:4)
GAP = 6
BG = (255, 255, 255)

cols, rows = 2, 2
grid_w = cols * CELL_W + (cols - 1) * GAP
grid_h = rows * CELL_H + (rows - 1) * GAP
canvas = Image.new("RGB", (grid_w, grid_h), BG)

LABEL_PAD_X = 18
LABEL_PAD_Y = 8
LABEL_MARGIN = 14
COLOR_BROKEN = (220, 53, 53)   # red
COLOR_INTACT = (40, 170, 70)   # green
TEXT_COLOR = (255, 255, 255)

font = None
for fp in [
    "/usr/share/fonts/truetype/dejavu/DejaVuSans-Bold.ttf",
    "/usr/share/fonts/truetype/liberation/LiberationSans-Bold.ttf",
]:
    if Path(fp).exists():
        font = ImageFont.truetype(fp, 38)
        break
if font is None:
    font = ImageFont.load_default()

for idx, (label, name) in enumerate(selected):
    r, c = divmod(idx, cols)
    img = Image.open(ROOT / name).convert("RGB")
    target_ratio = CELL_W / CELL_H
    w, h = img.size
    ratio = w / h
    if ratio > target_ratio:
        new_w = int(h * target_ratio)
        left = (w - new_w) // 2
        img = img.crop((left, 0, left + new_w, h))
    else:
        new_h = int(w / target_ratio)
        top = (h - new_h) // 2
        img = img.crop((0, top, w, top + new_h))
    img = img.resize((CELL_W, CELL_H), Image.LANCZOS)

    color = COLOR_BROKEN if label == "BROKEN" else COLOR_INTACT
    draw = ImageDraw.Draw(img)
    bbox = draw.textbbox((0, 0), label, font=font)
    tw = bbox[2] - bbox[0]
    th = bbox[3] - bbox[1]
    bw = tw + 2 * LABEL_PAD_X
    bh = th + 2 * LABEL_PAD_Y
    bx2 = CELL_W - LABEL_MARGIN
    by1 = LABEL_MARGIN
    bx1 = bx2 - bw
    by2 = by1 + bh
    draw.rectangle([bx1, by1, bx2, by2], fill=color)
    tx = bx1 + LABEL_PAD_X - bbox[0]
    ty = by1 + LABEL_PAD_Y - bbox[1]
    draw.text((tx, ty), label, fill=TEXT_COLOR, font=font)

    x = c * (CELL_W + GAP)
    y = r * (CELL_H + GAP)
    canvas.paste(img, (x, y))

canvas.save(OUT, "JPEG", quality=92)
print(f"Saved {OUT} ({grid_w}x{grid_h})")
