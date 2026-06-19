#!/usr/bin/env python3
from PIL import Image, ImageDraw, ImageFont
from pathlib import Path

SRC = Path("/home/ad.msoe.edu/celliot/Digital-twin-v2/isaac-sim-msoe/data/dh_doors")
OUT = Path("/home/ad.msoe.edu/celliot/Digital-twin-v2/isaac-sim-msoe/dh_doors_grid.jpg")

# 2x2 grid: 2 closed + 2 open
selected = [
    "IMG_5475_CLOSED.jpg", "IMG_5478_OPEN.jpg",
    "IMG_5545_CLOSED.jpg", "IMG_5575_OPEN.jpg",
]

CELL_W, CELL_H = 700, 933   # portrait cells
GAP = 6
BG = (255, 255, 255)

cols, rows = 2, 2
grid_w = cols * CELL_W + (cols - 1) * GAP
grid_h = rows * CELL_H + (rows - 1) * GAP
canvas = Image.new("RGB", (grid_w, grid_h), BG)

# Label styling
LABEL_PAD_X = 18
LABEL_PAD_Y = 8
LABEL_MARGIN = 14
COLOR_CLOSED = (220, 53, 53)   # red
COLOR_OPEN = (40, 170, 70)     # green
TEXT_COLOR = (255, 255, 255)

# Try a bold TrueType font, fall back to default
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

for idx, name in enumerate(selected):
    r, c = divmod(idx, cols)
    img = Image.open(SRC / name).convert("RGB")
    # Center-crop to cell aspect, then resize
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

    # Draw label badge in top-right corner
    label = "CLOSED" if "CLOSED" in name else "OPEN"
    color = COLOR_CLOSED if label == "CLOSED" else COLOR_OPEN
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
    # Center text inside the badge (account for textbbox offset)
    tx = bx1 + LABEL_PAD_X - bbox[0]
    ty = by1 + LABEL_PAD_Y - bbox[1]
    draw.text((tx, ty), label, fill=TEXT_COLOR, font=font)

    x = c * (CELL_W + GAP)
    y = r * (CELL_H + GAP)
    canvas.paste(img, (x, y))

canvas.save(OUT, "JPEG", quality=92)
print(f"Saved {OUT} ({grid_w}x{grid_h})")
