#!/usr/bin/env python3
from PIL import Image, ImageDraw, ImageFont
from pathlib import Path

ROOT = Path("/home/ad.msoe.edu/celliot/Digital-twin-v2/isaac-sim-msoe/real_doors_final/RGB/val")
OUT = Path("/home/ad.msoe.edu/celliot/Digital-twin-v2/isaac-sim-msoe/real_doors_grid.jpg")

# 2x2: 2 closed + 2 open
selected = [
    ("Closed", "Door0028.png"), ("Open", "Door1028.png"),
    ("Closed", "Door0294.png"), ("Open", "Door1294.png"),
]

CELL = 720
GAP = 6
BG = (255, 255, 255)

cols, rows = 2, 2
grid_w = cols * CELL + (cols - 1) * GAP
grid_h = rows * CELL + (rows - 1) * GAP
canvas = Image.new("RGB", (grid_w, grid_h), BG)

LABEL_PAD_X = 18
LABEL_PAD_Y = 8
LABEL_MARGIN = 14
COLOR_CLOSED = (220, 53, 53)
COLOR_OPEN = (40, 170, 70)
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

for idx, (cls, name) in enumerate(selected):
    r, c = divmod(idx, cols)
    img = Image.open(ROOT / cls / name).convert("RGB")
    w, h = img.size
    side = min(w, h)
    left = (w - side) // 2
    top = (h - side) // 2
    img = img.crop((left, top, left + side, top + side)).resize((CELL, CELL), Image.LANCZOS)

    label = "CLOSED" if cls == "Closed" else "OPEN"
    color = COLOR_CLOSED if cls == "Closed" else COLOR_OPEN
    draw = ImageDraw.Draw(img)
    bbox = draw.textbbox((0, 0), label, font=font)
    tw = bbox[2] - bbox[0]
    th = bbox[3] - bbox[1]
    bw = tw + 2 * LABEL_PAD_X
    bh = th + 2 * LABEL_PAD_Y
    bx2 = CELL - LABEL_MARGIN
    by1 = LABEL_MARGIN
    bx1 = bx2 - bw
    by2 = by1 + bh
    draw.rectangle([bx1, by1, bx2, by2], fill=color)
    tx = bx1 + LABEL_PAD_X - bbox[0]
    ty = by1 + LABEL_PAD_Y - bbox[1]
    draw.text((tx, ty), label, fill=TEXT_COLOR, font=font)

    x = c * (CELL + GAP)
    y = r * (CELL + GAP)
    canvas.paste(img, (x, y))

canvas.save(OUT, "JPEG", quality=92)
print(f"Saved {OUT} ({grid_w}x{grid_h})")
