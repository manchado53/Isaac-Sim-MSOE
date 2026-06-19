#!/usr/bin/env python3
"""Assemble a grid sheet visualization for one door's lighting x material cells."""

import argparse
from pathlib import Path

from PIL import Image, ImageDraw, ImageFont

PROJECT_ROOT = Path(__file__).resolve().parents[2]
IMAGE_DIR = PROJECT_ROOT / "data" / "visual_review_gridV2"


def load_cells(prefix: str):
    cells = {}
    max_l = max_m = -1
    for p in IMAGE_DIR.glob(f"{prefix}_cell_L*_M*.jpg"):
        stem = p.stem[len(prefix) + len("_cell_"):]
        l_str, m_str = stem.split("_")
        l = int(l_str[1:])
        m = int(m_str[1:])
        cells[(l, m)] = p
        max_l = max(max_l, l)
        max_m = max(max_m, m)
    return cells, max_l + 1, max_m + 1


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--prefix", default="CW_FG_New_Doors_12302795")
    parser.add_argument("--thumb-w", type=int, default=384)
    parser.add_argument("--gap", type=int, default=6)
    parser.add_argument("--margin", type=int, default=64)
    parser.add_argument("--label-pad", type=int, default=40)
    parser.add_argument("--bg", default="#111111")
    parser.add_argument("--fg", default="#EEEEEE")
    parser.add_argument("-o", "--output", default=None)
    args = parser.parse_args()

    cells, n_rows, n_cols = load_cells(args.prefix)
    if not cells:
        raise SystemExit(f"No cells found for prefix {args.prefix!r}")

    sample = Image.open(next(iter(cells.values())))
    w0, h0 = sample.size
    tw = args.thumb_w
    th = int(round(tw * h0 / w0))

    grid_w = n_cols * tw + (n_cols - 1) * args.gap
    grid_h = n_rows * th + (n_rows - 1) * args.gap
    sheet_w = grid_w + 2 * args.margin + args.label_pad
    sheet_h = grid_h + 2 * args.margin + args.label_pad + 60

    sheet = Image.new("RGB", (sheet_w, sheet_h), args.bg)
    draw = ImageDraw.Draw(sheet)

    try:
        title_font = ImageFont.truetype("DejaVuSans-Bold.ttf", 28)
        label_font = ImageFont.truetype("DejaVuSans.ttf", 20)
    except OSError:
        title_font = ImageFont.load_default()
        label_font = ImageFont.load_default()

    title = f"{args.prefix} — {n_rows} lighting × {n_cols} material variants"
    draw.text((args.margin, args.margin // 2), title, fill=args.fg, font=title_font)

    origin_x = args.margin + args.label_pad
    origin_y = args.margin + args.label_pad + 20

    for m in range(n_cols):
        x = origin_x + m * (tw + args.gap) + tw // 2
        draw.text((x, origin_y - 28), f"M{m}", fill=args.fg, font=label_font, anchor="mm")

    for l in range(n_rows):
        y = origin_y + l * (th + args.gap) + th // 2
        draw.text((origin_x - 22, y), f"L{l}", fill=args.fg, font=label_font, anchor="mm")

    for (l, m), path in cells.items():
        with Image.open(path) as im:
            thumb = im.convert("RGB").resize((tw, th), Image.LANCZOS)
        x = origin_x + m * (tw + args.gap)
        y = origin_y + l * (th + args.gap)
        sheet.paste(thumb, (x, y))

    out = Path(args.output) if args.output else PROJECT_ROOT / "videos" / f"{args.prefix}_grid_sheet.jpg"
    out.parent.mkdir(parents=True, exist_ok=True)
    sheet.save(out, quality=92)
    print(f"Saved {sheet.size[0]}x{sheet.size[1]} sheet → {out}")


if __name__ == "__main__":
    main()
