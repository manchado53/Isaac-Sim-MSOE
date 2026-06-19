#!/usr/bin/env python3
"""
overlay_caption.py — Render a patrol-log caption onto an image for slides.

Draws a semi-transparent dark bar along the bottom of the image with the VLM
caption in a monospace font, plus an optional header stripe ("PATROL LOG —
<timestamp>") at the top to sell the security-camera aesthetic.

No GPU required — runs on the login node.

Usage:
  python simulations/robot/overlay_caption.py \
      --image data/no_window_no_door_image/Gemini_Generated_Image_quxknpquxknpquxk.png \
      --caption "Empty corridor, nothing notable." \
      --out data/patrol_caption_overlay.png

Or read caption from a file:
  python simulations/robot/overlay_caption.py \
      --image <img> --caption-file data/patrol_caption_demo.txt \
      --out data/patrol_caption_overlay.png
"""

import argparse
import os
import textwrap
from datetime import datetime

from PIL import Image, ImageDraw, ImageFont


HEADER_TEXT = "PATROL LOG"
DEFAULT_TIMESTAMP = "02:14:44  |  FLOOR 4  |  NODE H4-18"


def _load_font(size, mono=True):
    candidates = [
        "/usr/share/fonts/truetype/dejavu/DejaVuSansMono-Bold.ttf" if mono
        else "/usr/share/fonts/truetype/dejavu/DejaVuSans-Bold.ttf",
        "/usr/share/fonts/truetype/dejavu/DejaVuSansMono.ttf" if mono
        else "/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf",
        "/usr/share/fonts/truetype/liberation/LiberationMono-Bold.ttf",
    ]
    for path in candidates:
        if os.path.exists(path):
            return ImageFont.truetype(path, size)
    return ImageFont.load_default()


def _parse_caption_file(path):
    """Pull the fine-tuned response out of a patrol_log_caption.py output file."""
    with open(path) as f:
        text = f.read()
    if "[fine-tuned]" in text:
        tail = text.split("[fine-tuned]", 1)[1]
        tail = tail.split("=" * 70, 1)[-1].strip()
        return tail.split("\n\n")[0].strip()
    if "[baseline]" in text:
        tail = text.split("[baseline]", 1)[1]
        tail = tail.split("=" * 70, 1)[-1].strip()
        return tail.split("\n\n")[0].strip()
    return text.strip()


def overlay(image_path, caption, out_path, timestamp=None, header=True):
    img = Image.open(image_path).convert("RGB")
    W, H = img.size

    caption_font_size = max(18, W // 45)
    header_font_size  = max(14, W // 60)
    caption_font = _load_font(caption_font_size, mono=True)
    header_font  = _load_font(header_font_size,  mono=True)

    # Wrap caption to ~75% of image width
    max_chars = max(30, int(W / (caption_font_size * 0.6)))
    lines = []
    for paragraph in caption.splitlines():
        lines.extend(textwrap.wrap(paragraph, width=max_chars) or [""])

    pad = caption_font_size
    line_h = caption_font_size + 8
    bar_h = pad * 2 + line_h * len(lines)

    header_h = 0
    if header:
        header_h = header_font_size + pad

    # Create overlay canvas
    canvas = Image.new("RGBA", (W, H), (0, 0, 0, 0))
    draw = ImageDraw.Draw(canvas)

    # Header stripe (top)
    if header:
        draw.rectangle([(0, 0), (W, header_h)], fill=(200, 30, 30, 230))
        ts = timestamp or datetime.now().strftime("%H:%M:%S  |  FLOOR 4  |  NODE H4-18")
        header_str = f"{HEADER_TEXT}  —  {ts}"
        draw.text((pad, pad // 2), header_str, fill=(255, 255, 255, 255), font=header_font)

    # Caption bar (bottom)
    bar_top = H - bar_h
    draw.rectangle([(0, bar_top), (W, H)], fill=(0, 0, 0, 210))
    # Thin accent line at top of bar
    draw.rectangle([(0, bar_top), (W, bar_top + 3)], fill=(200, 30, 30, 255))

    # Caption text
    y = bar_top + pad
    for line in lines:
        draw.text((pad, y), line, fill=(255, 255, 255, 255), font=caption_font)
        y += line_h

    out = Image.alpha_composite(img.convert("RGBA"), canvas).convert("RGB")
    os.makedirs(os.path.dirname(out_path) or ".", exist_ok=True)
    out.save(out_path, quality=95)
    print(f"Wrote {out_path}  ({out.size[0]}x{out.size[1]})")


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--image", required=True)
    p.add_argument("--caption", default=None, help="Caption text (inline).")
    p.add_argument("--caption-file", default=None,
                   help="Path to patrol_log_caption.py output file.")
    p.add_argument("--out", required=True)
    p.add_argument("--timestamp", default=DEFAULT_TIMESTAMP,
                   help="Custom header timestamp string.")
    p.add_argument("--no-header", action="store_true")
    args = p.parse_args()

    if args.caption and args.caption_file:
        raise SystemExit("Pass --caption OR --caption-file, not both.")
    if not args.caption and not args.caption_file:
        raise SystemExit("Must pass --caption or --caption-file.")

    caption = args.caption or _parse_caption_file(args.caption_file)
    overlay(args.image, caption, args.out,
            timestamp=args.timestamp, header=not args.no_header)


if __name__ == "__main__":
    main()
