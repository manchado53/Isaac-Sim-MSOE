#!/usr/bin/env python3
"""
Compare Diagnostic Screenshots -- Side-by-Side Before/After

Creates side-by-side comparison images from the before/after diagnostic
screenshots.  No Isaac Sim required -- uses PIL only.

Usage:
    python compare_screenshots.py
    python compare_screenshots.py --dir images/diagnostics
    python compare_screenshots.py --before-prefix before --after-prefix after
"""

import argparse
import os
import sys


def main():
    parser = argparse.ArgumentParser(description="Create side-by-side before/after comparison images.")
    parser.add_argument("--dir", default="images/diagnostics", help="Directory with screenshots")
    parser.add_argument("--before-prefix", default="before", help="Prefix for before images")
    parser.add_argument("--after-prefix", default="after", help="Prefix for after images")
    args = parser.parse_args()

    try:
        from PIL import Image, ImageDraw, ImageFont
    except ImportError:
        print("ERROR: PIL/Pillow not installed. Install with: pip install Pillow", file=sys.stderr)
        sys.exit(1)

    img_dir = args.dir
    before_prefix = args.before_prefix
    after_prefix = args.after_prefix

    camera_keys = ["G", "1", "2", "3"]
    camera_rooms = {
        "G": "Event Floor IC 100",
        "1": "Robotics Lab IC 101",
        "2": "Classroom 40 IC 302",
        "3": "Capstone Studio IC 405",
    }

    comparisons = []
    for key in camera_keys:
        before_path = os.path.join(img_dir, f"{before_prefix}_cam_{key}.png")
        after_path = os.path.join(img_dir, f"{after_prefix}_cam_{key}.png")

        if not os.path.exists(before_path):
            print(f"  SKIP Camera {key}: missing {before_path}")
            continue
        if not os.path.exists(after_path):
            print(f"  SKIP Camera {key}: missing {after_path}")
            continue

        before_img = Image.open(before_path)
        after_img = Image.open(after_path)

        # Create side-by-side with header
        w, h = before_img.size
        header_h = 40
        gap = 4
        canvas_w = w * 2 + gap
        canvas_h = h + header_h

        canvas = Image.new("RGB", (canvas_w, canvas_h), color=(30, 30, 30))
        canvas.paste(before_img, (0, header_h))
        canvas.paste(after_img, (w + gap, header_h))

        draw = ImageDraw.Draw(canvas)
        try:
            font = ImageFont.truetype("/usr/share/fonts/truetype/dejavu/DejaVuSans-Bold.ttf", 24)
            small_font = ImageFont.truetype("/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf", 18)
        except (OSError, IOError):
            font = ImageFont.load_default()
            small_font = font

        room = camera_rooms.get(key, f"Camera {key}")
        draw.text((10, 8), f"BEFORE  --  Camera {key}: {room}", fill=(255, 200, 100), font=font)
        draw.text((w + gap + 10, 8), f"AFTER  --  Camera {key}: {room}", fill=(100, 200, 255), font=font)

        out_path = os.path.join(img_dir, f"comparison_cam_{key}.png")
        canvas.save(out_path)
        comparisons.append(out_path)
        print(f"  Camera {key}: {out_path}")

    # Create 2x2 grid of all comparisons
    if len(comparisons) == 4:
        imgs = [Image.open(p) for p in comparisons]
        cw, ch = imgs[0].size
        grid = Image.new("RGB", (cw * 2, ch * 2), color=(0, 0, 0))
        grid.paste(imgs[0], (0, 0))
        grid.paste(imgs[1], (cw, 0))
        grid.paste(imgs[2], (0, ch))
        grid.paste(imgs[3], (cw, ch))
        grid_path = os.path.join(img_dir, "comparison_grid.png")
        grid.save(grid_path)
        print(f"\n  Grid: {grid_path}")

    print(f"\nDone. {len(comparisons)} comparison images created.")


if __name__ == "__main__":
    main()
