#!/usr/bin/env python3
"""
navigation/annotate_tsp_video_vlm.py — Post-process a TSP demo onboard MP4
with the unified Qwen3-VL LoRA (door/glass classifier) + a zero-shot base
Qwen3-VL pass (scene captioner).

Per sampled frame:
  1. Scene pass (adapter disabled): "Describe this indoor scene in one
     short sentence." — always runs. Overlay line 1.
  2. Classifier pass (adapter enabled): only if the scene caption mentions
     door/glass/window. Produces [DOOR_OPEN]/[DOOR_CLOSED]/[GLASS_CRACKED]/
     [GLASS_INTACT] — overlay line 2.

Carries the last overlay forward on frames between samples.

Usage:
  ~/isaacsim/python.sh navigation/annotate_tsp_video_vlm.py \
      --video videos/tsp_doors_onboard_f4_20260419_144246_last30.mp4
"""

import argparse
import csv
import os
import re
import sys
from pathlib import Path

import imageio.v2 as imageio
import numpy as np
import torch
from PIL import Image, ImageDraw, ImageFont
from peft import PeftModel
from transformers import AutoProcessor, Qwen3VLForConditionalGeneration


REPO_ROOT = Path(__file__).resolve().parent.parent
DEFAULT_ADAPTER = REPO_ROOT / "vlm" / "qwen3vl_unified_lora_v4"
DEFAULT_BASE = "Qwen/Qwen3-VL-2B-Instruct"

# Exact 4-class prompt — training only saw this phrasing (see USAGE.md).
CLASS_PROMPT = (
    "<|vision_start|><|image_pad|><|vision_end|>"
    "What is shown in this image? Options: door open, door closed, "
    "glass cracked, glass intact."
)
SCENE_PROMPT = (
    "<|vision_start|><|image_pad|><|vision_end|>"
    "Describe this indoor scene in one short sentence."
)

TRIGGER_RE = re.compile(r"\b(door|doors|glass|window|windows|pane|panes)\b", re.I)
CLASS_RE = re.compile(r"\[(DOOR_OPEN|DOOR_CLOSED|GLASS_CRACKED|GLASS_INTACT)\]")


def parse_args():
    p = argparse.ArgumentParser(description=__doc__.splitlines()[1])
    p.add_argument("--video", required=True)
    p.add_argument("--adapter", default=str(DEFAULT_ADAPTER))
    p.add_argument("--base-model", default=DEFAULT_BASE)
    p.add_argument("--every", type=int, default=30,
                   help="Run VLM every Nth frame (default 30 ≈ 2s at 15fps)")
    p.add_argument("--output", default=None)
    p.add_argument("--max-frames", type=int, default=-1)
    return p.parse_args()


def parse_class_label(text: str):
    m = CLASS_RE.search(text.upper())
    if not m:
        return None
    tag = m.group(1)
    return {
        "DOOR_OPEN":     ("Door",  "OPEN"),
        "DOOR_CLOSED":   ("Door",  "CLOSED"),
        "GLASS_CRACKED": ("Glass", "CRACKED"),
        "GLASS_INTACT":  ("Glass", "INTACT"),
    }[tag]


def load_font(size):
    for path in [
        "/usr/share/fonts/truetype/dejavu/DejaVuSans-Bold.ttf",
        "/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf",
    ]:
        if os.path.isfile(path):
            return ImageFont.truetype(path, size)
    return ImageFont.load_default()


def wrap_text(draw, text, font, max_width):
    lines, cur = [], ""
    for w in text.split():
        test = (cur + " " + w).strip()
        if draw.textlength(test, font=font) <= max_width:
            cur = test
        else:
            if cur:
                lines.append(cur)
            cur = w
    if cur:
        lines.append(cur)
    return lines


LABEL_COLOR = {
    ("Door",  "OPEN"):     (0,   220, 0,   255),
    ("Door",  "CLOSED"):   (255, 180, 0,   255),
    ("Glass", "CRACKED"):  (255, 60,  60,  255),
    ("Glass", "INTACT"):   (120, 200, 255, 255),
}


def draw_overlay(frame_rgb, scene, label, font_big, font_small):
    img = Image.fromarray(frame_rgb).convert("RGB")
    W, H = img.size
    draw = ImageDraw.Draw(img, "RGBA")

    scene_lines = wrap_text(draw, scene, font_small, W - 16) if scene else []
    scene_lines = scene_lines[:2]

    line_h_small = font_small.size + 4
    line_h_big = font_big.size + 4
    scene_h = line_h_small * len(scene_lines)
    label_h = line_h_big if label else 0
    banner_h = scene_h + label_h + 16 if (scene_lines or label) else 0

    if banner_h:
        draw.rectangle([(0, H - banner_h), (W, H)], fill=(0, 0, 0, 180))

    y = H - banner_h + 6
    for line in scene_lines:
        draw.text((8, y), line, font=font_small, fill=(255, 255, 255, 255))
        y += line_h_small

    if label:
        color = LABEL_COLOR[label]
        draw.text((8, y + 2), f"{label[0]}: {label[1]}",
                  font=font_big, fill=color)

    return np.asarray(img)


@torch.no_grad()
def generate(model, processor, image, prompt, max_new_tokens):
    convs = [{"role": "user", "content": prompt}]
    text = processor.apply_chat_template(
        convs, tokenize=False, add_generation_prompt=True,
    )
    inputs = processor(text=text, images=[image], return_tensors="pt").to(model.device)
    out = model.generate(**inputs, max_new_tokens=max_new_tokens, do_sample=False)
    return processor.tokenizer.decode(
        out[0][inputs["input_ids"].shape[1]:], skip_special_tokens=True
    ).strip()


def main():
    args = parse_args()
    in_path = Path(args.video).resolve()
    if not in_path.is_file():
        print(f"[vlm] ERROR: input video not found: {in_path}")
        sys.exit(1)
    out_path = (Path(args.output) if args.output
                else in_path.with_name(in_path.stem + "_vlm.mp4"))
    csv_path = out_path.with_suffix(".csv")

    print(f"[vlm] in={in_path}")
    print(f"[vlm] out={out_path}")
    print(f"[vlm] adapter={args.adapter}")
    print(f"[vlm] sample every {args.every} frame(s)")

    if not torch.cuda.is_available():
        print("[vlm] WARN: CUDA not available — this will be very slow on CPU.")

    print(f"[vlm] loading {args.base_model} (fp16)…")
    base = Qwen3VLForConditionalGeneration.from_pretrained(
        args.base_model, torch_dtype=torch.float16, device_map="auto",
    )
    print(f"[vlm] attaching LoRA adapter {args.adapter}")
    model = PeftModel.from_pretrained(base, args.adapter)
    model.eval()
    # Adapter's tokenizer_config.json uses an older list-form of
    # `extra_special_tokens` that breaks transformers >=4.50. Tokens listed
    # there are stock Qwen3-VL tokens, so the base processor is equivalent.
    processor = AutoProcessor.from_pretrained(args.base_model)
    print(f"[vlm] model on {next(model.parameters()).device}")

    reader = imageio.get_reader(str(in_path), "ffmpeg")
    meta = reader.get_meta_data()
    fps = float(meta.get("fps", 15))
    print(f"[vlm] input fps={fps}")

    writer = imageio.get_writer(
        str(out_path), fps=fps, codec="libx264",
        quality=7, macro_block_size=1,
    )

    font_big = load_font(22)
    font_small = load_font(16)

    csv_f = open(csv_path, "w", newline="")
    csv_w = csv.writer(csv_f)
    csv_w.writerow(["frame_idx", "seconds", "scene_caption",
                    "class_kind", "class_state"])

    last_scene, last_label = "", None
    n_out = 0
    n_samples = 0

    try:
        for frame_idx, frame in enumerate(reader):
            if args.max_frames > 0 and frame_idx >= args.max_frames:
                break

            if frame_idx % args.every == 0:
                pil = Image.fromarray(frame).convert("RGB")

                with model.disable_adapter():
                    scene = generate(model, processor, pil,
                                     SCENE_PROMPT, max_new_tokens=48)
                last_scene = scene

                if TRIGGER_RE.search(scene):
                    raw = generate(model, processor, pil,
                                   CLASS_PROMPT, max_new_tokens=16)
                    last_label = parse_class_label(raw)
                else:
                    last_label = None

                kind = last_label[0] if last_label else ""
                state = last_label[1] if last_label else ""
                csv_w.writerow([frame_idx, f"{frame_idx / fps:.2f}",
                                scene, kind, state])
                csv_f.flush()
                n_samples += 1
                print(f"[vlm] f{frame_idx:4d} ({frame_idx/fps:6.2f}s) "
                      f"scene={scene[:70]!r} label={last_label}")

            out_frame = draw_overlay(frame, last_scene, last_label,
                                     font_big, font_small)
            writer.append_data(out_frame)
            n_out += 1
    finally:
        reader.close()
        writer.close()
        csv_f.close()

    print(f"[vlm] wrote {n_out} frames ({n_samples} VLM samples) → {out_path}")
    print(f"[vlm] log → {csv_path}")


if __name__ == "__main__":
    main()
