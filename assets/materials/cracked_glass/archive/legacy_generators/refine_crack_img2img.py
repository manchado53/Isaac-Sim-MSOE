#!/usr/bin/env python3
"""
Refine a crack line-art texture via SDXL img2img.

Two modes:
  --mode lineart   (Option A) — refine the line drawing while keeping it
                    black-on-white so it re-enters the Sobel pipeline cleanly.
  --mode realistic (Option B) — transform the line art into a photographic
                    cracked-glass texture. Skips Sobel; output is used as a
                    color/diffuse texture directly.

Usage:
    # Option A — refined line art → Sobel → normal map (drop-in for current pipeline)
    ~/isaacsim/python.sh assets/materials/cracked_glass/refine_crack_img2img.py \\
        --input crack_raw_v4.png --mode lineart --variant v4_refined

    # Option B — realistic crack photo texture (different integration path)
    ~/isaacsim/python.sh assets/materials/cracked_glass/refine_crack_img2img.py \\
        --input crack_raw_v4.png --mode realistic --variant v4_photo
"""

import argparse
import os
import sys
import time

_HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, _HERE)

import numpy as np
import torch
from PIL import Image

os.environ.setdefault("HF_HUB_DISABLE_TELEMETRY", "1")
os.environ.setdefault("TRANSFORMERS_VERBOSITY", "error")

from diffusers import StableDiffusionXLImg2ImgPipeline
from diffusers.utils import logging as dlog
dlog.set_verbosity_error()


PROMPTS = {
    "lineart": {
        "prompt": (
            "highly detailed ink drawing of broken glass cracks, pure black "
            "lines on pure white background, fine hairline fractures between "
            "major cracks, realistic glass fracture mechanics, sharp edges, "
            "technical illustration, no shading, no color, no gradient"
        ),
        "negative": (
            "photograph, color, shading, gradient, gray, blur, soft, "
            "3d, realistic lighting, shadows, texture"
        ),
        "strength": 0.35,
        "guidance": 8.0,
    },
    "realistic": {
        "prompt": (
            "extreme close-up photograph of cracked glass, shattered window, "
            "real glass fractures with light refracting through cracks, "
            "transparent glass with visible crack lines, natural lighting, "
            "sharp focus, photorealistic, macro photography"
        ),
        "negative": (
            "cartoon, drawing, illustration, ink, line art, sketch, "
            "blurry, low quality, text, watermark"
        ),
        "strength": 0.55,
        "guidance": 7.0,
    },
}


def parse_args():
    p = argparse.ArgumentParser(
        description="Refine crack texture via SDXL img2img"
    )
    p.add_argument("--input", default=os.path.join(_HERE, "crack_raw_v4.png"),
                   help="Input crack line-art PNG")
    p.add_argument("--mode", choices=["lineart", "realistic"], default="lineart",
                   help="lineart = stay B&W for Sobel pipeline; "
                        "realistic = photographic texture")
    p.add_argument("--variant", default=None,
                   help="Output variant suffix (default: v4_refined or v4_photo)")
    p.add_argument("--strength", type=float, default=None,
                   help="Override img2img strength (default depends on mode)")
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--steps", type=int, default=30)
    p.add_argument("--prompt", default=None, help="Override the positive prompt")
    p.add_argument("--negative", default=None, help="Override the negative prompt")
    p.add_argument("--out-dir", default=_HERE)
    return p.parse_args()


def main():
    args = parse_args()

    if not os.path.isfile(args.input):
        print(f"[refine] ERROR: input not found: {args.input}", flush=True)
        sys.exit(1)

    cfg = PROMPTS[args.mode]
    variant = args.variant or (f"v4_refined" if args.mode == "lineart" else "v4_photo")
    strength = args.strength or cfg["strength"]
    prompt = args.prompt or cfg["prompt"]
    negative = args.negative or cfg["negative"]

    print(f"[refine] mode={args.mode} variant={variant} "
          f"strength={strength} seed={args.seed}", flush=True)

    # Load input
    img = Image.open(args.input).convert("RGB")
    w, h = img.size
    new_w = (w // 8) * 8
    new_h = (h // 8) * 8
    if (new_w, new_h) != (w, h):
        img = img.resize((new_w, new_h), Image.LANCZOS)

    # Load SDXL img2img
    print("[refine] loading SDXL img2img (cached)", flush=True)
    t0 = time.time()
    pipe = StableDiffusionXLImg2ImgPipeline.from_pretrained(
        "stabilityai/stable-diffusion-xl-base-1.0",
        torch_dtype=torch.float16, variant="fp16", use_safetensors=True,
    ).to("cuda")
    pipe.set_progress_bar_config(disable=True)
    print(f"[refine] loaded in {time.time()-t0:.1f}s", flush=True)

    gen = torch.Generator(device="cuda").manual_seed(args.seed)

    t0 = time.time()
    out = pipe(
        prompt=prompt,
        negative_prompt=negative,
        image=img,
        strength=strength,
        num_inference_steps=args.steps,
        guidance_scale=cfg["guidance"],
        generator=gen,
    ).images[0]
    print(f"[refine] generated in {time.time()-t0:.1f}s", flush=True)

    # Save raw img2img output
    raw_path = os.path.join(args.out_dir, f"crack_raw_{variant}.png")
    out.save(raw_path)
    print(f"[refine] saved {os.path.basename(raw_path)}", flush=True)

    if args.mode == "lineart":
        # Post-process same as generate_textures_diffusion.py:
        # grayscale → contrast → threshold → Sobel → normal map
        from generate_textures import opacity_to_normal

        gray = out.convert("L")
        arr = np.asarray(gray, dtype=np.float32)
        mn, mx = float(arr.min()), float(arr.max())
        stretched = np.clip(
            (arr - mn) / max(mx - mn, 1.0) * 255.0, 0, 255
        ).astype(np.uint8)
        cleaned = np.where(stretched > 160, 255, stretched).astype(np.uint8)
        opacity = Image.fromarray(cleaned).resize((2048, 2048), Image.LANCZOS)

        opacity_path = os.path.join(args.out_dir, f"crack_opacity_{variant}.png")
        opacity.save(opacity_path)
        print(f"[refine] saved {os.path.basename(opacity_path)}", flush=True)

        normal = opacity_to_normal(opacity)
        normal_path = os.path.join(args.out_dir, f"crack_normal_{variant}.png")
        normal.save(normal_path)
        print(f"[refine] saved {os.path.basename(normal_path)}", flush=True)
        print(f"[refine] Use as normal map: --variants {variant}", flush=True)

    else:
        # Realistic mode: upscale and save as a color texture.
        # No Sobel — this gets bound to glass_color_texture, not normalmap.
        color = out.resize((2048, 2048), Image.LANCZOS)
        color_path = os.path.join(args.out_dir, f"crack_color_{variant}.png")
        color.save(color_path)
        print(f"[refine] saved {os.path.basename(color_path)}", flush=True)
        print(f"[refine] This is a COLOR texture — bind to "
              f"glass_color_texture in OmniGlass, not normalmap.", flush=True)

    print("[refine] done", flush=True)


if __name__ == "__main__":
    main()
