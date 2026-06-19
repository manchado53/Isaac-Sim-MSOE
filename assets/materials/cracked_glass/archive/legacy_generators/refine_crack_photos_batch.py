#!/usr/bin/env python3
"""
Run SDXL img2img on the 5 reference photographs, one SDXL load per job.

For each (variant, photo) pair:
  1. SDXL img2img (lineart prompt, strength 0.65) — photo -> B&W line art
  2. Grayscale + contrast stretch + threshold -> opacity map
  3. Sobel -> normal map

Outputs (overwrites existing):
  crack_raw_<variant>.png
  crack_opacity_<variant>.png
  crack_normal_<variant>.png

Run via: ~/isaacsim/python.sh assets/materials/cracked_glass/refine_crack_photos_batch.py
"""

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

from generate_textures import opacity_to_normal


REFS = _HERE + "/references"

JOBS = [
    ("impact",   f"{REFS}/impact_pexels.jpg",       0.55),
    ("stress",   f"{REFS}/crack_on_window.jpg",     0.65),
    ("edge",     f"{REFS}/edge_thermal_break.jpg",  0.65),
    ("thermal",  f"{REFS}/thermal_windowwise.jpg",  0.65),
    ("tempered", f"{REFS}/tempered_safety2.jpg",    0.55),
]

PROMPT = (
    "highly detailed ink drawing of broken glass cracks, pure black lines on "
    "pure white background, fine hairline fractures between major cracks, "
    "realistic glass fracture mechanics, sharp edges, technical illustration, "
    "no shading, no color, no gradient"
)
NEGATIVE = (
    "photograph, color, shading, gradient, gray, blur, soft, 3d, realistic "
    "lighting, shadows, texture, window frame, wall, background objects"
)
GUIDANCE = 8.0
STEPS = 30
SEED = 42


def prep_for_sdxl(img, target_long_edge=1024):
    """Downscale so the long edge is target_long_edge, then snap to multiple of 8.
    SDXL is trained at ~1024 and OOMs on 4k inputs."""
    w, h = img.size
    scale = target_long_edge / float(max(w, h))
    if scale < 1.0:
        w, h = int(round(w * scale)), int(round(h * scale))
    w = (w // 8) * 8
    h = (h // 8) * 8
    return img.resize((w, h), Image.LANCZOS)


def sdxl_to_opacity_and_normal(out_img, variant):
    """Same post-process as refine_crack_img2img.py lineart mode."""
    gray = out_img.convert("L")
    arr = np.asarray(gray, dtype=np.float32)
    mn, mx = float(arr.min()), float(arr.max())
    stretched = np.clip((arr - mn) / max(mx - mn, 1.0) * 255.0, 0, 255).astype(np.uint8)
    cleaned = np.where(stretched > 160, 255, stretched).astype(np.uint8)
    opacity = Image.fromarray(cleaned).resize((2048, 2048), Image.LANCZOS)

    op_path = os.path.join(_HERE, f"crack_opacity_{variant}.png")
    opacity.save(op_path)
    print(f"[batch] saved {os.path.basename(op_path)}", flush=True)

    normal = opacity_to_normal(opacity)
    nm_path = os.path.join(_HERE, f"crack_normal_{variant}.png")
    normal.save(nm_path)
    print(f"[batch] saved {os.path.basename(nm_path)}", flush=True)


def main():
    for variant, path, _s in JOBS:
        if not os.path.isfile(path):
            print(f"[batch] ERROR: missing reference {path}", flush=True)
            sys.exit(1)

    print("[batch] loading SDXL img2img", flush=True)
    t0 = time.time()
    pipe = StableDiffusionXLImg2ImgPipeline.from_pretrained(
        "stabilityai/stable-diffusion-xl-base-1.0",
        torch_dtype=torch.float16, variant="fp16", use_safetensors=True,
    ).to("cuda")
    pipe.set_progress_bar_config(disable=True)
    pipe.enable_vae_tiling()
    print(f"[batch] loaded in {time.time()-t0:.1f}s", flush=True)

    for variant, path, strength in JOBS:
        print(f"\n[batch] === {variant}  (strength={strength}) ===", flush=True)
        print(f"[batch] input: {os.path.basename(path)}", flush=True)

        img = Image.open(path).convert("RGB")
        img = prep_for_sdxl(img, target_long_edge=1024)
        print(f"[batch] input size: {img.size}", flush=True)

        gen = torch.Generator(device="cuda").manual_seed(SEED)
        t0 = time.time()
        out = pipe(
            prompt=PROMPT,
            negative_prompt=NEGATIVE,
            image=img,
            strength=strength,
            num_inference_steps=STEPS,
            guidance_scale=GUIDANCE,
            generator=gen,
        ).images[0]
        print(f"[batch] generated in {time.time()-t0:.1f}s", flush=True)

        raw_path = os.path.join(_HERE, f"crack_raw_{variant}.png")
        out.save(raw_path)
        print(f"[batch] saved {os.path.basename(raw_path)}", flush=True)

        sdxl_to_opacity_and_normal(out, variant)

    print("\n[batch] all 5 variants generated", flush=True)


if __name__ == "__main__":
    main()
