#!/usr/bin/env python3
"""
SDXL img2img post-render augmentation pass.

Walks an input directory of rendered PNGs and runs each one through
Stable Diffusion XL's img2img pipeline at low strength, producing a
photo-realistic version next to the original. The intent is to close
the sim-to-real gap on dataset frames before they're handed to a
detector for training.

IMPORTANT — labels:

    Img2img REWRITES whole-image pixels. If a frame has a Replicator
    segmentation mask attached, that mask will no longer match the
    augmented image — SDXL may add, remove, or relocate features.

    For the cracked-glass anomaly pipeline this means: only run this
    augmentation on frames where there is NOTHING TO LABEL — i.e., the
    intact-background frames that fill the negative class. Cracked
    frames must be left untouched so their masks remain valid.

This script intentionally does NOT load Isaac Sim or import sim_utils.
It runs as a pure SDXL post-processor and can execute on any machine
with a GPU + the diffusers stack installed. Isaac Sim is not in scope.

Usage:
    /home/thescript/isaacsim/python.sh simulations/dataset/diffusion_augment.py \\
        --input-dir datasets/cracked_glass/preview \\
        --pattern "panel00_*.png" --strength 0.38

    # Or augment a whole dataset run:
    /home/thescript/isaacsim/python.sh simulations/dataset/diffusion_augment.py \\
        --input-dir datasets/cracked_glass/<timestamp> \\
        --pattern "**/rgb_*.png" --strength 0.30 \\
        --suffix _aug
"""

import argparse
import glob
import os
import sys
import time

os.environ.setdefault("HF_HUB_DISABLE_TELEMETRY", "1")
os.environ.setdefault("TRANSFORMERS_VERBOSITY", "error")

import torch  # noqa: E402
from PIL import Image  # noqa: E402

from diffusers import StableDiffusionXLImg2ImgPipeline  # noqa: E402
from diffusers.utils import logging as dlog  # noqa: E402
dlog.set_verbosity_error()


DEFAULT_PROMPT = (
    "photograph of a modern academic building window, curtain wall glass, "
    "architectural photography, natural daylight, reflections, "
    "sharp focus, high detail, photorealistic, 35mm lens"
)

DEFAULT_NEG = (
    "cartoon, painting, illustration, 3d render, cgi, videogame, blurry, "
    "distorted, low quality, watermark"
)


def parse_args():
    p = argparse.ArgumentParser(
        description="SDXL img2img post-render augmentation pass"
    )
    p.add_argument("--input-dir", required=True,
                   help="Directory containing PNG renders to augment")
    p.add_argument("--pattern", default="*.png",
                   help="Glob pattern relative to --input-dir "
                        "(default: *.png; supports ** for recursion)")
    p.add_argument("--suffix", default="_aug",
                   help="Suffix appended to the original filename for the "
                        "augmented copy (default: _aug)")
    p.add_argument("--output-dir", default=None,
                   help="If set, augmented files go here instead of next "
                        "to the originals")
    p.add_argument("--prompt", default=DEFAULT_PROMPT)
    p.add_argument("--negative", default=DEFAULT_NEG)
    p.add_argument("--strength", type=float, default=0.38,
                   help="img2img denoise strength: 0=passthrough, 1=ignore "
                        "input. Lower preserves geometry/labels better; "
                        "higher gives more photo-real shift. (default 0.38)")
    p.add_argument("--steps", type=int, default=30,
                   help="Diffusion steps (default 30)")
    p.add_argument("--guidance", type=float, default=7.0,
                   help="Classifier-free guidance scale (default 7.0)")
    p.add_argument("--seed", type=int, default=42,
                   help="Single seed used for all images so results are "
                        "reproducible (default 42)")
    p.add_argument("--limit", type=int, default=0,
                   help="Stop after this many images (0 = no limit)")
    p.add_argument("--skip-existing", action="store_true",
                   help="Don't re-augment if the output PNG already exists")
    return p.parse_args()


def find_inputs(input_dir, pattern, limit):
    pattern_path = os.path.join(input_dir, pattern)
    files = sorted(glob.glob(pattern_path, recursive=True))
    if limit > 0:
        files = files[:limit]
    return files


def main():
    args = parse_args()

    if not os.path.isdir(args.input_dir):
        print(f"[aug] ERROR: input dir not found: {args.input_dir}",
              flush=True)
        sys.exit(1)

    inputs = find_inputs(args.input_dir, args.pattern, args.limit)
    if not inputs:
        print(f"[aug] ERROR: no files matched "
              f"{args.input_dir}/{args.pattern}", flush=True)
        sys.exit(2)

    print(f"[aug] {len(inputs)} files matched", flush=True)
    print(f"[aug] strength={args.strength} steps={args.steps} "
          f"guidance={args.guidance}", flush=True)

    print("[aug] loading SDXL img2img (cached after first run)", flush=True)
    t0 = time.time()
    pipe = StableDiffusionXLImg2ImgPipeline.from_pretrained(
        "stabilityai/stable-diffusion-xl-base-1.0",
        torch_dtype=torch.float16,
        variant="fp16",
        use_safetensors=True,
    ).to("cuda")
    pipe.set_progress_bar_config(disable=True)
    print(f"[aug] loaded in {time.time() - t0:.1f}s", flush=True)

    gen = torch.Generator(device="cuda").manual_seed(args.seed)

    n_done = 0
    n_skipped = 0
    for src in inputs:
        rel = os.path.relpath(src, args.input_dir)
        base, ext = os.path.splitext(rel)
        if args.output_dir:
            dst = os.path.join(args.output_dir, base + args.suffix + ext)
        else:
            dst = os.path.join(args.input_dir, base + args.suffix + ext)

        if args.skip_existing and os.path.isfile(dst):
            n_skipped += 1
            continue

        os.makedirs(os.path.dirname(dst) or ".", exist_ok=True)
        try:
            img = Image.open(src).convert("RGB")
        except Exception as e:
            print(f"  skip {rel}: {e}", flush=True)
            continue

        # SDXL latents need 8-divisible dims
        w, h = img.size
        new_w = (w // 8) * 8
        new_h = (h // 8) * 8
        if (new_w, new_h) != (w, h):
            img = img.resize((new_w, new_h), Image.LANCZOS)

        t0 = time.time()
        out = pipe(
            prompt=args.prompt,
            negative_prompt=args.negative,
            image=img,
            strength=args.strength,
            num_inference_steps=args.steps,
            guidance_scale=args.guidance,
            generator=gen,
        ).images[0]
        dt = time.time() - t0

        out.save(dst)
        n_done += 1
        print(f"  [{n_done}/{len(inputs)}] {rel} -> {os.path.basename(dst)} "
              f"({dt:.1f}s)", flush=True)

    print(f"[aug] done. augmented={n_done} skipped={n_skipped} "
          f"(of {len(inputs)} matched)", flush=True)


if __name__ == "__main__":
    main()
