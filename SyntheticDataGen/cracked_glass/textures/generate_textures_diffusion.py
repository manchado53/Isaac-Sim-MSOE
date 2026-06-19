#!/usr/bin/env python3
"""
Generate diffusion-authored crack normal-map textures via SDXL.

The canonical replacement for the procedural PIL pipeline in
generate_textures.py. Prompts Stable Diffusion XL for a clean
black-on-white line-art crack pattern, post-processes the output, and
runs it through the same Sobel-based `opacity_to_normal` helper from
generate_textures.py to produce a drop-in normal map PNG.

Output filename follows the existing convention so the rest of the
pipeline (SyntheticDataGen/cracked_glass/glass_anomaly.py:create_cracked_glass_materials)
discovers it automatically:
    crack_normal_<variant>.png      (the file Isaac Sim/OmniGlass reads)
    crack_opacity_<variant>.png     (intermediate, kept for inspection)
    crack_raw_<variant>.png         (unmodified SDXL output)

Run with Isaac Sim's bundled Python so it reuses the same torch+diffusers
install we set up earlier in the project:

    /home/thescript/isaacsim/python.sh \\
        assets/materials/cracked_glass/generate_textures_diffusion.py \\
        --variant v5 --seed 7 --prompt "edge chipping only, ..."

CLI knobs are deliberately exposed for experimentation — different
seeds and prompts produce noticeably different crack styles in seconds.
"""

import argparse
import os
import sys
import time

# Make the sibling generate_textures module importable regardless of CWD
_HERE = os.path.dirname(os.path.abspath(__file__))
_ASSETS = os.path.abspath(os.path.join(_HERE, "..", "..", "..", "assets", "materials", "cracked_glass"))
sys.path.insert(0, _HERE)
from generate_textures import opacity_to_normal  # noqa: E402  (Sobel helper)

import numpy as np  # noqa: E402
import torch  # noqa: E402
from PIL import Image  # noqa: E402

os.environ.setdefault("HF_HUB_DISABLE_TELEMETRY", "1")
os.environ.setdefault("TRANSFORMERS_VERBOSITY", "error")

from diffusers import StableDiffusionXLPipeline  # noqa: E402
from diffusers.utils import logging as dlog  # noqa: E402
dlog.set_verbosity_error()


DEFAULT_PROMPT = (
    "ink drawing of broken glass cracks, pure black lines on plain white "
    "background, technical line art, radial shatter with spider-web secondary "
    "cracks, high contrast, detailed fractures, sharp edges, no shading, "
    "no color, no gradient, no gray fill"
)

DEFAULT_NEG = (
    "photograph, color, shading, gradient, texture, noise, realistic, 3d, "
    "render, lighting, shadows, soft, blur, gray fill, background objects, "
    "frame, border"
)


def parse_args():
    p = argparse.ArgumentParser(
        description="SDXL diffusion-authored crack texture generator"
    )
    p.add_argument("--variant", default="v4",
                   help="Variant suffix; output is crack_normal_<variant>.png "
                        "(default: v4)")
    p.add_argument("--seed", type=int, default=1337)
    p.add_argument("--prompt", default=DEFAULT_PROMPT,
                   help="SDXL positive prompt (default is engineered for line "
                        "art that re-enters the Sobel pipeline cleanly)")
    p.add_argument("--negative", default=DEFAULT_NEG,
                   help="SDXL negative prompt")
    p.add_argument("--steps", type=int, default=40,
                   help="Diffusion inference steps (default 40)")
    p.add_argument("--guidance", type=float, default=8.5,
                   help="Classifier-free guidance scale (default 8.5)")
    p.add_argument("--size", type=int, default=1024,
                   help="SDXL output width/height (must be 8-divisible; "
                        "default 1024 — SDXL's native size)")
    p.add_argument("--upscale", type=int, default=2048,
                   help="Final upscale resolution (default 2048 to match v1-v3)")
    p.add_argument("--threshold", type=int, default=160,
                   help="Brightness >= this clamps to pure white in the "
                        "opacity map (default 160; lower keeps more midtone)")
    p.add_argument("--out-dir", default=_HERE,
                   help="Where to write crack_*.png files "
                        "(default: assets/materials/cracked_glass/)")
    return p.parse_args()


def main():
    args = parse_args()
    os.makedirs(args.out_dir, exist_ok=True)

    raw_path = os.path.join(args.out_dir, f"crack_raw_{args.variant}.png")
    opacity_path = os.path.join(args.out_dir, f"crack_opacity_{args.variant}.png")
    normal_path = os.path.join(args.out_dir, f"crack_normal_{args.variant}.png")

    print(f"[gen] variant={args.variant} seed={args.seed} steps={args.steps} "
          f"guidance={args.guidance}", flush=True)
    print(f"[gen] out_dir: {args.out_dir}", flush=True)

    print("[gen] loading SDXL text2img (cached after first run)", flush=True)
    t0 = time.time()
    pipe = StableDiffusionXLPipeline.from_pretrained(
        "stabilityai/stable-diffusion-xl-base-1.0",
        torch_dtype=torch.float16,
        variant="fp16",
        use_safetensors=True,
    ).to("cuda")
    pipe.set_progress_bar_config(disable=True)
    print(f"[gen] loaded in {time.time() - t0:.1f}s", flush=True)

    gen = torch.Generator(device="cuda").manual_seed(args.seed)

    t0 = time.time()
    raw = pipe(
        prompt=args.prompt,
        negative_prompt=args.negative,
        num_inference_steps=args.steps,
        guidance_scale=args.guidance,
        height=args.size,
        width=args.size,
        generator=gen,
    ).images[0]
    print(f"[gen] sampled in {time.time() - t0:.1f}s", flush=True)

    raw.save(raw_path)
    print(f"[gen] wrote {os.path.basename(raw_path)}", flush=True)

    # Post-process: grayscale → auto-contrast stretch → threshold bright pixels
    # to pure white. Goal: match the procedural v1/v2/v3 opacity convention
    # (dark cracks, white background) so opacity_to_normal produces normals
    # comparable to the hand-drawn variants.
    gray = raw.convert("L")
    arr = np.asarray(gray, dtype=np.float32)
    mn, mx = float(arr.min()), float(arr.max())
    stretched = np.clip(
        (arr - mn) / max(mx - mn, 1.0) * 255.0, 0, 255
    ).astype(np.uint8)
    cleaned = np.where(stretched > args.threshold, 255, stretched).astype(np.uint8)
    opacity = Image.fromarray(cleaned)
    opacity_up = opacity.resize((args.upscale, args.upscale), Image.LANCZOS)
    opacity_up.save(opacity_path)
    print(f"[gen] wrote {os.path.basename(opacity_path)}", flush=True)

    normal = opacity_to_normal(opacity_up)
    normal.save(normal_path)
    print(f"[gen] wrote {os.path.basename(normal_path)}", flush=True)

    print(f"[gen] done. Bind via --variants {args.variant} in "
          f"cracked_glass_dataset.py", flush=True)


if __name__ == "__main__":
    main()
