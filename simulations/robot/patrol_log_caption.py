#!/usr/bin/env python3
"""
patrol_log_caption.py — Run the fine-tuned Qwen3-VL adapter on a frame with
no door/window and generate a dispatcher-style scene caption.

Demonstrates that the fine-tuned VLM retains its general-knowledge captioning
ability alongside door classification — useful as a "free metadata" logger
during patrol for frames where no trained anomaly is present.

Usage:
  ~/.conda/envs/vlm/bin/python simulations/robot/patrol_log_caption.py \
      --image data/no_window_no_door_image/Gemini_Generated_Image_quxknpquxknpquxk.png \
      --adapter data/qwen3vl_door_lora_v4_curated \
      --out data/patrol_caption_demo.txt
"""

import argparse
import os
import time

import torch
from PIL import Image
from transformers import AutoProcessor, Qwen3VLForConditionalGeneration

VLM_MODEL_ID = "Qwen/Qwen3-VL-2B-Instruct"

PATROL_PROMPT = (
    "You are an autonomous security patrol robot producing a structured "
    "log entry for a dispatcher. Scan this frame methodically and produce "
    "a short report with the exact format below.\n\n"
    "Fill in every field. Use concrete visual details, not speculation.\n"
    "Keep each field to one short sentence.\n\n"
    "SCENE: <type of space — corridor, lobby, stairwell, office, etc. — "
    "and one observable detail about it, e.g. 'narrow corridor with "
    "tiled flooring and overhead lighting'>\n"
    "OCCUPANTS: <number of people visible and what they are doing, or "
    "'none visible'>\n"
    "OBJECTS: <list any free-standing items, equipment, furniture, bags, "
    "carts, debris, or signage actually visible in the frame; or 'none'>\n"
    "LIGHTING: <current state — bright/dim/mixed, any dark patches, any "
    "flickering, any smoke or haze>\n"
    "HAZARDS: <liquids on floor, damage, obstructions, broken items; "
    "or 'none observed'>\n"
    "PRIORITY: <LOW / MEDIUM / HIGH — based on whether anything in the "
    "frame would warrant a human response>\n"
    "NOTES: <one sentence summarizing anything unusual a dispatcher "
    "should know, or 'routine frame, no action required'>\n\n"
    "Rules:\n"
    "- Report only what is visually present. Do not invent objects.\n"
    "- Do not describe paint color, building architecture, or normal "
    "structural elements as if they were notable.\n"
    "- If a field is truly empty, say 'none' or 'none observed' — do "
    "not skip the field.\n"
    "- Output the fields in the exact order shown, one per line."
)


def run_prompt(model, processor, image, prompt_text, device, max_new_tokens=320):
    messages = [{
        "role": "user",
        "content": [
            {"type": "image", "image": image},
            {"type": "text",  "text": prompt_text},
        ],
    }]
    text = processor.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
    inputs = processor(text=[text], images=[image], return_tensors="pt", padding=True).to(device)
    with torch.no_grad():
        out = model.generate(**inputs, max_new_tokens=max_new_tokens, do_sample=False)
    generated = out[:, inputs["input_ids"].shape[1]:]
    return processor.batch_decode(generated, skip_special_tokens=True)[0].strip()


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--image", required=True, help="Path to input image.")
    p.add_argument("--adapter", default="data/qwen3vl_door_lora_v4_curated",
                   help="LoRA adapter dir. Omit for base model only.")
    p.add_argument("--out", default=None, help="Optional path to save output text.")
    p.add_argument("--also-baseline", action="store_true",
                   help="Also run the base model (no adapter) for comparison.")
    args = p.parse_args()

    device = "cuda" if torch.cuda.is_available() else "cpu"
    attn_impl = "flash_attention_2" if (
        torch.cuda.is_available() and torch.cuda.get_device_capability()[0] >= 8
    ) else "sdpa"
    print(f"Device: {device}  Attention: {attn_impl}")
    print(f"Image:   {args.image}")
    print(f"Adapter: {args.adapter or 'none (base model)'}\n")

    t0 = time.time()
    processor = AutoProcessor.from_pretrained(VLM_MODEL_ID)
    base_model = Qwen3VLForConditionalGeneration.from_pretrained(
        VLM_MODEL_ID,
        torch_dtype=torch.float16,
        device_map="auto",
        attn_implementation=attn_impl,
    )
    print(f"Base model loaded in {time.time()-t0:.1f}s")

    image = Image.open(args.image).convert("RGB")
    print(f"Image size: {image.size}\n")

    results = []

    if args.also_baseline:
        print("=" * 70)
        print("[BASELINE — no adapter]")
        print("=" * 70)
        base_model.eval()
        t0 = time.time()
        baseline_out = run_prompt(base_model, processor, image, PATROL_PROMPT, device)
        print(baseline_out)
        print(f"  ({time.time()-t0:.1f}s)\n")
        results.append(("baseline", baseline_out))

    if args.adapter:
        from peft import PeftModel
        print(f"Loading adapter: {args.adapter}")
        model = PeftModel.from_pretrained(base_model, args.adapter)
        model.eval()
        print("=" * 70)
        print(f"[FINE-TUNED — {os.path.basename(args.adapter)}]")
        print("=" * 70)
        t0 = time.time()
        ft_out = run_prompt(model, processor, image, PATROL_PROMPT, device)
        print(ft_out)
        print(f"  ({time.time()-t0:.1f}s)")
        results.append(("fine-tuned", ft_out))

    if args.out:
        os.makedirs(os.path.dirname(args.out) or ".", exist_ok=True)
        with open(args.out, "w") as f:
            f.write(f"Image:   {args.image}\n")
            f.write(f"Adapter: {args.adapter or 'none'}\n")
            f.write(f"\nPrompt:\n{PATROL_PROMPT}\n\n")
            for tag, txt in results:
                f.write("=" * 70 + "\n")
                f.write(f"[{tag}]\n")
                f.write("=" * 70 + "\n")
                f.write(txt + "\n\n")
            f.flush()
            os.fsync(f.fileno())
        print(f"\nSaved to: {args.out}")


if __name__ == "__main__":
    main()
