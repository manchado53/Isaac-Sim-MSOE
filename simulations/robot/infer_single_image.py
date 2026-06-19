#!/usr/bin/env python3
"""
infer_single_image.py — Run the fine-tuned Qwen2-VL adapter on one image.

Prints the raw model response for two prompts:
  1. The door-classification prompt used during training.
  2. An open-ended "describe this image" prompt — lets you see whether general
     captioning behavior survived LoRA + projector fine-tuning, or whether
     the model collapses to Door Status: [...] regardless of input.

Usage:
  ~/.conda/envs/vlm/bin/python simulations/robot/infer_single_image.py \
      --image data/no_window_no_door_image/Gemini_Generated_Image_quxknpquxknpquxk.png \
      --adapter data/qwen2vl_door_lora_dgx
"""

import argparse
import os
import time

import torch
from PIL import Image
from transformers import AutoProcessor

# Default base model — overridable via --model. The matching loader is picked
# by --model-class (qwen2_vl, qwen3_vl). Qwen3-VL needs its own model class
# because AutoModelForImageTextToText doesn't always pick the right one.
DEFAULT_MODEL_ID = "Qwen/Qwen2-VL-2B-Instruct"


def _load_qwen2_vl(model_id, dtype, attn_impl):
    from transformers import Qwen2VLForConditionalGeneration
    processor = AutoProcessor.from_pretrained(model_id)
    model = Qwen2VLForConditionalGeneration.from_pretrained(
        model_id, torch_dtype=dtype, device_map="auto", attn_implementation=attn_impl,
    )
    return model, processor


def _load_qwen3_vl(model_id, dtype, attn_impl):
    from transformers import Qwen3VLForConditionalGeneration
    processor = AutoProcessor.from_pretrained(model_id)
    model = Qwen3VLForConditionalGeneration.from_pretrained(
        model_id, torch_dtype=dtype, device_map="auto", attn_implementation=attn_impl,
    )
    return model, processor


LOADERS = {"qwen2_vl": _load_qwen2_vl, "qwen3_vl": _load_qwen3_vl}

PROMPTS = [
    ("door-classification", "Is this door open or closed? Answer with exactly: Door Status: [OPEN]. or Door Status: [CLOSED]."),
    ("open-ended-describe", "Describe this image in detail."),
    ("open-ended-objects",  "What objects do you see in this image?"),
]

_PROMPT = (
    "Analyze this image. If there is a door in the image, respond with exactly: "
    "Door Status: [OPEN]. or Door Status: [CLOSED]. "
    "If there is no door in the image, instead describe the image in detail, "
    "starting your response with: No door detected. Description:"
)BRANCHED

# Prompt-engineering sweep to find a branched prompt that keeps classification
# accuracy while also triggering the no-door-description branch correctly.
# Unified 4-class fine-tune (qwen3vl_unified_lora_v4) was trained on:
#   prompt:  "What is shown in this image? Options: door open, door closed,
#             glass cracked, glass intact."
#   answer:  "State: [DOOR_OPEN]." / [DOOR_CLOSED] / [GLASS_CRACKED] / [GLASS_INTACT]
# The prompts below match that exact framing — anything else (e.g. the door-only
# PROMPTS / BRANCHED_PROMPT above) measures prompt-mismatch, not forgetting.
UNIFIED_TRAINING_PROMPT = (
    "What is shown in this image? "
    "Options: door open, door closed, glass cracked, glass intact."
)

UNIFIED_BRANCHED_PROMPT = (
    UNIFIED_TRAINING_PROMPT +
    " If none of these are present, give a general description of the "
    "scene instead in at most 20 words."
)

UNIFIED_PROMPTS = [
    ("unified-classification", UNIFIED_TRAINING_PROMPT),
    ("unified-branched",       UNIFIED_BRANCHED_PROMPT),
    ("open-ended-describe",    "Describe this image in 1-2 sentences."),
]

PROMPT_SWEEP = [
    ("V1_analyze_first", BRANCHED_PROMPT),

    # V2: keep the training question exactly, append the no-door fallback.
    ("V2_training_first",
     "Is this door open or closed? Answer with exactly: Door Status: [OPEN]. "
     "or Door Status: [CLOSED]. "
     "If there is no door visible in the image, instead start your response "
     "with 'No door detected.' and then describe the image."),

    # V3: minimal conditional, no filler.
    ("V3_minimal_conditional",
     "If a door is visible, answer: Door Status: [OPEN]. or Door Status: [CLOSED]. "
     "Otherwise, describe the image starting with 'No door detected.'"),

    # V4: explicit two-step reasoning.
    ("V4_stepwise",
     "Step 1: Check if a door is visible. "
     "Step 2a: If yes, respond only with Door Status: [OPEN]. or Door Status: [CLOSED]. "
     "Step 2b: If no, describe the image starting with 'No door detected.'"),
]


def run_prompt(model, processor, image, prompt_text, device, max_new_tokens=128):
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
    p.add_argument("--image",   action="append", required=True,
                   help="Image path(s). Repeat --image to test multiple.")
    p.add_argument("--adapter", default=None,
                   help="LoRA adapter dir. Omit for base model.")
    p.add_argument("--model",   default=DEFAULT_MODEL_ID,
                   help="HF model ID for the base model (must match the one the adapter was trained on).")
    p.add_argument("--model-class", choices=list(LOADERS.keys()), default="qwen2_vl",
                   help="Loader to use — pick the family matching --model.")
    p.add_argument("--prompt-mode",
                   choices=["all", "branched", "sweep", "unified"], default="all",
                   help="'all' = the 3 door-only PROMPTS (legacy); "
                        "'branched' = single door-only conditional prompt; "
                        "'sweep' = all door-only PROMPT_SWEEP variants; "
                        "'unified' = 4-class unified prompts matching qwen3vl_unified_lora_v4")
    args = p.parse_args()

    device = "cuda" if torch.cuda.is_available() else "cpu"
    attn_impl = "flash_attention_2" if (
        torch.cuda.is_available() and torch.cuda.get_device_capability()[0] >= 8
    ) else "sdpa"
    print(f"Device:  {device}  Attention: {attn_impl}")
    print(f"Model:   {args.model}  ({args.model_class})")
    print(f"Image:   {args.image}")
    print(f"Adapter: {args.adapter or 'none (base model)'}")

    t0 = time.time()
    loader = LOADERS[args.model_class]
    model, processor = loader(args.model, torch.float16, attn_impl)
    if args.adapter:
        from peft import PeftModel
        model = PeftModel.from_pretrained(model, args.adapter)
    model.eval()
    print(f"Loaded in {time.time()-t0:.1f}s\n")

    if args.prompt_mode == "all":
        prompts = PROMPTS
    elif args.prompt_mode == "branched":
        prompts = [("branched", BRANCHED_PROMPT)]
    elif args.prompt_mode == "unified":
        prompts = UNIFIED_PROMPTS
    else:
        prompts = PROMPT_SWEEP

    for img_path in args.image:
        image = Image.open(img_path).convert("RGB")
        print("\n" + "#" * 70)
        print(f"# IMAGE: {img_path}   size={image.size}")
        print("#" * 70)
        for tag, prompt in prompts:
            print("=" * 70)
            print(f"[{tag}]  prompt: {prompt!r}")
            print("-" * 70)
            t0 = time.time()
            response = run_prompt(model, processor, image, prompt, device, max_new_tokens=160)
            print(response)
            print(f"  ({time.time()-t0:.1f}s)")
        print("=" * 70)


if __name__ == "__main__":
    main()
