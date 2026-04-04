#!/usr/bin/env python3
"""
Nova Carter — VLM Patrol Demo  (Phase 2+3: VLM inference + video assembly)

Reads the frame_cache.npz written by nova_carter_vlm_demo.py (Phase 1),
runs Qwen2-VL-2B-Instruct inference on sampled frames, overlays captions,
and writes two MP4s plus per-frame PNGs for inspection.

Run after Phase 1 completes — invoked automatically by submit_isaac_job.sh.

Outputs (relative to project root):
  videos/vlm_demo_onboard.mp4  — annotated 1st-person feed
  videos/vlm_demo_follow.mp4   — 3rd-person debug view
  frames/NN_frameXXXX_<caption>.png — VLM-inspected frames

Optional: pass --adapter <path> to load a LoRA fine-tuned adapter
  (produced by finetune_qwen2vl.py) instead of zero-shot inference.
"""

import argparse
import os
import re
import sys
import textwrap
import subprocess
import shutil

import numpy as np

_parser = argparse.ArgumentParser(add_help=False)
_parser.add_argument("--adapter", default=None,
                     help="Path to LoRA adapter dir (from finetune_qwen2vl.py). "
                          "If omitted, uses zero-shot inference.")
_args, _ = _parser.parse_known_args()

_script_dir  = os.path.dirname(os.path.abspath(__file__))
_project_dir = os.path.abspath(os.path.join(_script_dir, "..", ".."))

VLM_MODEL_ID      = "Qwen/Qwen2-VL-2B-Instruct"
CLOSED_THRESHOLD  = 5.0 / 90.0  # normalized label threshold matching VLM_fine-tuning.md

# ---------------------------------------------------------------------------
# Load frame cache
# ---------------------------------------------------------------------------

_cache_path = os.path.join(_project_dir, "frame_cache.npz")
if not os.path.isfile(_cache_path):
    print(f"ERROR: frame cache not found at {_cache_path}")
    print("  Phase 1 (nova_carter_vlm_demo.py) must complete successfully first.")
    sys.exit(1)

print(f"[Phase 2] Loading frame cache from {_cache_path} ...")
cache = np.load(_cache_path)
onboard_frames  = list(cache["onboard_frames"])   # list of (H,W,4) uint8
follow_frames   = list(cache["follow_frames"])
vlm_frame_idxs  = list(cache["vlm_frame_idxs"].astype(int))
FPS, W, H       = int(cache["config"][0]), int(cache["config"][1]), int(cache["config"][2])
print(f"[Phase 2] {len(onboard_frames)} onboard frames, "
      f"{len(vlm_frame_idxs)} queued for VLM.")

# ---------------------------------------------------------------------------
# Phase 2: Qwen2-VL Inference
# ---------------------------------------------------------------------------

try:
    import torch
    from transformers import Qwen2VLForConditionalGeneration, AutoProcessor
    from qwen_vl_utils import process_vision_info
    from PIL import Image, ImageDraw, ImageFont
except ImportError as e:
    print(f"[Phase 2] ERROR: Missing dependency — {e}")
    print("  Fix: ~/isaacsim/python.sh -m pip install "
          "transformers accelerate qwen-vl-utils Pillow")
    sys.exit(1)

print(f"[Phase 2] Loading {VLM_MODEL_ID}...")
vlm_model = Qwen2VLForConditionalGeneration.from_pretrained(
    VLM_MODEL_ID,
    torch_dtype=torch.float16,
    device_map="auto",
)

if _args.adapter:
    try:
        from peft import PeftModel
        print(f"[Phase 2] Loading LoRA adapter from {_args.adapter} ...")
        vlm_model = PeftModel.from_pretrained(vlm_model, _args.adapter)
        print("[Phase 2] Fine-tuned adapter loaded.")
    except ImportError:
        print("[Phase 2] WARN: peft not installed — running zero-shot without adapter.")
    except Exception as e:
        print(f"[Phase 2] WARN: adapter load failed ({e}) — running zero-shot.")

vlm_model.eval()
processor = AutoProcessor.from_pretrained(VLM_MODEL_ID)
print(f"[Phase 2] Model loaded on {next(vlm_model.parameters()).device}")

VLM_PROMPT = (
    "A door is visible in this image. "
    "Is it open or closed? "
    "Reply with only [OPEN] or [CLOSED]."
)

def rgba_to_pil(rgba: np.ndarray) -> "Image.Image":
    return Image.fromarray(rgba[:, :, :3].astype(np.uint8), "RGB")

def run_vlm(pil_image: "Image.Image") -> str:
    messages = [{
        "role": "user",
        "content": [
            {"type": "image", "image": pil_image},
            {"type": "text",  "text": VLM_PROMPT},
        ],
    }]
    text = processor.apply_chat_template(
        messages, tokenize=False, add_generation_prompt=True
    )
    image_inputs, video_inputs = process_vision_info(messages)
    inputs = processor(
        text=[text],
        images=image_inputs,
        videos=video_inputs,
        padding=True,
        return_tensors="pt",
    ).to(vlm_model.device)
    with torch.no_grad():
        generated_ids = vlm_model.generate(
            **inputs, max_new_tokens=64, do_sample=False,
        )
    trimmed = [out[len(inp):] for inp, out in zip(inputs.input_ids, generated_ids)]
    return processor.batch_decode(
        trimmed, skip_special_tokens=True, clean_up_tokenization_spaces=False,
    )[0].strip()

def parse_door_class(text: str) -> "str | None":
    """Extract [OPEN] or [CLOSED] from VLM output. Returns 'OPEN', 'CLOSED', or None."""
    text_up = text.upper()
    if "[OPEN]" in text_up or re.search(r'\bOPEN\b', text_up):
        return "OPEN"
    if "[CLOSED]" in text_up or re.search(r'\bCLOSED\b', text_up):
        return "CLOSED"
    return None


captions:      dict[int, str] = {}
door_classes:  dict[int, "str | None"] = {}   # frame_idx → 'OPEN' / 'CLOSED' / None

for i, frame_idx in enumerate(vlm_frame_idxs):
    try:
        pil_img = rgba_to_pil(onboard_frames[frame_idx])
        caption = run_vlm(pil_img)
    except Exception as e:
        caption = f"[VLM error: {type(e).__name__}]"

    cls = parse_door_class(caption)
    captions[frame_idx]     = caption
    door_classes[frame_idx] = cls

    print(f"[Phase 2]   [{i+1:2d}/{len(vlm_frame_idxs)}] "
          f"frame {frame_idx:4d}: raw={caption[:40]!r}  class={cls}")

parsed_count = sum(1 for c in door_classes.values() if c is not None)
print(f"[Phase 2] Done — {len(captions)} captions, "
      f"{parsed_count}/{len(vlm_frame_idxs)} successfully parsed as OPEN/CLOSED.")

open_count   = sum(1 for c in door_classes.values() if c == "OPEN")
closed_count = sum(1 for c in door_classes.values() if c == "CLOSED")
print(f"[Phase 2] Predictions — OPEN: {open_count}, CLOSED: {closed_count}, "
      f"unparsed: {len(vlm_frame_idxs) - parsed_count}")

# Save VLM-inspected onboard frames
frames_dir = os.path.join(_project_dir, "frames")
os.makedirs(frames_dir, exist_ok=True)
for i, frame_idx in enumerate(vlm_frame_idxs):
    caption = captions.get(frame_idx, "")
    safe_caption = caption[:40].replace(" ", "_").replace("/", "-").replace("\\", "-")
    fname = f"{i+1:02d}_frame{frame_idx:04d}_{safe_caption}.png"
    rgba_to_pil(onboard_frames[frame_idx]).save(os.path.join(frames_dir, fname))
print(f"[Phase 2] Onboard frames saved to {frames_dir}/")

# Save corresponding follow camera frames for the same VLM-inspected indices
frames_ext_dir = os.path.join(_project_dir, "frames_external_camera")
os.makedirs(frames_ext_dir, exist_ok=True)
for i, frame_idx in enumerate(vlm_frame_idxs):
    if frame_idx < len(follow_frames):
        caption = captions.get(frame_idx, "")
        safe_caption = caption[:40].replace(" ", "_").replace("/", "-").replace("\\", "-")
        fname = f"{i+1:02d}_frame{frame_idx:04d}_{safe_caption}.png"
        rgba_to_pil(follow_frames[frame_idx]).save(os.path.join(frames_ext_dir, fname))
print(f"[Phase 2] External camera frames saved to {frames_ext_dir}/")

del vlm_model
if torch.cuda.is_available():
    torch.cuda.empty_cache()

# ---------------------------------------------------------------------------
# Phase 3: Annotate frames and write videos
# ---------------------------------------------------------------------------

print("\n[Phase 3] Writing videos...")

videos_dir = os.path.join(_project_dir, "videos")
os.makedirs(videos_dir, exist_ok=True)
ONBOARD_OUT = os.path.join(videos_dir, "vlm_demo_onboard.mp4")
FOLLOW_OUT  = os.path.join(videos_dir, "vlm_demo_follow.mp4")

def find_ffmpeg():
    try:
        import imageio_ffmpeg
        return imageio_ffmpeg.get_ffmpeg_exe()
    except ImportError:
        pass
    local = os.path.expanduser("~/bin/ffmpeg")
    if os.path.isfile(local) and os.access(local, os.X_OK):
        return local
    return shutil.which("ffmpeg")

def open_writer(path: str, width: int, height: int, fps: int):
    ffmpeg = find_ffmpeg()
    if ffmpeg is None:
        raise RuntimeError("ffmpeg not found — imageio-ffmpeg should be bundled with Isaac Sim")
    cmd = [
        ffmpeg, "-y",
        "-f", "rawvideo", "-vcodec", "rawvideo",
        "-s", f"{width}x{height}", "-pix_fmt", "rgb24", "-r", str(fps),
        "-i", "-",
        "-c:v", "libx264", "-pix_fmt", "yuv420p",
        path,
    ]
    return subprocess.Popen(cmd, stdin=subprocess.PIPE,
                            stdout=subprocess.PIPE, stderr=subprocess.PIPE)

def get_caption(frame_idx: int) -> str:
    candidates = [fi for fi in captions if fi <= frame_idx]
    if not candidates:
        return "Initializing scene analysis..."
    nearest = max(candidates)
    cls = door_classes.get(nearest)
    raw = captions[nearest]
    if cls == "OPEN":
        return f"[OPEN]  {raw[:60]}"
    if cls == "CLOSED":
        return f"[CLOSED]  {raw[:60]}"
    return raw[:80]

def load_font(size: int):
    for path in [
        "/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf",
        "/usr/share/fonts/truetype/liberation/LiberationSans-Regular.ttf",
        "/usr/share/fonts/truetype/ubuntu/Ubuntu-R.ttf",
    ]:
        try:
            return ImageFont.truetype(path, size)
        except (IOError, AttributeError):
            continue
    return None

font_main  = load_font(15)
font_badge = load_font(12)

def annotate(rgba: np.ndarray, caption: str, frame_num: int, total: int) -> np.ndarray:
    img  = Image.fromarray(rgba[:, :, :3].astype(np.uint8), "RGB")
    draw = ImageDraw.Draw(img)
    h, w = rgba.shape[:2]

    lines   = textwrap.wrap(caption, width=62)
    line_h  = 20
    padding = 8
    bar_h   = len(lines) * line_h + 2 * padding
    bar_y   = h - bar_h
    draw.rectangle([(0, bar_y), (w, h)], fill=(10, 10, 10))
    y = bar_y + padding
    for line in lines:
        draw.text((padding, y), line, fill=(255, 255, 255), font=font_main)
        y += line_h

    # Badge color: green = OPEN, red = CLOSED, purple = unknown
    cls = door_classes.get(frame_num - 1)
    badge_color = (30, 160, 60) if cls == "OPEN" else (200, 40, 40) if cls == "CLOSED" else (80, 40, 140)
    badge_label = f"DOOR: {cls}" if cls else "QWEN2-VL"
    draw.rectangle([(0, 0), (160, 22)], fill=badge_color)
    draw.text((6, 4), badge_label, fill=(255, 255, 255), font=font_badge)
    draw.text((w - 70, 4), f"{frame_num}/{total}", fill=(180, 180, 180), font=font_badge)

    return np.array(img)

# Write annotated onboard video
total = len(onboard_frames)
proc  = open_writer(ONBOARD_OUT, W, H, FPS)
for i, frame in enumerate(onboard_frames):
    annotated = annotate(frame, get_caption(i), i + 1, total)
    proc.stdin.write(annotated.tobytes())
    if (i + 1) % 50 == 0:
        print(f"[Phase 3]   onboard {i+1}/{total}")
proc.stdin.close()
proc.wait()
if proc.returncode == 0:
    print(f"[Phase 3] ✓ {ONBOARD_OUT}")
else:
    print(f"[Phase 3] ffmpeg error: {proc.stderr.read().decode()}")

# Write follow video
if follow_frames:
    proc2 = open_writer(FOLLOW_OUT, W, H, FPS)
    for frame in follow_frames:
        proc2.stdin.write(frame[:, :, :3].tobytes())
    proc2.stdin.close()
    proc2.wait()
    if proc2.returncode == 0:
        print(f"[Phase 3] ✓ {FOLLOW_OUT}")
    else:
        print(f"[Phase 3] ffmpeg error: {proc2.stderr.read().decode()}")

print(f"\nDemo complete!")
print(f"  Onboard (annotated): {ONBOARD_OUT}")
print(f"  Follow view:         {FOLLOW_OUT}")
print(f"  Onboard frames:      {frames_dir}/")
print(f"  External cam frames: {frames_ext_dir}/")
