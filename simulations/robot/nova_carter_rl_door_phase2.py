#!/usr/bin/env python3
"""
nova_carter_rl_door_phase2.py — VLM inference + video for the RL door demo.

Runs as a separate process after nova_carter_rl_door_demo.py (Phase 1) because
simulation_app.close() calls os._exit(0), killing any in-process code that follows.

Loads door_demo_cache.npz written by Phase 1, runs Qwen2-VL on the door frames,
and writes the annotated video + per-frame PNGs + text log.

Called automatically by submit_isaac_job.sh when door_demo_cache.npz is present.
"""

import os
import sys
import time
import textwrap

import numpy as np
from PIL import Image, ImageDraw, ImageFont

# ---------------------------------------------------------------------------
# Load cache written by Phase 1
# ---------------------------------------------------------------------------

import argparse
_p = argparse.ArgumentParser()
_p.add_argument("--cache", default="door_demo_cache.npz", help="Path to frame cache npz")
_pargs, _ = _p.parse_known_args()

_script_dir  = os.path.dirname(os.path.abspath(__file__))
_project_dir = os.path.abspath(os.path.join(_script_dir, "..", ".."))

# Cache path can be relative (submit script passes relative) or absolute
_cache_path = _pargs.cache if os.path.isabs(_pargs.cache) else os.path.join(_project_dir, _pargs.cache)

if not os.path.isfile(_cache_path):
    print(f"ERROR: cache not found at {_cache_path}")
    sys.exit(1)

print(f"[Phase2] Loading frame cache: {_cache_path}")
_cache = np.load(_cache_path, allow_pickle=False)

scan_rgba  = _cache["scan_rgba"]    # (N, H, W, 4) uint8
scan_dist  = _cache["scan_dist"]    # (N,)
scan_step  = _cache["scan_step"]    # (N,)
door_rgba  = _cache["door_rgba"]
door_dist  = _cache["door_dist"]
door_step  = _cache["door_step"]
all_rgba   = _cache["all_rgba"]     # (T, H, W, 4) uint8
all_dist   = _cache["all_dist"]     # (T,)
all_state  = _cache["all_state"]    # (T,) string
follow_rgba = _cache["follow_rgba"]
DOOR_ANGLE_DEG = float(_cache["door_angle"])
RUN_STAMP      = str(_cache["run_stamp"])
ADAPTER_PATH   = str(_cache["adapter_path"])
DOOR_CENTER_XY = _cache["door_center_xy"]
TARGET_RADIUS  = float(_cache["target_radius"])

print(f"[Phase2] Run stamp   : {RUN_STAMP}")
print(f"[Phase2] Door angle  : {DOOR_ANGLE_DEG}°")
print(f"[Phase2] Scan frames : {len(scan_rgba)}")
print(f"[Phase2] Door frames : {len(door_rgba)}")
print(f"[Phase2] Total frames: {len(all_rgba)}")

STATE_APPROACH = "APPROACH"
STATE_INSPECT  = "INSPECT"

VLM_MODEL_ID = "Qwen/Qwen2-VL-2B-Instruct"

ANOMALY_PROMPT = (
    "You are an autonomous security patrol robot navigating an indoor facility. "
    "Look carefully at this image and describe what you observe. "
    "Specifically flag any anomalies, safety hazards, or unusual situations such as: "
    "fire, smoke, water leaks, fallen objects, obstructions, or people in distress. "
    "If the scene looks completely normal, confirm that. "
    "Be concise — two or three sentences."
)

DOOR_PROMPT = (
    "Is this door open or closed? "
    "Answer with exactly: Door Status: [OPEN]. or Door Status: [CLOSED]."
)

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def rgba_to_pil(rgba):
    return Image.fromarray(rgba[:, :, :3].astype(np.uint8), "RGB")


def load_font(size):
    for p in ["/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf",
              "/usr/share/fonts/truetype/liberation/LiberationSans-Regular.ttf",
              "/usr/share/fonts/truetype/ubuntu/Ubuntu-R.ttf"]:
        try:
            return ImageFont.truetype(p, size)
        except (IOError, AttributeError):
            continue
    return None


# ---------------------------------------------------------------------------
# Load torch + VLM
# ---------------------------------------------------------------------------

print("\n[VLM] Loading dependencies...")
import torch
from transformers import Qwen2VLForConditionalGeneration, AutoProcessor

device    = "cuda" if torch.cuda.is_available() else "cpu"
attn_impl = "flash_attention_2" if (
    torch.cuda.is_available() and torch.cuda.get_device_capability()[0] >= 8
) else "sdpa"


def run_vlm(model, processor, pil_image, prompt):
    messages = [{"role": "user", "content": [
        {"type": "image", "image": pil_image},
        {"type": "text",  "text": prompt},
    ]}]
    text   = processor.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
    inputs = processor(text=[text], images=[pil_image], return_tensors="pt", padding=True).to(device)
    with torch.no_grad():
        out_ids = model.generate(**inputs, max_new_tokens=80, do_sample=False)
    generated = out_ids[:, inputs["input_ids"].shape[1]:]
    return processor.batch_decode(generated, skip_special_tokens=True)[0].strip()


# ---------------------------------------------------------------------------
# 2a — Anomaly scan (base model)
# ---------------------------------------------------------------------------

print(f"\n[VLM] Loading BASE model for anomaly scan ({len(scan_rgba)} frames)...")
t0 = time.time()
processor  = AutoProcessor.from_pretrained(VLM_MODEL_ID)
base_model = Qwen2VLForConditionalGeneration.from_pretrained(
    VLM_MODEL_ID, torch_dtype=torch.float16, device_map="auto",
    attn_implementation=attn_impl,
)
base_model.eval()
print(f"[VLM] Base model loaded in {time.time()-t0:.1f}s")

scan_results = []
for i in range(len(scan_rgba)):
    pil = rgba_to_pil(scan_rgba[i])
    t_s = time.time()
    resp = run_vlm(base_model, processor, pil, ANOMALY_PROMPT)
    scan_results.append((float(scan_dist[i]), int(scan_step[i]), resp))
    print(f"\n[VLM] ANOMALY SCAN {i+1}/{len(scan_rgba)}  dist={scan_dist[i]:.2f} m  ({time.time()-t_s:.1f}s)")
    print(f"      Response: {resp}")

del base_model
if torch.cuda.is_available():
    torch.cuda.empty_cache()

# ---------------------------------------------------------------------------
# 2b — Door inspection (fine-tuned model)
# ---------------------------------------------------------------------------

door_results = []

if len(door_rgba) == 0:
    print("\n[VLM] WARNING: No door frames — robot did not reach door.")
else:
    print(f"\n[VLM] Loading FINE-TUNED model for door inspection ({len(door_rgba)} frames)...")
    sys.stdout.flush()

    try:
        if not os.path.isdir(ADAPTER_PATH):
            print(f"[VLM] WARN: adapter not found at {ADAPTER_PATH} — using base model")
            fine_model = Qwen2VLForConditionalGeneration.from_pretrained(
                VLM_MODEL_ID, torch_dtype=torch.float16, device_map="auto",
                attn_implementation=attn_impl,
            )
        else:
            from peft import PeftModel
            t0 = time.time()
            print("[VLM] Loading base model weights...")
            fine_model = Qwen2VLForConditionalGeneration.from_pretrained(
                VLM_MODEL_ID, torch_dtype=torch.float16, device_map="auto",
                attn_implementation=attn_impl,
            )
            print(f"[VLM] Attaching LoRA adapter from {ADAPTER_PATH}...")
            fine_model = PeftModel.from_pretrained(fine_model, ADAPTER_PATH)
            fine_model.eval()
            print(f"[VLM] Fine-tuned model ready in {time.time()-t0:.1f}s")
    except Exception as _e:
        import traceback
        print(f"\n[VLM] ERROR loading fine-tuned model: {_e}")
        traceback.print_exc()
        print("[VLM] Falling back to base model...")
        fine_model = Qwen2VLForConditionalGeneration.from_pretrained(
            VLM_MODEL_ID, torch_dtype=torch.float16, device_map="auto",
            attn_implementation=attn_impl,
        )
        fine_model.eval()

    for i in range(len(door_rgba)):
        pil = rgba_to_pil(door_rgba[i])
        t_s = time.time()
        resp   = run_vlm(fine_model, processor, pil, DOOR_PROMPT)
        parsed = "OPEN" if "[OPEN]" in resp.upper() else "CLOSED" if "[CLOSED]" in resp.upper() else "UNKNOWN"
        door_results.append((float(door_dist[i]), int(door_step[i]), resp, parsed))
        print(f"\n[VLM] DOOR CHECK {i+1}/{len(door_rgba)}  dist={door_dist[i]:.2f} m  ({time.time()-t_s:.1f}s)")
        print(f"      Response : {resp}")
        print(f"      Parsed   : {parsed}")

    del fine_model
    if torch.cuda.is_available():
        torch.cuda.empty_cache()

# Majority vote
door_votes  = [r[3] for r in door_results]
final_state = max(set(door_votes), key=door_votes.count) if door_votes else "UNKNOWN"
print(f"\n[VLM] FINAL DOOR STATE: {final_state}  "
      f"(votes: OPEN={door_votes.count('OPEN')}, CLOSED={door_votes.count('CLOSED')})")

# ---------------------------------------------------------------------------
# Output paths
# ---------------------------------------------------------------------------

frames_out_dir    = os.path.join(_project_dir, "data",   f"rl_door_demo_frames_{RUN_STAMP}")
log_path          = os.path.join(_project_dir, "data",   f"rl_door_demo_{RUN_STAMP}.txt")
video_path        = os.path.join(_project_dir, "videos", f"rl_door_demo_{RUN_STAMP}.mp4")
follow_video_path = os.path.join(_project_dir, "videos", f"rl_door_demo_follow_{RUN_STAMP}.mp4")
os.makedirs(frames_out_dir, exist_ok=True)
os.makedirs(os.path.join(_project_dir, "videos"), exist_ok=True)

# ---------------------------------------------------------------------------
# Save annotated VLM frames
# ---------------------------------------------------------------------------

font_sm = load_font(12)
font_bd = load_font(11)


def draw_label(pil_img, header, body, dist, badge_color, badge_text):
    draw  = ImageDraw.Draw(pil_img)
    w, h  = pil_img.size
    lines = textwrap.wrap(body, width=80)[:8]
    line_h = 16
    bar_h  = len(lines) * line_h + 12
    bar_y  = h - bar_h
    draw.rectangle([(0, bar_y), (w, h)], fill=(10, 10, 10))
    y = bar_y + 6
    for line in lines:
        draw.text((8, y), line, fill=(230, 230, 230), font=font_sm)
        y += line_h
    draw.rectangle([(0, 0), (200, 22)], fill=badge_color)
    draw.text((6, 4), f"◉ {header}", fill=(255, 255, 255), font=font_bd)
    if badge_text:
        draw.rectangle([(w - 140, 0), (w, 22)], fill=badge_color)
        draw.text((w - 134, 4), badge_text, fill=(255, 255, 255), font=font_bd)
    draw.text((8, 28), f"Door dist: {dist:.1f} m", fill=(200, 200, 200), font=font_bd)
    return pil_img


for i, (dist, step_n, resp) in enumerate(scan_results):
    pil = rgba_to_pil(scan_rgba[i])
    pil = draw_label(pil, "ANOMALY SCAN", resp, dist, (180, 90, 0), None)
    pil.save(os.path.join(frames_out_dir, f"scan_{i+1:02d}_dist{dist:.1f}m.png"))

for i, (dist, step_n, resp, parsed) in enumerate(door_results):
    pil = rgba_to_pil(door_rgba[i])
    badge_color = (30, 160, 60) if parsed == "OPEN" else \
                  (180, 30, 30) if parsed == "CLOSED" else (80, 40, 140)
    pil = draw_label(pil, "DOOR INSPECTION", resp, dist, badge_color, f"DOOR: {parsed}")
    pil.save(os.path.join(frames_out_dir, f"door_{i+1:02d}_dist{dist:.1f}m_{parsed}.png"))

print(f"[Frames] Saved → {frames_out_dir}/")

# ---------------------------------------------------------------------------
# Text log
# ---------------------------------------------------------------------------

with open(log_path, "w") as lf:
    lf.write(f"RL Door Demo Log — {time.strftime('%Y-%m-%d %H:%M:%S')}\n")
    lf.write(f"Door centre XY : {DOOR_CENTER_XY}\n")
    lf.write(f"Door angle set : {DOOR_ANGLE_DEG}° ({'OPEN' if DOOR_ANGLE_DEG > 5 else 'CLOSED'})\n")
    lf.write(f"Target radius  : {TARGET_RADIUS} m\n\n")
    lf.write("=== ANOMALY SCAN (base model) ===\n")
    for dist, step_n, resp in scan_results:
        lf.write(f"  step {step_n:4d} | dist {dist:.2f} m\n  {resp}\n\n")
    lf.write("=== DOOR INSPECTION (fine-tuned) ===\n")
    for dist, step_n, resp, parsed in door_results:
        lf.write(f"  step {step_n:4d} | dist {dist:.2f} m | {parsed}\n  {resp}\n\n")
    lf.write(f"=== FINAL VERDICT: {final_state} ===\n")
print(f"[Log] Saved → {log_path}")

# ---------------------------------------------------------------------------
# Annotated video
# ---------------------------------------------------------------------------

import subprocess
import shutil


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


scan_result_map = {s: r for (_, s, r) in scan_results}
door_result_map = {s: (r, p) for (_, s, r, p) in door_results}


def get_annotation(frame_state, frame_step):
    if frame_state == STATE_APPROACH:
        candidates = [(s, r) for (_, s, r) in scan_results if s <= frame_step]
        caption = candidates[-1][1] if candidates else "Scanning scene..."
        return ("ANOMALY SCAN", caption, (200, 120, 0), "SCANNING")
    else:
        candidates = [(p, r) for (_, s, r, p) in door_results if s <= frame_step]
        if candidates:
            parsed, resp = candidates[-1]
            badge_color = (30, 160, 60) if parsed == "OPEN" else \
                          (180, 30, 30) if parsed == "CLOSED" else (80, 40, 140)
            return ("DOOR INSPECTION", resp, badge_color, f"DOOR: {parsed}")
        return ("DOOR INSPECTION", "Inspecting door...", (80, 40, 140), "DOOR: ?")


def annotate_frame(rgba, frame_state, frame_dist, frame_step, frame_num, total):
    phase, caption, badge_color, badge_text = get_annotation(frame_state, frame_step)
    img  = Image.fromarray(rgba[:, :, :3].astype(np.uint8), "RGB")
    draw = ImageDraw.Draw(img)
    h, w = rgba.shape[:2]
    lines  = textwrap.wrap(caption, width=80)[:8]
    line_h = 16
    bar_h  = len(lines) * line_h + 12
    bar_y  = h - bar_h
    draw.rectangle([(0, bar_y), (w, h)], fill=(10, 10, 10))
    y = bar_y + 6
    for line in lines:
        draw.text((8, y), line, fill=(230, 230, 230), font=font_sm)
        y += line_h
    phase_color = (180, 90, 0) if phase == "ANOMALY SCAN" else badge_color
    draw.rectangle([(0, 0), (190, 22)], fill=phase_color)
    draw.text((6, 4), f"◉ {phase}", fill=(255, 255, 255), font=font_bd)
    if phase == "DOOR INSPECTION" and "?" not in badge_text:
        draw.rectangle([(w - 130, 0), (w, 22)], fill=badge_color)
        draw.text((w - 124, 4), badge_text, fill=(255, 255, 255), font=font_bd)
    draw.text((8, 28), f"Door dist: {frame_dist:.1f} m", fill=(200, 200, 200), font=font_bd)
    draw.text((w - 70, 28), f"{frame_num}/{total}", fill=(150, 150, 150), font=font_bd)
    return np.array(img)


ffmpeg_exe = find_ffmpeg()
print(f"\n[Video] ffmpeg: {ffmpeg_exe}")

if ffmpeg_exe and len(all_rgba) > 0:
    print(f"[Video] Writing onboard video ({len(all_rgba)} frames) → {video_path}")
    H, W = all_rgba.shape[1], all_rgba.shape[2]
    cmd = [ffmpeg_exe, "-y", "-f", "rawvideo", "-vcodec", "rawvideo",
           "-s", f"{W}x{H}", "-pix_fmt", "rgb24", "-r", "15",
           "-i", "pipe:0", "-vcodec", "libx264", "-pix_fmt", "yuv420p", video_path]
    proc = subprocess.Popen(cmd, stdin=subprocess.PIPE)
    total = len(all_rgba)
    for i in range(total):
        ann = annotate_frame(all_rgba[i], str(all_state[i]), float(all_dist[i]), i, i + 1, total)
        proc.stdin.write(ann.tobytes())
    proc.stdin.close()
    proc.wait()
    print(f"[Video] Onboard video saved.")

    if len(follow_rgba) > 0:
        print(f"[Video] Writing follow video ({len(follow_rgba)} frames) → {follow_video_path}")
        H2, W2 = follow_rgba.shape[1], follow_rgba.shape[2]
        cmd2 = [ffmpeg_exe, "-y", "-f", "rawvideo", "-vcodec", "rawvideo",
                "-s", f"{W2}x{H2}", "-pix_fmt", "rgb24", "-r", "15",
                "-i", "pipe:0", "-vcodec", "libx264", "-pix_fmt", "yuv420p", follow_video_path]
        proc2 = subprocess.Popen(cmd2, stdin=subprocess.PIPE)
        for frame in follow_rgba:
            proc2.stdin.write(frame[:, :, :3].tobytes())
        proc2.stdin.close()
        proc2.wait()
        print(f"[Video] Follow video saved.")
else:
    print("[Video] ffmpeg not found or no frames — skipping video.")

# ---------------------------------------------------------------------------
# Clean up cache
# ---------------------------------------------------------------------------

os.remove(_cache_path)
print(f"[Phase2] Cache deleted: {_cache_path}")

print("\n" + "="*60)
print(f"  FINAL DOOR STATE : {final_state}")
print(f"  Door angle set   : {DOOR_ANGLE_DEG}°")
print(f"  Onboard video    : {video_path}")
print(f"  Log              : {log_path}")
print("="*60)
