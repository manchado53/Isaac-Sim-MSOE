# ─────────────────────────────────────────────
# Ridgeback Patrol Training Configuration
# ─────────────────────────────────────────────

# ── Training ──
NUM_ENVS = 256
SEED = 42
MAX_ITERATIONS = 50         # short test run to verify NaN is fixed (bump to 1000+ once stable)
CHECKPOINT = None         # path to .zip to resume, or None

# ── Video Recording ──
VIDEO = True  # disable until training is stable (crashes before video interval anyway)
VIDEO_LENGTH = 2000        # steps per recorded clip
VIDEO_INTERVAL = 2000     # steps between recordings

# ── SLURM ──
PARTITION = "teaching"    # teaching (T4), batch (T4), dgx (V100), dgxh100 (H100)
TIME = "08:00:00"
MEMORY = "32G"
GPUS = 1
