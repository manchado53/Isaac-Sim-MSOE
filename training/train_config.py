# ─────────────────────────────────────────────
# Nova Carter Patrol Training Configuration
# ─────────────────────────────────────────────

# ── Training ──
NUM_ENVS = 256              # reduced from 256 — Nova Carter USD is heavier than Ridgeback
SEED = 42
MAX_ITERATIONS = 500          # 4-stage curriculum test: 50+150+150+150 iters (~5h on T4)
CHECKPOINT = None

# ── Video Recording ──
VIDEO = True               # disabled during training to save RAM; eval records video automatically
VIDEO_LENGTH = 900         # steps per recorded clip
VIDEO_INTERVAL = 5000      # steps between recordings

# ── SLURM ──
PARTITION = "teaching"      # teaching (T4), batch (T4), dgx (V100), dgxh100 (H100)
TIME = "06:00:00"
MEMORY = "64G"              # increased from 32G — Nova Carter needs more RAM
GPUS = 1
