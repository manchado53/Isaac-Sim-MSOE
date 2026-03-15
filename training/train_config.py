# ─────────────────────────────────────────────
# Nova Carter Patrol Training Configuration
# ─────────────────────────────────────────────

# ── Training ──
NUM_ENVS = 128              # reduced from 256 — Nova Carter USD is heavier than Ridgeback
SEED = 42
MAX_ITERATIONS = 5000         # longer run to let policy converge
CHECKPOINT = "training/logs/sb3/NovaCarter-Patrol/2026-03-14_23-10-04/model.zip"

# ── Video Recording ──
VIDEO = False               # disabled during training to save RAM; eval records video automatically
VIDEO_LENGTH = 2000         # steps per recorded clip
VIDEO_INTERVAL = 2000       # steps between recordings

# ── SLURM ──
PARTITION = "teaching"      # teaching (T4), batch (T4), dgx (V100), dgxh100 (H100)
TIME = "10:00:00"
MEMORY = "64G"              # increased from 32G — Nova Carter needs more RAM
GPUS = 1
