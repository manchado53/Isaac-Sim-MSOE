# ─────────────────────────────────────────────
# Nova Carter Patrol Training Configuration
# ─────────────────────────────────────────────

# ── Training ──
NUM_ENVS = 256
SEED = 42
MAX_ITERATIONS = None         # run until SLURM time limit
CHECKPOINT = None

# ── Video Recording ──
VIDEO = True               # disabled during training to save RAM; eval records video automatically
VIDEO_LENGTH = 900         # steps per recorded clip
VIDEO_INTERVAL = 5000      # steps between recordings

