# ─────────────────────────────────────────────
# Nova Carter Patrol Training Configuration
# ─────────────────────────────────────────────

# ── Training ──
NUM_ENVS = 512              # 512 for RTX 5090 (32 GB VRAM)
SEED = 42
MAX_ITERATIONS = None         # run until SLURM time limit
CHECKPOINT = None

# ── Video Recording ──
VIDEO = True               # disabled during training to save RAM; eval records video automatically
VIDEO_LENGTH = 900         # steps per recorded clip
VIDEO_INTERVAL = 5000      # steps between recordings

# ── SLURM ──
PARTITION = "teaching"      # teaching (T4), batch (T4), dgx (V100), dgxh100 (H100)
TIME = "12:00:00"
MEMORY = "64G"              # increased from 32G — Nova Carter needs more RAM
GPUS = 1
