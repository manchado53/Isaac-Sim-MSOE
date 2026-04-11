#!/bin/bash
#SBATCH --job-name=navgraph-floor4
#SBATCH --partition=teaching
#SBATCH --gres=gpu:0
#SBATCH --time=00:30:00
#SBATCH --mem=8G
#SBATCH --cpus-per-task=4
#SBATCH --output=logs/navgraph_%j.out
#SBATCH --error=logs/navgraph_%j.err

# Navigation Graph Generation Job — Floor 4
#
# This job does NOT require a GPU (skeletonization runs on CPU via scikit-image).
# A GPU partition is used only if you want to install and use NVIDIA SWAGGER
# with GPU-accelerated cucim/cupy (see NOTE below).
#
# Usage:
#   sbatch floorplan_vision/run_graph_job.sh
#   sbatch floorplan_vision/run_graph_job.sh --sample-dist 1.5
#   sbatch floorplan_vision/run_graph_job.sh --safety-dist 0.50 --sample-dist 1.0
#
# All extra arguments are forwarded to auto_graph_generator.py.
# See: python3 floorplan_vision/auto_graph_generator.py --help
#
# NOTE on NVIDIA SWAGGER (GPU path):
#   If you want to use SWAGGER's GPU-accelerated skeletonization instead of
#   scikit-image, change --gres=gpu:0 to --gres=gpu:1 and uncomment the
#   SWAGGER install block below. Then replace _skeletonize() in
#   auto_graph_generator.py with the WaypointGraphGenerator call from Section 5.2
#   of swagger_node_placement_guide.md.
#
#   Uncomment to install SWAGGER on first run:
#   pip install --quiet cupy-cuda12x cucim-cu12
#   pip install --quiet -e "$HOME/isaacsim/SWAGGER/"

set -euo pipefail

REPO_DIR="$HOME/isaacsim/isaac-sim-msoe"
SCRIPT="$REPO_DIR/floorplan_vision/auto_graph_generator.py"
OUT_DIR="$REPO_DIR/floorplan_vision/results"

echo "========================================="
echo "Floor 4 Navigation Graph Generator"
echo "Job ID:  $SLURM_JOB_ID"
echo "Node:    $SLURMD_NODENAME"
echo "Script:  $SCRIPT"
echo "Out dir: $OUT_DIR"
echo "Args:    $*"
echo "========================================="
echo ""

# Ensure logs directory exists
mkdir -p "$REPO_DIR/logs"
mkdir -p "$OUT_DIR"

# Change to repo root so relative paths in the script resolve correctly
cd "$REPO_DIR"

echo "Python: $(python3 --version)"
echo "Packages:"
python3 -c "import cv2, networkx, skimage, scipy, numpy; \
    import cv2, networkx; \
    import skimage; import scipy; import numpy; \
    print(f'  numpy={numpy.__version__}  scipy={scipy.__version__}  skimage={skimage.__version__}  networkx={networkx.__version__}  cv2={cv2.__version__}')"
echo ""

echo "Running auto_graph_generator.py..."
echo ""
python3 "$SCRIPT" --out-dir "$OUT_DIR" "$@"

EXIT_CODE=$?

echo ""
echo "========================================="
echo "Job completed — exit code: $EXIT_CODE"
if [ $EXIT_CODE -eq 0 ]; then
    echo "Outputs:"
    ls -lh "$OUT_DIR"/floor_4_auto_navgraph.* 2>/dev/null || true
fi
echo "========================================="

exit $EXIT_CODE
