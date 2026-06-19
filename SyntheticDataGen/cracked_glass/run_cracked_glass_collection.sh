#!/usr/bin/env bash
# run_cracked_glass_collection.sh -- orchestrate 13 per-panel runs.
#
# Matches Tanner's one-job-per-door pattern: sequential runs, each one a
# fresh Isaac Sim process with 27 cameras (3 arcs x 3 dists x 3 heights).
# No GPU contention, crash in panel N only loses that panel.
#
# Usage:
#   bash SyntheticDataGen/cracked_glass/run_cracked_glass_collection.sh
#
# Override shared output dir or Python launcher via env vars:
#   OUTPUT=datasets/cracked_glass/testrun \
#   PYTHON=$HOME/isaacsim/python.sh \
#   bash SyntheticDataGen/cracked_glass/run_cracked_glass_collection.sh
#
# Pass through CLI flags to the collector (e.g. to smoke-test with 1 lighting):
#   bash SyntheticDataGen/cracked_glass/run_cracked_glass_collection.sh \
#       --lightings bright_fluorescent --arc-angles 0 --arc-distances 2.0 \
#       --uv-tilings 2 --heights 0

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_DIR="$(cd "$SCRIPT_DIR/../.." && pwd)"
cd "$PROJECT_DIR"

OUTPUT="${OUTPUT:-datasets/cracked_glass/$(date +%Y%m%d_%H%M%S)}"
PYTHON="${PYTHON:-$HOME/isaacsim/python.sh}"
PANELS="${PANELS:-2 3 4 6 13 15 18 22 28 31 36 40 42}"

mkdir -p "$OUTPUT" logs
echo "=== Cracked-glass dataset collection ==="
echo "Project:  $PROJECT_DIR"
echo "Output:   $OUTPUT"
echo "Python:   $PYTHON"
echo "Panels:   $PANELS"
echo "Started:  $(date)"
echo ""

T_TOTAL_START=$(date +%s)
FAILED_PANELS=()

for IDX in $PANELS; do
    echo "=== Panel $IDX ==="
    T_PANEL_START=$(date +%s)
    LOG="logs/cracked_glass_pan${IDX}_$(date +%Y%m%d_%H%M%S).log"

    if "$PYTHON" SyntheticDataGen/cracked_glass/cracked_glass_collection.py \
            --panel-idx "$IDX" \
            --output "$OUTPUT" \
            "$@" 2>&1 | tee "$LOG"; then
        DT=$(( $(date +%s) - T_PANEL_START ))
        echo "--- Panel $IDX done in ${DT}s ---"
    else
        echo "!!! Panel $IDX FAILED (see $LOG) !!!"
        FAILED_PANELS+=("$IDX")
    fi
    echo ""
done

T_TOTAL=$(( $(date +%s) - T_TOTAL_START ))
echo "=== All panels processed in $((T_TOTAL/60)) min ==="
echo "Output: $OUTPUT"
if [ ${#FAILED_PANELS[@]} -gt 0 ]; then
    echo "FAILED panels: ${FAILED_PANELS[*]}"
    echo "  Re-run individual panels with:"
    echo "    $PYTHON SyntheticDataGen/cracked_glass/cracked_glass_collection.py --panel-idx <N> --output $OUTPUT"
    exit 1
fi
echo "All 13 panels succeeded."
