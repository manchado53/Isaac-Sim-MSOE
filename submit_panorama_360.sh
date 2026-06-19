#!/bin/bash
# Convenience launcher for the panorama renders.
# Each job takes ~4-6 min on a T4.
#
# Usage:
#   ./submit_panorama_360.sh            # ground-level orbit of all 3 buildings
#   ./submit_panorama_360.sh innovation # just MSOEBuildingV4 (trees stripped)
#   ./submit_panorama_360.sh dierks     # just DierksHallV2
#   ./submit_panorama_360.sh viets      # just VietsField (ground-level)
#   ./submit_panorama_360.sh field      # ELEVATED drone-style orbit of Viets Field
#                                       # (uses field_panorama.py, not panorama_360.py)

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$SCRIPT_DIR"

ASSETS="assets/environments/msoe_building"

# Heavy pattern set — catches every flavor of site planting / entourage seen
# in Revit exports. Safe to apply even if patterns don't match (a non-match
# just logs "0 prims hidden").
FULL_CLUTTER="Tree,Planting,Vegetation,Entourage,RPC,Plant,Shrub,Bush,Landscape,Topography,Site_"
# Viets Field: keep the turf/field, strip only trees and RPCs.
VIETS_CLUTTER="Tree,Planting,RPC"

submit() {
    local tag="$1"; shift
    echo "[submit_panorama_360] sbatch $tag"
    sbatch --job-name="panorama-$tag" submit_isaac_job.sh \
        simulations/environment/panorama_360.py "$@"
}

target="${1:-all}"

case "$target" in
    innovation|all)
        submit "innovation" \
            --input "$ASSETS/MSOEBuildingV4.usd" \
            --hide-patterns "$FULL_CLUTTER"
        ;;
esac
case "$target" in
    dierks|all)
        submit "dierks" \
            --input "$ASSETS/DierksHallV2.usd" \
            --hide-patterns "$FULL_CLUTTER"
        ;;
esac
case "$target" in
    viets|all)
        submit "viets" \
            --input "$ASSETS/VietsField.usd" \
            --hide-patterns "$VIETS_CLUTTER"
        ;;
esac
case "$target" in
    field|field-only)
        echo "[submit_panorama_360] sbatch field-elevated"
        sbatch --job-name="panorama-field-elevated" submit_isaac_job.sh \
            simulations/environment/field_panorama.py \
            --input "$ASSETS/VietsField.usd" \
            --hide-patterns "$VIETS_CLUTTER"
        ;;
esac

echo ""
echo "Watch queue with:  squeue -u \$USER"
echo "Output videos:     videos/environment/<stem>_panorama_360.mp4"
