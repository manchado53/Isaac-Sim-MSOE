#!/bin/bash
#
# Setup Shared Assets
#
# Copies building assets from the shared /data/ai_club/isaac-sim-msoe/
# directory into your local repo so you can run the physics pipeline.
#
# Usage:
#   ./setup_shared_assets.sh           # Copy all shared assets
#   ./setup_shared_assets.sh --list    # Show what's available
#
# Shared assets are readable by all ai_club members. If you generate
# new physics USD files, copy them back with:
#   cp assets/environments/msoe_building/my_physics.usd /data/ai_club/isaac-sim-msoe/

set -euo pipefail

SHARED_DIR="/data/ai_club/isaac-sim-msoe"
LOCAL_ENV="assets/environments"
LOCAL_BUILDING="assets/environments/msoe_building"

if [ ! -d "$SHARED_DIR" ]; then
    echo "ERROR: Shared directory not found: $SHARED_DIR"
    echo "Make sure you have access to /data/ai_club/"
    exit 1
fi

if [ "${1:-}" = "--list" ]; then
    echo "Shared assets in $SHARED_DIR:"
    echo ""
    ls -lh "$SHARED_DIR"
    exit 0
fi

# Ensure local directories exist
mkdir -p "$LOCAL_ENV" "$LOCAL_BUILDING"

echo "========================================="
echo "Copying shared assets"
echo "========================================="
echo "From: $SHARED_DIR"
echo "To:   $LOCAL_ENV / $LOCAL_BUILDING"
echo ""

# Copy IFC and raw USD to environments/
for f in "$SHARED_DIR"/*.ifc "$SHARED_DIR"/*.usd; do
    [ -f "$f" ] || continue
    basename_f=$(basename "$f")

    # Physics USD files go to msoe_building/ subdirectory
    if [[ "$basename_f" == *_physics.usd ]] || [[ "$basename_f" == *_finalV1.usd ]]; then
        dest="$LOCAL_BUILDING/$basename_f"
    else
        dest="$LOCAL_ENV/$basename_f"
    fi

    if [ -f "$dest" ]; then
        local_size=$(stat -c%s "$dest")
        shared_size=$(stat -c%s "$f")
        if [ "$local_size" -eq "$shared_size" ]; then
            echo "  SKIP (same size): $basename_f"
            continue
        fi
    fi

    echo "  COPY: $basename_f -> $dest"
    cp "$f" "$dest"
done

echo ""
echo "========================================="
echo "Done! Your local assets:"
echo "========================================="
echo ""
echo "Raw inputs (for physics pipeline):"
ls -lh "$LOCAL_ENV"/*.ifc "$LOCAL_ENV"/*.usd 2>/dev/null || echo "  (none)"
echo ""
echo "Physics-ready USDs (for simulations):"
ls -lh "$LOCAL_BUILDING"/*.usd 2>/dev/null || echo "  (none)"
echo ""
echo "Run the physics pipeline:"
echo "  ./run_physics_pipeline.sh $LOCAL_ENV/msoe-engineering-innovation-USD.usd"
echo ""
echo "Or use the IFC pipeline (full conversion):"
echo "  ./run_physics_pipeline.sh $LOCAL_ENV/msoe-engineering-innovation-center.ifc"
