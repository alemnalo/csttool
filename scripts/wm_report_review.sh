#!/usr/bin/env bash
set -euo pipefail

# ============================================================
# WM-mask phase 5: report/QC print review collector
#
# Gathers the PDF report and the QC-strip PNG + sidecar of one
# run per WM source (legacy, auto-fallback, dam, external) into
# a single review directory, so the print review — one A4 page,
# WM panel legible at 48 mm, key fits whatever it says — can be
# done side by side.
#
# v2: collects from the -v2 trees (the v1 DAM runs fell back to
# legacy before the input-sanitizer fix; the v1 external runs
# used the strict mask that produced zero CST streamlines).
#
# NOTE: the WM panel switched to a saturated magenta fill +
# same-hue outline after the first print review showed the
# muted blue fill disappearing against the FA. Strips rendered
# before that change (e.g. the phase-1 alem runs) still show
# the old colour — rerun wm_legacy_regression_alem.sh into a
# fresh OUT_ROOT if you want the alem strips in the new style.
#
# Run AFTER the phase scripts. Destructive only inside its own
# output directory.
# ============================================================

OUT_ROOT_20260922=/home/alem/neurodata/out/20260922
REVIEW_ROOT="$OUT_ROOT_20260922/wm-report-review"

mkdir -p "$REVIEW_ROOT"

collect() {
    local src_sub_dir="$1"   # a completed run's sub-*/ BIDS directory
    local label="$2"         # review label
    local dest="$REVIEW_ROOT/$label"
    local strip sidecar pdf

    if [[ ! -d "$src_sub_dir" ]]; then
        echo "SKIP $label: $src_sub_dir does not exist"
        return 0
    fi
    mkdir -p "$dest"

    strip=$(find "$src_sub_dir" -name "*_qc-qc-strip.png" | head -1)
    pdf=$(find "$src_sub_dir" -name "*_report.pdf" | head -1)

    if [[ -n "$strip" ]]; then
        cp "$strip" "$dest/"
        sidecar="${strip%.png}.json"
        [[ -f "$sidecar" ]] && cp "$sidecar" "$dest/"
        echo "  $label: strip -> $dest/$(basename "$strip")"
    else
        echo "  $label: no QC strip PNG found under $src_sub_dir"
    fi
    if [[ -n "$pdf" ]]; then
        cp "$pdf" "$dest/"
        echo "  $label: pdf  -> $dest/$(basename "$pdf")"
    else
        echo "  $label: no report PDF found under $src_sub_dir"
    fi
}

# Phase 1 (alem, single-shell): legacy and the auto fallback.
collect "$OUT_ROOT_20260922/wm-regression/legacy/sub-control-sh-order-6"        legacy
collect "$OUT_ROOT_20260922/wm-regression/auto/sub-control-sh-order-6"          auto

# Phase 3 (cdmd, multi-shell): the DAM threshold sweep (v3 tree, thresholds
# recalibrated to the measured P regime).
collect "$OUT_ROOT_20260922/wm-dam-calibration-v3/sub_005-dam-0-9/sub_005-dam-0-9" "dam-0-9"
collect "$OUT_ROOT_20260922/wm-dam-calibration-v3/sub_005-dam-1-0/sub_005-dam-1-0" "dam-1-0"
collect "$OUT_ROOT_20260922/wm-dam-calibration-v3/sub_005-dam-1-1/sub_005-dam-1-1" "dam-1-1"

# Phase 4 (cdmd): the external aseg WM mask, v3 'cst-d2' variant by default
# (DILATE=2). Adjust the tag if you swept other dilations.
collect "$OUT_ROOT_20260922/wm-external-mask-v3/sub_005-extwm-cst-d2/sub_005-extwm-cst-d2" "external-sub_005"
collect "$OUT_ROOT_20260922/wm-external-mask-v3/sub_024-extwm-cst-d2/sub_024-extwm-cst-d2" "external-sub_024"

echo
echo "=== Review directory ==="
find "$REVIEW_ROOT" -type f | sort
echo
echo "Open each PDF and check:"
echo "  1. exactly one A4 page still holds (the strip height is aspect-derived)"
echo "  2. WM panel legible at 48 mm: magenta fill + outline clearly visible over FA"
echo "  3. key names the source: white matter mask (legacy|dam|external)"
echo "  4. degraded slot text 'WM mask not produced' only where expected"
echo "  5. sidecar WmMaskSource matches what the key says"