#!/usr/bin/env bash
set -euo pipefail

# ============================================================
# WM-mask phase 3: DAM calibration on multi-shell CDMD data
#
# The alem validation subject is single-shell (DAM cannot run
# there). The CDMD full-shell acquisitions are the DAM
# candidates: 8 shells (50/200/350/800/950/1500/2300 s/mm^2,
# 32 dirs each) + 50 b0 volumes, with a CDMD-provided DWI-space
# brain mask and a FreeSurfer aparc+aseg in the same space —
# the latter gives a trustworthy WM reference to judge DAM
# against (Dice, volume, L/R symmetry, CST-ROI coverage).
#
# v2 (2026-09-22 fixes):
#   * The first sweep's DAM calls all failed with
#     'Input y contains NaN' — the CDMD data carries negative
#     eddy-interpolation values, which break DIPY's per-voxel
#     log fit. csttool now sanitizes the DAM input (NaN -> 0,
#     negatives clipped to 0) inside segment_wm_dam, so this
#     sweep actually tests DAM. The stale v1 runs (which fell
#     back to legacy) are left untouched; this writes to a
#     fresh -v2 tree.
#   * Analysis report glob fixed: the BIDS reorganiser prefixes
#     'sub-', so reports are sub-<sid>_*.json, not <sid>_*.json
#     (the v1 IndexError).
# v3 (2026-09-22): the v2 sweep {0.3,0.5,0.7} produced near-empty masks
# (233-5,524 voxels vs ~82,000 legacy). The cause is the acquisition, not a
# bug: the CDMD series has 17 b-values up to b=17,800, and on that ultra-
# multi-shell protocol DIPY's DAM slope P sits far above the value its example
# uses. Measured on sub_005: P in the FA>0.2 territory has median ~0.92
# (25th 0.84, 75th 1.03), so a 0.5 cut keeps ~3% of WM. This sweep therefore
# targets the measured regime: {0.9, 1.0, 1.1}. Judge the masks against the
# aseg WM reference (Dice/coverage in the analysis below) — precision was
# already high at 0.5, recall is what these thresholds recover.
#
# NOTE ON TRACKING: with the mask as the SEED territory, a too-small mask
# collapses the whole-brain tractogram (v2: 447-12k streamlines, 0 surviving
# the 20mm filter). Whether the mask should seed OR only gate peak estimation
# is an open design question (see the session notes); this script tests the
# mask as-seed, the current behaviour.

CSTTOOL=/home/alem/miniconda3/envs/csttool/bin/csttool
PYTHON=/home/alem/miniconda3/envs/csttool/bin/python

INPUT_ROOT=/home/alem/neurodata/in/cdmd
OUT_ROOT=/home/alem/neurodata/out/20260922/wm-dam-calibration-v3
# Editable cohort. sub_005 and sub_024 verified present with
# dwi/bval/bvec/brainmask/aparc+aseg. Add more sub_XXX as needed.
SUBJECTS=(sub_005 sub_024)
THRESHOLDS=(0.9 1.0 1.1)

mkdir -p "$OUT_ROOT"

for SUBJECT in "${SUBJECTS[@]}"; do
    DWI_DIR="$INPUT_ROOT/$SUBJECT/dwi"
    NIFTI="$DWI_DIR/${SUBJECT}_dwi.nii.gz"
    BVAL="$DWI_DIR/${SUBJECT}_dwi.bval"
    BVEC="$DWI_DIR/${SUBJECT}_dwi.bvec"
    BRAIN_MASK="$DWI_DIR/${SUBJECT}_dwi_brainmask.nii.gz"
    ASEG="$DWI_DIR/${SUBJECT}_dwi_aparc+aseg.nii.gz"

    for REQUIRED in "$NIFTI" "$BVAL" "$BVEC" "$BRAIN_MASK"; do
        if [[ ! -f "$REQUIRED" ]]; then
            echo "ERROR: missing input $REQUIRED" >&2
            exit 1
        fi
    done
    if [[ ! -f "$ASEG" ]]; then
        echo "WARNING: no aparc+aseg for $SUBJECT — Dice stats will be skipped" >&2
    fi

    # Shell report, so the run log states what DAM actually saw.
    echo "=== $SUBJECT shells: ==="
    tr -s ' \t\n' '\n' < "$BVAL" | sort -n | uniq -c

    for THR in "${THRESHOLDS[@]}"; do
        SID="${SUBJECT}-dam-${THR/./-}"
        OUT="$OUT_ROOT/$SID"
        REPORT="$OUT/${SID}_pipeline_report.json"
        if [[ -f "$REPORT" ]]; then
            echo "SKIP: $OUT already completed"
            continue
        fi
        if [[ -e "$OUT" ]]; then
            echo "ERROR: $OUT exists without a completed report; remove it first." >&2
            exit 1
        fi
        echo "=== DAM run: $SID (wm_threshold=$THR) ==="
        "$CSTTOOL" run \
            --nifti "$NIFTI" \
            --brain-mask "$BRAIN_MASK" \
            --out "$OUT" \
            --subject-id "$SID" \
            --input-corrected topup-eddy \
            --wm-method dam \
            --dam-wm-threshold "$THR" \
            --fa-thr 0.20 \
            --seed-density 1 \
            --step-size 0.50 \
            --sh-order 6 \
            --fit-method WLS \
            --npeaks 2 \
            --rng-seed 42 \
            --b0-threshold 50 \
            --dilate-brainstem 2 \
            --dilate-motor 1 \
            --min-length 20 \
            --max-length 200 \
            --generate-pdf \
            --save-visualizations
    done
done

# ------------------------------------------------------------
# Analysis: per (subject, threshold) mask statistics and Dice
# against the aseg WM reference. Globs are suffix-based: the
# BIDS reorganiser prefixes 'sub-' to the subject id, so exact
# '{sid}_*' globs miss (the v1 IndexError).
# ------------------------------------------------------------
echo
echo "=== DAM calibration analysis ==="
"$PYTHON" - "$OUT_ROOT" "${SUBJECTS[@]}" <<'PYEOF'
import json, sys
from pathlib import Path
import nibabel as nib
import numpy as np

out_root = Path(sys.argv[1])
subjects = sys.argv[2:]

# FreeSurfer aseg labels for cerebral white matter.
ASEG_WM_LABELS = (2, 41)

rows = []
for subject in subjects:
    in_dwi = Path("/home/alem/neurodata/in/cdmd") / subject / "dwi"
    aseg_wm = None
    aseg_path = in_dwi / f"{subject}_dwi_aparc+aseg.nii.gz"
    if aseg_path.exists():
        aseg = np.asarray(nib.load(str(aseg_path)).dataobj)
        aseg_wm = np.isin(aseg, ASEG_WM_LABELS)
    brain = np.asarray(nib.load(
        str(in_dwi / f"{subject}_dwi_brainmask.nii.gz")).dataobj) > 0

    for thr in ("0.9", "1.0", "1.1"):
        sid = f"{subject}-dam-{thr.replace('.', '-')}"
        sub_dir = out_root / sid
        if not sub_dir.is_dir():
            print(f"  {sid}: run directory missing, skipped")
            continue
        wm_cands = (sorted(sub_dir.rglob("*desc-WM_mask.nii.gz"))
                    or sorted(sub_dir.rglob("*_wm_mask.nii.gz")))
        reports = (sorted(sub_dir.rglob("*_log-tracking.json"))
                   or sorted(sub_dir.rglob("*_tracking_report.json")))
        if not wm_cands or not reports:
            print(f"  {sid}: INCOMPLETE (wm product: {bool(wm_cands)}, "
                  f"tracking report: {bool(reports)})")
            continue
        wm = np.asarray(nib.load(str(wm_cands[0])).dataobj) > 0
        trk = json.loads(reports[0].read_text())
        tp = trk["tracking_parameters"]
        source = tp.get("wm_mask_source")
        reason = tp.get("wm_fallback_reason")
        if source != "dam":
            print(f"  {sid}: INCONCLUSIVE — source={source}, reason={reason}")
        n_wm = int(wm.sum())
        mid = wm.shape[0] // 2  # L/R split on the voxel grid (approximate)
        lr_ratio = wm[:mid].sum() / max(wm[mid:].sum(), 1)
        # CST ROI coverage: fraction of extraction ROI voxels inside the mask.
        roi_cov = None
        dseg = sorted(sub_dir.rglob("*desc-CSTroi_dseg.nii.gz")) \
            or sorted(sub_dir.rglob("*_roi_dseg.nii.gz")) \
            or sorted(sub_dir.rglob("extraction/nifti/*_roi_cst_combined.nii.gz"))
        if dseg:
            d = np.asarray(nib.load(str(dseg[0])).dataobj)
            roi = d > 0
            if roi.any():
                roi_cov = float((wm & roi).sum() / int(roi.sum()))
        dice = None
        if aseg_wm is not None and aseg_wm.shape == wm.shape:
            inter = int((wm & aseg_wm).sum())
            dice = 2.0 * inter / (n_wm + int(aseg_wm.sum()))
        rows.append({
            "subject": subject, "threshold": float(thr),
            "source": source,
            "fallback_reason": reason,
            "dam_input_sanitized": (tp.get("wm_dam_params") or {}).get(
                "input_sanitized"),
            "wm_voxels": n_wm,
            "fraction_of_brain": round(n_wm / max(int(brain.sum()), 1), 4),
            "lr_ratio": round(float(lr_ratio), 3),
            "cst_roi_coverage": None if roi_cov is None else round(roi_cov, 4),
            "dice_vs_aseg_wm": None if dice is None else round(dice, 4),
            "streamlines": trk["streamline_stats"]["count"],
        })

table = out_root / "dam_calibration.json"
table.write_text(json.dumps(rows, indent=2))
print(json.dumps(rows, indent=2))
print(f"\nWrote {table}")
n_dam = sum(1 for r in rows if r["source"] == "dam")
print(f"\n{n_dam}/{len(rows)} runs actually used DAM. Any 'INCONCLUSIVE' line "
      "above means DAM fell back — check its fallback_reason before reading "
      "the numbers as DAM behaviour.")
PYEOF