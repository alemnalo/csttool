#!/usr/bin/env bash
set -euo pipefail

# ============================================================
# WM-mask phase 4: external WM mask path on CDMD data
#
# v2 (2026-09-22): the strict cerebral-WM-only mask (aseg
# labels 2/41) produced ZERO CST streamlines: FA-threshold
# stopping terminates streamlines at the cortical rim, which
# the strict mask excludes from seeding, and the motor-ROI
# endpoint filter then found nothing (whole-brain 54k vs 107k
# streamlines, 0/62 streamlines reaching motor/brainstem
# ROIs). The pipeline behaved correctly and honestly — the
# failure is a scientific finding about the mask choice.
#
# The default mask now includes the labels CST tracking needs:
#   2  Left-Cerebral-White-Matter
#   41 Right-Cerebral-White-Matter
#   16 Brain-Stem
#   7  Left-Cerebellum-White-Matter
#   46 Right-Cerebellum-White-Matter
# plus ONE binary dilation to reach the GM/WM interface — the
# same rationale as the legacy mask's dilation, applied here
# deliberately and visibly IN THE SUPPLIED MASK, because the
# pipeline itself never edits an external mask.
#
# Set STRICT=1 to reproduce the conservative 2/41-only variant
# (kept as the documented negative case).
#
# Verifies afterwards that the persisted desc-WM_mask product
# is bit-identical to the supplied mask (proving "verbatim")
# and that the provenance names the external source.
# ============================================================

CSTTOOL=/home/alem/miniconda3/envs/csttool/bin/csttool
PYTHON=/home/alem/miniconda3/envs/csttool/bin/python

INPUT_ROOT=/home/alem/neurodata/in/cdmd
OUT_ROOT=/home/alem/neurodata/out/20260922/wm-external-mask-v3

SUBJECTS=(sub_005 sub_024)

# v3 (2026-09-22): the v2 'cst' mask (labels 2/41/16/7/46, 1 dilation) still
# starved the motor-ROI endpoint filter — 117k whole-brain streamlines but
# only 1/17 and 190/40 passing the motor ROIs, so 18 and 230 CST streamlines
# (vs ~1,000 with FA>0.2 legacy seeding). An anatomical WM mask excludes the
# cortical rim that FA-stopped, ROI-endpoint extraction needs. DILATE is now
# tunable: sweep DILATE=2 and DILATE=3 to test whether reaching the rim
# recovers the yield. The VARIANT tag carries the dilation so runs don't
# collide.
DILATE=${DILATE:-2}
STRICT=${STRICT:-0}
if [[ "$STRICT" == "1" ]]; then
    LABELS=(2 41)
    DILATE=0
    VARIANT="strict"
else
    LABELS=(2 41 16 7 46)
    VARIANT="cst-d${DILATE}"
fi

mkdir -p "$OUT_ROOT"

for SUBJECT in "${SUBJECTS[@]}"; do
    DWI_DIR="$INPUT_ROOT/$SUBJECT/dwi"
    NIFTI="$DWI_DIR/${SUBJECT}_dwi.nii.gz"
    BVAL="$DWI_DIR/${SUBJECT}_dwi.bval"
    BVEC="$DWI_DIR/${SUBJECT}_dwi.bvec"
    BRAIN_MASK="$DWI_DIR/${SUBJECT}_dwi_brainmask.nii.gz"
    ASEG="$DWI_DIR/${SUBJECT}_dwi_aparc+aseg.nii.gz"
    WM_MASK="$OUT_ROOT/${SUBJECT}_aseg_wm_${VARIANT}.nii.gz"

    for REQUIRED in "$NIFTI" "$BVAL" "$BVEC" "$BRAIN_MASK" "$ASEG"; do
        if [[ ! -f "$REQUIRED" ]]; then
            echo "ERROR: missing input $REQUIRED" >&2
            exit 1
        fi
    done

    # --- Build the external WM mask (once per subject/variant) ---
    if [[ ! -f "$WM_MASK" ]]; then
        echo "=== $SUBJECT: building WM mask (aseg labels ${LABELS[*]}, dilate=$DILATE) ==="
        "$PYTHON" - "$ASEG" "$WM_MASK" "$DILATE" "${LABELS[@]}" <<'PYEOF'
import sys
import json
from pathlib import Path
import nibabel as nib
import numpy as np
from scipy.ndimage import binary_dilation

aseg_img = nib.load(sys.argv[1])
out_path = Path(sys.argv[2])
dilate = int(sys.argv[3])
labels = tuple(int(x) for x in sys.argv[4:])

aseg = np.asarray(aseg_img.dataobj)
wm = np.isin(aseg, labels)
n_before = int(wm.sum())
if dilate:
    wm = binary_dilation(wm, iterations=dilate)
nib.save(nib.Nifti1Image(wm.astype(np.uint8), aseg_img.affine), str(out_path))
print(f"  WM voxels: {int(wm.sum()):,} (labels {labels}, dilation {dilate})")
# A sidecar so the mask documents its own construction — the pipeline
# will not add provenance for a file it did not make.
out_path.with_suffix('.json').write_text(json.dumps({
    "Description": "External WM mask built from CDMD aparc+aseg (DWI space)",
    "Labels": list(labels),
    "Dilation": dilate,
    "VoxelsBeforeDilation": n_before,
    "Voxels": int(wm.sum()),
}))
PYEOF
    else
        echo "SKIP: $WM_MASK already exists"
    fi

    SID="${SUBJECT}-extwm-${VARIANT}"
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

    echo "=== External WM mask run: $SID ==="
    "$CSTTOOL" run \
        --nifti "$NIFTI" \
        --brain-mask "$BRAIN_MASK" \
        --wm-mask "$WM_MASK" \
        --out "$OUT" \
        --subject-id "$SID" \
        --input-corrected topup-eddy \
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

# ------------------------------------------------------------
# Verification: verbatim use + provenance. Globs are
# suffix-based (BIDS reorg prefixes 'sub-').
# ------------------------------------------------------------
echo
echo "=== Verifying external-mask verbatim use ==="
"$PYTHON" - "$OUT_ROOT" "$VARIANT" "${SUBJECTS[@]}" <<'PYEOF'
import json, sys
from pathlib import Path
import nibabel as nib
import numpy as np

out_root = Path(sys.argv[1])
variant = sys.argv[2]
subjects = sys.argv[3:]
failures = []

for subject in subjects:
    sid = f"{subject}-extwm-{variant}"
    sub_dir = out_root / sid
    supplied = np.asarray(nib.load(
        str(out_root / f"{subject}_aseg_wm_{variant}.nii.gz")).dataobj) > 0
    persisted = sorted(sub_dir.rglob("*desc-WM_mask.nii.gz")) \
        or sorted(sub_dir.rglob("*_wm_mask.nii.gz"))
    if not persisted:
        print(f"  [FAIL] {sid}: no WM mask product under {sub_dir}")
        failures.append(sid)
        continue
    used = np.asarray(nib.load(str(persisted[0])).dataobj) > 0
    ok = used.shape == supplied.shape and np.array_equal(used, supplied)
    print(f"  [{'OK' if ok else 'FAIL'}] {sid}: persisted mask is bit-identical to the supplied mask")
    if not ok:
        failures.append(sid)

    reports = sorted(sub_dir.rglob("*_log-tracking.json"))
    if not reports:
        print(f"  [FAIL] {sid}: no tracking report")
        failures.append(sid + "/report")
        continue
    trk = json.loads(reports[0].read_text())
    tp = trk["tracking_parameters"]
    ok_src = tp.get("wm_mask_source") == "external"
    print(f"  [{'OK' if ok_src else 'FAIL'}] {sid}: provenance names external "
          f"(got {tp.get('wm_mask_source')}, path={tp.get('wm_mask_path')})")
    if not ok_src:
        failures.append(sid + "/source")
    print(f"  [info] {sid}: whole-brain streamlines "
          f"{trk['streamline_stats']['count']:,}")

if failures:
    print(f"\nEXTERNAL-MASK VERIFICATION FAILED: {failures}")
    sys.exit(1)
print("\nEXTERNAL-MASK VERIFICATION PASSED")
PYEOF