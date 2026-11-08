#!/usr/bin/env bash
set -euo pipefail

# ============================================================
# WM-mask phase 1: legacy regression + single-shell fallback
#
# Subject: alem, the input behind the existing out/20260909
# cohort (verified from its tracking provenance). This is a
# SINGLE-SHELL acquisition (b0 + b~1000, 116 dirs), so:
#
#   1. --wm-method legacy must reproduce the existing run
#      bit-for-bit where it should (FA maps, seed set) —
#      the thesis-baseline reproducibility guard.
#   2. --wm-method auto must fall back to legacy with the
#      recorded 'dam_inapplicable' reason, and produce an
#      identical WM mask to the legacy run.
#
# Reference outputs: the 20260909 run was made BEFORE the
# wm-mask feature, with --fa-thr 0.2 --seed-density 1
# --step-size 0.5 --sh-order 6 --npeaks 2 --rng-seed 42,
# an external brain mask, PDFs on.
# ============================================================

CSTTOOL=/home/alem/miniconda3/envs/csttool/bin/csttool
PYTHON=/home/alem/miniconda3/envs/csttool/bin/python

NIFTI=/home/alem/neurodata/fsl/alem/csttool/corrected/eddy_corrected_repol.nii.gz
BVAL=/home/alem/neurodata/fsl/alem/csttool/corrected/eddy_corrected_repol.bval
BVEC=/home/alem/neurodata/fsl/alem/csttool/corrected/eddy_corrected_repol.bvec
BRAIN_MASK=/home/alem/neurodata/fsl/alem/csttool/mask/brain_mask.nii.gz

# Existing pre-feature reference run (BIDS reorganized outputs).
REFERENCE=/home/alem/neurodata/out/20260909/sh-order-6/sub-control-sh-order-6

OUT_ROOT=/home/alem/neurodata/out/20260922/wm-regression

# The exact parameters of the reference run.
COMMON_ARGS=(
    --nifti "$NIFTI"
    --brain-mask "$BRAIN_MASK"
    --subject-id control-sh-order-6
    --fa-thr 0.20
    --seed-density 1
    --step-size 0.50
    --sh-order 6
    --fit-method WLS
    --npeaks 2
    --rng-seed 42
    --b0-threshold 50
    --generate-pdf
    --save-visualizations
)

mkdir -p "$OUT_ROOT"

for METHOD in legacy auto; do
    OUT="$OUT_ROOT/${METHOD}"
    REPORT="$OUT/sub-control-sh-order-6_pipeline_report.json"
    if [[ -f "$REPORT" ]]; then
        echo "SKIP: $OUT already completed"
        continue
    fi
    if [[ -e "$OUT" ]]; then
        echo "ERROR: $OUT exists without a completed report; remove it first." >&2
        exit 1
    fi
    echo "=== alem regression run: --wm-method $METHOD ==="
    "$CSTTOOL" run \
        "${COMMON_ARGS[@]}" \
        --out "$OUT" \
        --wm-method "$METHOD"
done

# ------------------------------------------------------------
# Verification: legacy reproduces the reference; auto falls
# back to legacy identically.
# ------------------------------------------------------------
echo
echo "=== Verifying regression ==="
"$PYTHON" - "$OUT_ROOT" "$REFERENCE" <<'PYEOF'
import json, sys
from pathlib import Path
import nibabel as nib
import numpy as np
from scipy.ndimage import binary_dilation

out_root, reference = Path(sys.argv[1]), Path(sys.argv[2])
fa_thr = 0.2
failures = []

def check(name, ok, detail=""):
    print(f"  [{'OK' if ok else 'FAIL'}] {name}" + (f" — {detail}" if detail else ""))
    if not ok:
        failures.append(name)

legacy = out_root / "legacy" / "sub-control-sh-order-6"
auto_ = out_root / "auto" / "sub-control-sh-order-6"

fa_new = nib.load(str(legacy / "dwi" / "sub-control-sh-order-6_space-orig_model-DTI_param-FA_dwimap.nii.gz"))
fa_ref = nib.load(str(reference / "dwi" / "sub-control-sh-order-6_space-orig_model-DTI_param-FA_dwimap.nii.gz"))
check("legacy FA is bit-identical to the 20260909 reference",
      np.array_equal(np.asarray(fa_new.dataobj), np.asarray(fa_ref.dataobj)))

brain = nib.load("/home/alem/neurodata/fsl/alem/csttool/mask/brain_mask.nii.gz")
brain = np.asarray(brain.dataobj) > 0
fa = np.asarray(fa_new.dataobj)
expected_wm = binary_dilation((fa > fa_thr) & brain, iterations=1)

def wm_of(sub_dir):
    p = sub_dir / "dwi" / "sub-control-sh-order-6_space-orig_desc-WM_mask.nii.gz"
    if not p.exists():
        # pre-BIDS stage location
        cands = sorted(sub_dir.rglob("*_wm_mask.nii.gz"))
        p = cands[0]
    return np.asarray(nib.load(str(p)).dataobj) > 0

wm_legacy, wm_auto = wm_of(legacy), wm_of(auto_)
check("legacy WM mask == dilate((FA > 0.2) & brain_mask), the historical formula",
      np.array_equal(wm_legacy, expected_wm))
check("auto WM mask is identical to legacy (single-shell fallback)",
      np.array_equal(wm_auto, expected_wm))

def sidecar_of(sub_dir, name):
    cands = sorted(sub_dir.rglob(name))
    return json.loads(cands[0].read_text())

sc = sidecar_of(legacy, "sub-control-sh-order-6_report_qc_strip.json")
check("legacy strip sidecar names the legacy source",
      sc.get("WmMaskSource") == "legacy", str(sc.get("WmMaskSource")))
sc_auto = sidecar_of(auto_, "sub-control-sh-order-6_report_qc_strip.json")
check("auto strip sidecar names the legacy source too (fallback)",
      sc_auto.get("WmMaskSource") == "legacy", str(sc_auto.get("WmMaskSource")))
trk_report = sidecar_of(auto_, "sub-control-sh-order-6_log-tracking.json")
wm_provenance = trk_report["tracking_parameters"]
check("auto records the dam_inapplicable fallback reason",
      "dam_inapplicable" in (wm_provenance.get("wm_fallback_reason") or ""),
      str(wm_provenance.get("wm_fallback_reason")))

n_ref = json.loads((reference / "reports" / "sub-control-sh-order-6_log-tracking.json")
                   .read_text())["streamline_stats"]["count"]
n_new = trk_report["streamline_stats"]["count"]
check("whole-brain streamline count unchanged vs reference",
      n_ref == n_new, f"ref={n_ref} new={n_new}")

print("  [info] compare the bilateral metrics JSONs by eye for the full profile:")
print(f"         new: {auto_}/reports  ref: {reference}/reports")

print()
if failures:
    print(f"REGRESSION FAILED: {failures}")
    sys.exit(1)
print("REGRESSION PASSED")
PYEOF