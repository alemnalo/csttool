# Reproducing the thesis (v0.5.0) pipeline

The headline results in the master's thesis were produced with a different denoiser and
a different correction chain than csttool's current defaults. Current defaults do
**not** reproduce those numbers, and are not meant to — MPPCA replaced NLMeans as the
default deliberately (see [Design Decisions](../explanation/design-decisions.md#why-mppca-is-the-default-denoiser)).

This page records the historical recipe so it stays runnable. It is documentation only:
there is no `--preset thesis`, because freezing one historical recipe into the CLI
surface adds a second way to set the same knobs forever.

## What the thesis actually ran

The canonical headline arm (`fsl_nlm` in the sensitivity study) chained the steps in
this order:

```
csttool NLMeans denoise + Gibbs unringing
    → FSL topup
    → FSL eddy
    → csttool masking, tractography, extraction, metrics
```

Note the ordering: csttool's denoising came **first**, and the FSL distortion and
eddy-current correction ran on the denoised series. That was a deliberate finding of
the study — on this acquisition, denoising after `eddy` inflated MD severely, and
NLMeans was robust in every ordering while the then-current Patch2Self configuration
was not.

Tracking baseline: `--fa-thr 0.20 --seed-density 2 --step-size 0.5 --min-length 20
--extraction-method passthrough`.

## Running it today

The chain crosses tools, so it is two csttool invocations with FSL in between. Each
invocation writes its own report, and the two reports together are the provenance.

**Step 1 — csttool preprocessing on the raw series:**

```bash
csttool preprocess \
    --nifti raw_dwi.nii.gz \
    --out work/preproc \
    --denoise-method nlmeans --coil-count 4 \
    --unring \
    --input-corrected none \
    --save-visualizations
```

`--input-corrected none` is the accurate declaration here: nothing had been done to the
input yet.

**Step 2 — FSL `topup` and `eddy`** on `work/preproc/*_dwi_preproc_nomc.nii.gz`, outside
csttool. Make sure `eddy` writes its rotated `.bvec` (`*.eddy_rotated_bvecs`) and carry
that file forward — csttool reads whatever sidecars sit next to the NIfTI you hand it.

**Step 3 — csttool tractography onward, on the corrected series:**

```bash
csttool run \
    --nifti eddy_outlier_free_data.nii.gz \
    --input-corrected topup-eddy \
    --extraction-method passthrough \
    --fa-thr 0.2 --seed-density 2 --step-size 0.5 \
    --out results/
```

No `--preprocess` here: csttool already did its part in step 1, and preprocessing again
would denoise twice. The declaration records that the input arrived corrected.

## If your external correction runs first

The more common workflow is the other order — correct with FSL or QSIPrep, then hand the
result to csttool. That is a single command:

```bash
csttool run \
    --nifti corrected_dwi.nii.gz \
    --preprocess --denoise-method nlmeans --coil-count 4 --unring \
    --input-corrected topup-eddy \
    --extraction-method passthrough --fa-thr 0.2 --seed-density 2 \
    --out results/
```

The stage ledger in the preprocessing report puts the declared external correction
before csttool's own stages, so the chronology serialises as it happened. See
[Output formats](../reference/output-formats.md#preprocessing-report).

## What will not reproduce exactly

- **The denoiser default changed.** Without `--denoise-method nlmeans` you get MPPCA and
  different numbers.
- **Motion correction now rotates b-vectors.** The thesis sweep configuration enabled
  motion correction, and at the time csttool shipped the *original* b-vectors alongside
  the resampled volumes. That was a defect, now fixed. A re-run with
  `--perform-motion-correction` will therefore produce different — and more correct —
  tensors than the original run. There is no flag to restore the old behaviour, and
  there should not be.
- **The registration optimiser is not bit-reproducible**, so motion-corrected runs are
  outside csttool's determinism guarantees regardless. See
  [Limitations](../explanation/limitations.md#motion-correction-an-approximation-and-not-a-deterministic-one).

Everything else — seeding, tracking, extraction, metrics — is deterministic under the
default seed.
