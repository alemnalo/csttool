# Limitations and Known Issues

This page documents the known limitations of csttool and important considerations for interpreting results.

---

## Critical Technical Risks

### Coordinate System Validation

**The single most important technical risk** in CST extraction is coordinate system mismatch between the input tractogram and FA map. This can lead to **anatomically plausible but incorrect results** - the output may look reasonable but represent the wrong anatomy.

#### Why This Matters

- Tractograms can be stored in voxel coordinates (indices) or world coordinates (mm)
- Different software packages use different conventions
- A mismatch produces streamlines that appear correct but are spatially misaligned
- Visual inspection may not reveal the problem if both hemispheres are equally affected

#### Mitigations in csttool

Starting with version 0.4.0, csttool performs automatic coordinate validation:

1. **Bounding box check**: Verifies streamline coordinates fall within FA volume bounds
2. **Unit detection**: Flags coordinates that appear to be voxel indices rather than mm
3. **Orientation verification**: Checks RAS orientation consistency

If validation fails, extraction will stop with an error unless `--skip-coordinate-validation` is explicitly passed.

#### Best Practices

- Always use tractograms generated in the same processing pipeline as your FA map
- If using external tractograms, verify coordinate systems match
- Review QC visualizations to confirm anatomical plausibility

---

## Registration Limitations

### No Automatic Acceptance Criteria

csttool does not automatically validate registration quality. The atlas-to-subject registration is a heuristic process that can fail silently.

**You should manually review registration QC images** in the output `visualizations/` directory:

- Check that MNI template contours align with subject anatomy
- Verify ROI masks fall on expected anatomical structures
- Flag subjects with poor registration for manual review

This is an intentional design choice: automated acceptance thresholds often reject acceptable registrations or accept poor ones. Expert visual review remains the gold standard for registration QC.

### Factors Affecting Registration

Registration quality can be compromised by:

- Large lesions or resection cavities
- Severe atrophy
- Motion artifacts in the FA map
- Non-standard head positioning

---

## Interpretation of Results

### CST Candidate Bundles

The streamlines output by csttool are best described as **CST candidate bundles**, not definitive corticospinal tract reconstructions.

This is because:

1. DTI tractography cannot resolve crossing fibers (e.g., SLF crossing CST at corona radiata)
2. Atlas-based ROI placement has inherent spatial uncertainty
3. The pipeline uses motor cortex and brainstem ROIs but does not explicitly constrain internal capsule, cerebral peduncle, or pyramidal decussation
4. Multiple descending motor-related tracts may be included

The extracted bundle is a **reproducible, rule-based proxy** for CST that is suitable for:

- Comparative analysis between hemispheres
- Longitudinal tracking within subjects
- Group-level statistical analysis

It should not be interpreted as a histologically pure CST reconstruction.

### Clinical Use

csttool outputs are intended for **research and exploratory analysis**. They should not be used as the sole basis for clinical decisions without additional validation.

---

## Data Requirements

### Required Units

- **Tractogram coordinates**: Must be in millimeters (world space), not voxel indices
- **FA/MD maps**: Standard NIfTI with valid affine matrix

See [Data Requirements](../getting-started/data-requirements.md) for detailed input specifications.

### Acquisition Coverage

CST extraction requires coverage from:

- Motor cortex (precentral gyrus)
- Through internal capsule
- Down to brainstem

Incomplete field of view will cause extraction to fail or produce incomplete results.

---

## CST Streamline Count Asymmetry

Atlas-based motor cortex ROIs (Harvard-Oxford, label 7/107: precentral gyrus) are defined
in MNI space and warped to subject space. The warped ROI centres do not land at identical
positions relative to the GM/WM boundary on each hemisphere. This causes the passthrough
and roi-seeded methods to yield different streamline counts for left vs right CST, even in
neurologically healthy subjects.

**Observed on in-vivo 3T data:**

| Method | R/L ratio | LI (streamlines) |
|--------|-----------|-----------------|
| Passthrough (NLMeans) | 1.29 | −0.128 |
| ROI-seeded (forward only) | 0.48 | +0.347 |
| Bidirectional | 1.00 | +0.002 |

A systematic audit confirmed the underlying tract is bilaterally symmetric (brainstem-seeded
reverse tracking: R/L = 0.987). The streamline count asymmetry in passthrough and roi-seeded
is a direction-dependent seeding artifact, not a structural finding.

**Mitigation:** Use `--extraction-method bidirectional` when bilateral symmetry matters.
All diffusion metrics (FA, MD, RD, AD) are unaffected and remain symmetric regardless of
extraction method.

See `docs/fixes/bidirectional_seeding.md` in the companion `csttool-devlog` repository for
the full technical analysis.

---

## Bidirectional Seeding — Known Limitation

The bilateral symmetry normalization in the bidirectional method enforces identical
streamline counts per side (`n_target = min` of all four pass counts). This means the
final count is bounded by the smallest of the four passes — typically the forward pass
from the more constrained hemisphere. In practice this produces substantially fewer
streamlines than passthrough, which is expected: only streamlines confirmed by both
directions are retained.

The method assumes the true CST is bilaterally symmetric. If a genuine structural
asymmetry exists (e.g. lesion, atrophy, surgical resection), bidirectional seeding will
undercount the more connected side. In clinical populations with known motor system
pathology, passthrough with manual laterality assessment is more appropriate.

---

## Preprocessing

### csttool does not correct distortion or eddy currents

`csttool preprocess` performs denoising, brain masking, optional Gibbs unringing and
optional **affine, between-volume motion correction**. That is the whole of it. There
is no susceptibility-distortion correction (no TOPUP-equivalent, no fieldmap or
fieldmapless method), no eddy-current model, no outlier-slice detection or
replacement, no slice-to-volume estimation and no bias-field correction.

Those corrections matter, and for most datasets they should be done — with FSL, QSIPrep
or an equivalent — **before** csttool sees the data. csttool's own preprocessing is off
by default precisely because the common case is data another tool has already corrected.

### Correction history is declared, not verified

Because csttool cannot detect what was already done to a DWI, `--input-corrected`
records the user's declaration and nothing more. Every such record carries
`verified_by_csttool: false`, and there is no code path that can set it true. A
declaration changes no processing decision; it exists so the provenance says what
happened rather than assuming it.

If you make no declaration and let csttool pass the data through untouched, a run-time
advisory says so, and the report records `external_correction.declared: "unknown"` —
which is deliberately distinct from `"none"`.

### Motion correction: an approximation, and not a deterministic one

When `--perform-motion-correction` is used:

- The correction is **affine and between-volume**. Within-volume (slice-to-volume)
  motion is not modelled, and the affine fit's shear and scale components are
  discarded by the polar decomposition used to extract the rotation — the same
  approximation class as FSL eddy's rotation propagation.
- b-vectors are rotated by the estimated per-volume transforms and the rotated `.bvec`
  is written alongside the output (Leemans & Jones 2009). **Before this was
  implemented, motion-corrected runs shipped unrotated gradients**; results produced by
  older versions with this flag should be regenerated.
- The registration optimiser is not guaranteed bit-reproducible across runs or
  platforms, so `--perform-motion-correction` is an **exception to csttool's
  determinism guarantees**. The determinism test suite does not cover the
  preprocessing stage.
- If the correction or the gradient rotation fails, the run continues with the
  uncorrected data and the original gradients — always a mutually consistent pair —
  and the report records `motion_correction_requested: true` alongside
  `motion_correction: false`.

### Batch preprocessing in v0.5.0 and earlier

`csttool batch` documented preprocessing as enabled by default, but the setting never
reached the worker process: every subject was tracked on raw data. Batch results
produced by v0.5.0 and earlier are not comparable with results produced now. Re-run
affected cohorts, or pass `--no-preprocessing` to reproduce the old behaviour
deliberately.

---

## Software Dependencies

csttool relies on:

- **DIPY** for tractography and registration
- **nilearn** for Harvard-Oxford atlas access
- **nibabel** for NIfTI handling
- **matplotlib** for visualizations

Version incompatibilities may cause unexpected behavior. See [Installation](../getting-started/installation.md) for tested versions.
