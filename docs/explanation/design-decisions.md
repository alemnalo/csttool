# Design Decisions

This page explains the reasoning behind non-obvious choices in csttool's architecture
and algorithms.

---

## Extraction methods: why four options exist

csttool offers four extraction methods, each trading off sensitivity, specificity, and
computational cost differently.

| Method | Input needed | L/R symmetric | Speed | Best for |
|--------|-------------|--------------|-------|----------|
| `passthrough` | Tractogram | Moderate (|LI| ≈ 0.1) | Fast | Cohort studies |
| `endpoint` | Tractogram | Moderate | Fast | Strict anatomical criterion |
| `roi-seeded` | Raw DWI | Poor (|LI| ≈ 0.35) | Moderate | Dense reconstruction |
| `bidirectional` | Raw DWI | Excellent (|LI| ≈ 0.002) | Slow | Single-subject symmetry |

---

## Bidirectional seeding: why not just use passthrough?

Passthrough filters a whole-brain tractogram for streamlines that traverse both the
brainstem and the motor cortex ROI. It is fast and works well for cohorts, but it
produces a modest streamline count asymmetry (|LI| ≈ 0.1 on typical data) because the
atlas-warped motor cortex ROI lands at a slightly different position relative to the
GM/WM boundary on each hemisphere.

A four-phase systematic audit on in-vivo 3T data confirmed:

1. The data quality is symmetric (no L/R signal imbalance across 71 DWI volumes)
2. Registration quality is symmetric (Jacobian determinant: L 1.000 ± 0.388, R 0.999 ± 0.330)
3. Motor ROI sizes and FA microstructure are symmetric (1,149 vs 1,135 FA > 0.2 voxels)
4. The underlying tract is symmetric — brainstem-seeded reverse tracking produces R/L = 0.987

The asymmetry is direction-dependent: passthrough gives R > L (LI = −0.128) while
roi-seeded gives L > R (LI = +0.347). An asymmetry that reverses sign with seeding
direction is the hallmark of a cortical interface placement artifact, not anatomy.

**Bidirectional seeding** eliminates this by:
- Running a forward pass (motor → brainstem) and a reverse pass (brainstem → motor)
- Retaining only forward streamlines whose count is bounded by the reverse count per side
- Enforcing the same bilateral target count (minimum across all four pass counts)
- Selecting from each forward bundle the streamlines with highest spatial overlap with
  the reverse density map

Result: LI = +0.002 — matches the brainstem-seeded ground truth (LI = +0.007).

Full technical write-up: [Bidirectional seeding — motivation and validation](../fixes/bidirectional_seeding.md)

---

## Why `bidirectional` is `run`-only, not available in `csttool extract`

`csttool extract` takes a pre-computed whole-brain tractogram as input. Bidirectional
seeding requires re-running tractography from two separate seed regions (motor cortex and
brainstem), which demands the raw DWI data. This is the same constraint as `roi-seeded`,
which has always been `run`-only.

Making it work with a pre-computed tractogram would require a different algorithm — for
example, filtering the existing tractogram by both endpoint regions and applying a
spatial overlap criterion. This is a valid future direction but would produce different
(and likely less accurate) results than the full bidirectional tracking approach.

---

## Why ODF parameters differ between passthrough and roi-seeded / bidirectional

The whole-brain tracking step (used by passthrough) uses stricter ODF parameters
(`relative_peak_threshold = 0.8`, `min_separation_angle = 45°`, `npeaks = 1`) to produce
a compact, high-quality whole-brain tractogram with minimal false connections.

The ROI-seeded and bidirectional methods use more permissive parameters
(`relative_peak_threshold = 0.5`, `min_separation_angle = 25°`) inherited from the
`roi_seeded_tracking` module. More permissive parameters allow the tracker to follow
complex crossing regions near the motor cortex and brainstem, which increases yield from
dense focal seeding.

This is an intentional asymmetry: whole-brain seeding needs conservative filtering to
keep tractogram size manageable; ROI seeding benefits from more flexibility because false
connections are later filtered by the ROI traversal criterion.

---

## Why `--extraction-method passthrough` is the default

Passthrough is the best balance of sensitivity, speed, and robustness for the common
case (cohort studies, first-time users). It works with a pre-computed tractogram (no
re-tracking needed), handles moderate motion and registration imperfection gracefully,
and has been validated on 167 TractoInferno subjects with 98.8% success rate.

Bidirectional is superior for single-subject bilateral symmetry analysis but is
approximately 3× slower (four tracking passes) and assumes symmetric anatomy — an
assumption that is invalid in stroke, tumour, or resection cases.

---

## Why the tract profile is reoriented inferior→superior

An along-tract profile averages a scalar across the bundle by point index, so point index
*i* must mean the same anatomical position in every streamline. Tractography does not
provide that: each streamline is stored `[backward_from_seed][forward_from_seed]`, so a
bundle mixes orientations. On in-vivo CST data about 20% of streamlines run
superior→inferior, which is enough to visibly bias the result — averaging index 0 pools
pontine samples from most streamlines with precentral samples from the rest, compressing
the two ends of the profile toward each other. `compute_tract_profile` therefore reorients
each streamline inferior→superior before sampling.

**Why an anatomical rule rather than a bundle centroid.** DIPY offers `orient_by_streamline`,
which orients a bundle relative to a reference such as its centroid. That makes streamlines
agree with *each other*, which is all that profile *invariance* requires — but it does not
fix which end comes first. `compute_localized_metrics` bins the profile by index and the
profile plots label position 0 "Pontine Level", so index 0 must be the inferior end
specifically. A centroid-relative rule would leave the whole bundle liable to being flipped
as a unit, silently swapping the pontine and precentral labels. The rule has to be tied to
anatomy.

**Why the Z axis, and not `orient_by_rois`.** DIPY's `orient_by_rois` is the textbook
choice, but it is unavailable where the work happens: `csttool metrics` receives only
tractograms and scalar maps, while the brainstem and motor ROI masks are built in a
different pipeline stage (`extract/modules/create_roi_masks.py`). Using them would mean
adding required inputs to the command. The Z axis is a sound substitute for a
supero-inferior tract: streamlines are in RASMM world coordinates, where +Z is superior by
the NIfTI standard, regardless of how the image's voxel axes are stored.

A useful property of comparing the two ends of a streamline is that it is
**translation-invariant**. The related concern that the subject is reoriented to RAS but
never recentered — so the world midline sits at X = M ≠ 0 — breaks anything that splits on
an absolute plane, but it cannot affect a within-streamline difference.

**Why quartile means rather than the two endpoints.** Orientation is decided from the mean Z
of the first and last quartiles. On healthy CST data the two rules agree on every streamline,
so this buys nothing today. It is insurance for the tortuous bundles the tool targets:
terminal points are the noisiest part of a streamline — that is where the stopping criterion
fired — and the CST's cortical end hooks laterally into the precentral gyrus. The quartile
mean averages that hook away for one extra mean per streamline and degrades to the endpoint
rule for short streamlines.

**Why non-vertical bundles warn rather than fail.** The Z rule is only meaningful for a
supero-inferior tract. `compute_tract_profile` is public API and a caller may legitimately
profile some other bundle, so a bundle whose dominant extent is not Z emits a warning rather
than an error. The dominant axis is chosen with `argmax`, which avoids inventing a threshold
constant.

**Why there is no RAS check.** The obvious guard — comparing `nib.aff2axcodes(affine)` to
`('R','A','S')` — would be wrong. That inspects the *voxel* axis order, which is irrelevant
here: world coordinates are RAS+ by definition, so the rule holds for validly-stored LAS or
LPS data too, and the check would reject them spuriously. The real precondition is that the
points are world coordinates rather than voxel indices, and that is already enforced
structurally, since the profiler applies the inverse affine to whatever it is given.

**Why the region extents live in one constant.** `TRACT_REGIONS` in
`metrics/modules/unilateral_analysis.py` defines each region's extent as a fraction of tract
length. `compute_localized_metrics` bins the profile with it and the profile figures place
their labels from it. This is the single source of truth on purpose: the bins and the labels
were previously stated independently and drifted, so the figures annotated 0/50/100% while the
table averaged 0-35/35-70/70-100%. Region boundaries are conventional, not validated against an
atlas — the JHU ICBM-DTI-81 landmark validation is outstanding, so the names are nominal.
