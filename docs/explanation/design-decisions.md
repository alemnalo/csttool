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
an absolute plane, but it cannot affect a within-streamline difference. The absolute-plane
splits elsewhere in the pipeline now use the warped MNI midline (see
[Why the hemisphere midline is the warped MNI midline, not world X=0](#why-the-hemisphere-midline-is-the-warped-mni-midline-not-world-x0)).

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

---

## Why MPPCA is the default denoiser

All three denoisers are offered, but the default has to work without asking the user for
information they usually do not have.

- **NLMeans** needs a receiver-coil count for PIESNO sigma estimation, and an assumption about
  whether the noise is Gaussian or Rician. On a modern multi-channel head coil with parallel
  imaging, the *effective* coil count is not the physical one and is rarely known; the tool was
  guessing `N=4` and assuming Gaussian noise on everyone's data.
- **Patch2Self** needs bvals and enough directions, so it cannot be a universal default.
- **MPPCA** estimates the noise level itself, from the eigenvalue distribution of local PCA
  patches against the Marchenko-Pastur distribution. It needs neither a coil count nor a noise
  model nor bvals — it exploits the redundancy of the 4D DWI directly.

MPPCA also has a reproducibility advantage that fell out of this: it exposes no thread-count
and no seed parameter, so the multithreaded-reduction non-determinism that made NLMeans
non-reproducible cannot arise on the default path. Its output is bitwise identical across
repeated runs.

Choosing MPPCA sidesteps rather than settles the Gaussian-vs-Rician question: with no noise
model to assume, there is nothing to get wrong. NLMeans remains available via
`--denoise-method nlmeans`, and its coil-count and noise-distribution caveats apply to anyone
who chooses it.

**Why the default lives in one constant.** `DEFAULT_DENOISE_METHOD` in `csttool/defaults.py` is
the single source of truth, read by the CLI parsers, the command wrappers, `run_preprocessing`
and `denoise`. The default was previously restated in eight places; a change applied to only one
of them left every CLI command on NLMeans while the signature, the CHANGELOG and the project
notes all said otherwise. Defaults that are stated more than once eventually disagree.

It is a standalone leaf module rather than part of `csttool.preprocess` because `denoise.py`
needs the value and is itself imported by `csttool/preprocess/__init__.py` — a constant defined
there could not be read back without a circular import. Having no imports of its own, it is
safe to read from anywhere regardless of import order.

---

## Why the headline mean is per-streamline, not point-weighted

Two summaries of a scalar along a bundle coexist, and they measure different things.
`sample_scalar_along_tract` pools every in-bounds point of every streamline into one flat
array, so a streamline with twice as many points casts twice as many votes — the result is
**point-weighted** (length-biased). `compute_tract_profile` normalises each streamline to a
fixed arc length first, so each streamline casts one vote. The report's global table and the
laterality indices used the point-weighted mean while the *displayed* profile used the
per-streamline one, so the headline and the picture could disagree, and the LI could shift
with bundle geometry (audit finding AU10).

The headline is now the **per-streamline mean**: each streamline contributes one value — the
mean of its sampled points — regardless of how many points it has. This is length-unbiased,
and it is the same population the displayed profile normalises over.

**Why `std` moved with it.** A mean does not travel alone. Reporting `0.498 ± 0.238` where
the mean is per-streamline but the SD is the scatter of every voxel would be incoherent: the
centre and the spread would describe different populations. So the whole headline block
(`mean`/`std`/`median`/`min`/`max`/`n_streamlines`) describes the per-streamline population.
`std` is the between-streamline SD of streamline means — the variability *of streamlines*,
not of the tissue — and is therefore much smaller than the old voxel scatter. Every report's
"±" changes meaning; this is deliberate, because the old "±" paired a per-streamline-style
mean (it was not) with a point-pool spread. `min`/`max` bound the same per-streamline
distribution.

**Why the point-pool summary is kept, not removed.** It is a different and legitimate
quantity (the scatter of all sampled voxels is a tissue-heterogeneity measure), and removing
it would hide the bias rather than name it. It lives under `*_point_weighted` keys and
`n_samples`, is exposed in the CSVs as `*_mean_point_weighted`, and is never used as the
headline. This is the same instinct as AU5/AU24 — label constructed numbers rather than
delete them.

**Why per-streamline and not the profile-derived mean.** The mean of the profile array is
also one-vote-per-streamline in spirit, but it is a *different population*: `compute_tract_profile`
resamples each streamline to a fixed arc length and drops streamlines with fewer than five
valid points. Deriving the headline from it would couple the headline to the profile's
resampling choices and to its short-streamline cutoff. The per-streamline mean is the pure
"one vote per streamline" definition, independent of the profile, and is what the LIs and the
global table should rest on. On in-vivo data the two differ by ~0.01 FA (0.4984 vs 0.4853 on
the healthy control) for exactly this reason.

**Why the bias is small.** Streamline length only spans ~146–322 points (mean ~227) and the
correlation between length and a streamline's mean FA is weak (−0.20 to +0.29, and it flips
sign between hemispheres and subjects). Longer streamlines do pull the point-weighted mean
in the direction their own mean FA points — confirmed, the point-weighted mean sits below the
per-streamline mean on both sides of the healthy control — but the effect is ~0.2–0.4% on the
mean and ~6% relative on the LI. All three definitions stay inside the 0.05 "symmetric" band
on both subjects examined, so the thesis's "FA symmetric, count asymmetric" conclusion
survives. AU10 is a correctness/coherence fix, not a results-changing one.

---

## Why the hemisphere midline is the warped MNI midline, not world X=0

Several diagnostics split the brain into left and right hemispheres: the streamline
midline-crossing (commissural) exclusion in `extract_cst_passthrough`, the cerebral-peduncle
FA sample (`sample_peduncle_fa`), the SyN-Jacobian hemisphere statistics, and the atlas-warp
centroid QC. All of them used to split on world X = 0. That is wrong whenever the subject is
not centred at the world origin — and `register_mni_to_subject` reorients the subject to RAS
but never recentres it, so the anatomical midline sits at world X = M, not 0 in general.

**Why this is a *diagnostic* problem, not a bundle problem.** Final left/right bundle
assignment is ROI-based: a streamline belongs to the left CST if it traverses the left motor
cortex, regardless of where the midline is. So the bundles were never misassigned. The four
sites above are diagnostics and filters, and they *were* biased — most consequentially
`sample_peduncle_fa`, whose "lower left peduncle FA" output underpinned [[07 Decisions]] D3.

**Why the warped MNI midline, not a centroid proxy.** The MNI152 template is symmetric about
world X = 0, so the MNI X = 0 plane *is* the anatomical midline. Warping it into subject
space gives the true subject midline. The cheaper proxies — the FA>0.15 brain centroid, or
the brainstem centroid — are not midline estimates: the brain is not perfectly symmetric, and
the brainstem is anatomically right-shifted in at least one subject (centroid +4.9 mm, against
a local warped midline of +2.9 mm there), so it overstates the offset.

On the two subjects examined, the warped midline is genuinely off-centre: its median world X is
−3.58 mm (healthy control) and +0.84 mm (ALS), and the healthy control's midline sits ~4 mm left
of X = 0 at every level (brainstem −4.75 mm, mid −3.91 mm, cortex −3.68 mm). The subject is
reoriented to RAS but never recentered, so X = 0 is simply not the midline.

**And the midline is not a plane at all.** It curves with the non-linear warp: the ALS subject's
runs from +2.94 mm at the brainstem to −2.91 mm at the cortex — a ~6 mm range. No single X value
is correct at both ends, which is the reason the per-voxel `hemisphere_mask` is the primary
output rather than a scalar.

Both effects land on a structure with no room for them: the cerebral peduncles are only
~10–15 mm wide, so a cut several mm off the true midline mislabels a large fraction of their
voxels. Judge a midline error against the structure being split, not against the head.

**Why one source of truth, in three views.** `compute_warped_midline` warps a whole-brain
MNI left-hemisphere mask (`X_mni < 0`) into subject space and returns three representations of
the same surface, because the consumers need different things:

- a per-voxel `hemisphere_mask` (the anatomical L/R label) for the voxel-based sites —
  `sample_peduncle_fa` and the Jacobian split. This is the gold standard: each voxel is
  assigned to the hemisphere it maps to in MNI, with no scalar approximation.
- a signed `midline_distance` volume (mm) for the one site that operates on continuous
  streamline *points* — the midline-crossing exclusion. A single scalar plane is a poor fit
  here: the warped midline curves with SyN, and a whole-brain median is biased by asymmetric
  sub-cortical structures (it sat at −3.6 mm for one subject while the cortical midline sat
  at −0.6 mm — using the scalar would have *regressed* that subject).
- a scalar `midline_x` (median world X of the central-slab midline surface) for the atlas-warp
  centroid QC, which only needs the *sign* of each motor centroid relative to the midline and
  is insensitive to a few mm.

**Why the Jacobian gradient is per mm (AU19).** The SyN forward field returned by
`DiffeomorphicMap.get_forward_field()` is in world (mm) units on dipy 1.9–1.12.1 — verified
empirically from `_warp_forward`, where the displacement is added to a world coordinate before
the world-to-grid transform is applied. The Jacobian is `J = I + d(displacement_mm)/d(world_mm)`,
so `np.gradient` must take the voxel size as spacing. The original code took it per voxel
index, scaling the spread by the voxel size (≈2× on 2 mm isotropic data) — which is why the
reported `L 1.000±0.388 / R 0.999±0.330` was miscalibrated. The hemisphere split also dropped
the off-diagonal affine terms; the full affine is now used (and the per-voxel mask is
preferred when available).

**What moved and what did not.** Bundle assignment and final CST counts are unchanged
(ROI-based). On the two real subjects the peduncle FA asymmetry largely vanishes under the
correct midline (the X = 0 split manufactured an ~8% "lower-left" asymmetry on the healthy
control that the warped-MNI mask makes symmetric), and the Jacobian std roughly halves while
the L/R symmetry it was quoted as evidence for survives. See the CHANGELOG `[Unreleased]`
entry for the measured table, and [[07 Decisions]] D3 for the consequence to the
"lower-left peduncle FA is a data property" claim.
