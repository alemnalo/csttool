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

Full technical write-up: `docs/fixes/bidirectional_seeding.md` in the companion
`csttool-devlog` repository, where investigation write-ups live rather than in the published
docs.

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

---

## Why the report shows a genuine length median, and why a mean is never shown as a median

`compute_morphology` historically returned `mean_length`, `std_length`, `min_length`,
`max_length` — but no median. The report's global table paired each row with a `median
(min–max)` column, so the Length row showed only `(min–max)` while FA/MD/RD/AD showed a
genuine median. That was a data gap, not an intentional alternate statistic.

`compute_morphology` now also returns `median_length` (`float(np.median(lengths))`), so the
Length row uses the same `median (min–max)` format as the other rows. This is **additive
only**: the length laterality index is defined on `mean_length` (the per-streamline
headline, AU10), so `median_length` is a descriptive companion that changes no existing
metric value and no LI.

A mean is never displayed as a median. Legacy morphology dicts that predate `median_length`
(and any report re-rendered from them) render an explicit em dash in the Length median
column, not the mean dressed as a median. The Streamlines and Volume rows carry an em dash
by design — Streamlines is a count and Volume is a single bundle volume, neither of which has
a per-streamline distribution from which a median is meaningful in this report.

## Why the PDF report is one A4 page with a dynamic orientation label

The clinical report is a single A4 portrait page, enforced by an automated PDF page-count
test across a realistic envelope of inputs (full/empty metadata, FA-only scalars, long
subject IDs, zero streamlines, pathologically long provenance strings). The test asserts on
the rendered PDF, and a companion test asserts the layout still leaves several millimetres
of headroom, so a report that fits only by a hair fails before it reaches a second page.

The orientation label (`RAS`, `LAS`, …) is computed from the FA affine via
`nib.orientations.aff2axcodes`, never hardcoded. **`LAS` is a normal value here, not a bug.**
csttool does not force subject data into RAS: the primary `dcm2niix` import path preserves
the scanner's native voxel orientation, which for a typical Siemens axial DWI series is LAS.
Only the `dicom2nifti` fallback reorients to RAS, and it does so for gradient-consistency
reasons (image and bvecs reoriented *together*, see AU21), not because RAS is the pipeline's
declared output space. So the label genuinely varies by subject and by import path, which is
exactly why it is derived rather than written down. It is shown once in the methods band, not
repeated per QC slice.

Laterality colour is reserved for data (blue = Left, orange = Right); structural elements are
neutral, and no red is used for laterality.

---

## DEC colour encoding: world frame, not voxel frame

The direction-encoded colour (DEC) image colours each voxel by the principal
diffusion direction. csttool stores the principal eigenvector **V1** as a
scientific data product in the **anatomical world (RAS+) frame**, with the frame
declared in the sidecar (`VectorFrame: "world-RAS"`), and derives the DEC from
it.

### Why the world frame, not the voxel frame

DIPY's `TensorModel.fit` solves for the tensor in the frame of the b-vectors,
which csttool pins to the data's voxel axes. `tenfit.evecs[..., :, 0]` are
therefore **direction cosines with respect to the voxel axes** — unit-length
physical directions, not index-space displacements. For an axis-aligned
acquisition (RAS, LAS, LPS) the voxel axes coincide (up to sign) with the
anatomical axes, so the voxel-frame DEC is correct. But for any oblique
acquisition — or for LAS/LPS storage combined with a coordinate convention
that differs from the b-vec frame — colouring by the voxel-frame eigenvector is
scientifically wrong: it colours by the *array* axes rather than by the
*anatomical* axes the reader interprets.

### How the rotation is computed

The voxel→world affine `M = affine[:3,:3]` carries voxel scaling. Applying it
directly would rescale components by voxel size and turn anisotropic voxels into
a fake anisotropy of direction (a 45° in-plane/through-plane direction on
2×2×6 mm data would render at ≈72°). Normalising the columns removes scale but
leaves shear, so the result is not orthonormal. csttool therefore uses the
**orthonormal factor `R` of the polar decomposition** `M = R·S` (`scipy.linalg.polar`),
the closest orthogonal matrix to `M` in the Frobenius norm: the pure
rotation/reflection content with all scale and shear removed.

Reflections (`det(R) = −1` for LAS/LPS-stored data) are **not** "corrected" to
`det = +1`. DEC takes the componentwise absolute value, so an axis reflection
cannot change any colour; flipping a sign would silently mutate the stored V1
field for no visual gain. `det(R)` is recorded in the sidecar so a consumer can
tell whether the source was reflected; `|det(R)| ≈ 1` is asserted (a violation
means `M` was singular).

### The DEC formula

DEC colour = `|V1_world| × clip(FA, 0, 1)` — the formula `dipy.reconst.dti.color_fa`
executes, applied after the rotation. Red = L–R, green = A–P, blue = S–I, in
anatomical world axes. No gamma, no percentile stretch: brightness is FA, so a
dark panel is a real finding and must not be cosmetically brightened. A
canonical V1 product that is reproducible from the stored file + the FA map
using two public DIPY/NumPy operations is the whole point of making V1 — not the
DEC PNG — the source of truth.

See `csttool/spatial.py` for the tested implementation and `../csttool-devlog/report-improvement/
visualization-refactoring-plan.md` §5.1 for the full mathematical treatment.

---

## CST density: a normalized fraction, not a raw count

The CST density volume answers "is the extracted bundle spatially coherent and
left/right symmetric?" Its value is the **fraction of distinct retained
bilateral CST streamlines that visit each voxel at least once**:

* **Numerator:** `dipy.tracking.utils.density_map` per voxel — the number of
  *distinct* streamlines that visit it. Repeated points of one streamline in one
  voxel, and re-entries, are each counted **once** (verified DIPY semantics).
  Left and right are computed separately then summed (the bundles are disjoint,
  so summing is exact; a midline voxel visited from both sides accumulates both).
* **Denominator:** the total number of retained bilateral streamlines
  (`StreamlineCountLeft + StreamlineCountRight`), recorded in the sidecar so a
  reader can recover raw counts and so two subjects' maps can be compared knowing
  exactly what each was divided by.

This makes the density **independent of streamline count** (the dominant
confound: a denser bundle should not look "more CST" by construction) and puts
left/right on one shared `[0,1]` scale by construction. A raw-count volume would
make any left/right or between-subject comparison meaningless.

### Step-size gap guard

`density_map` counts *sampled points*, not geometric traversal: a tracking step
long enough to skip a voxel would silently under-count it. This is unreachable
at the default `step_size = 0.5` mm against ≥1.5 mm voxels, but step size is
user-configurable, so the wrapper densifies the streamlines with
`dipy.tracking.utils.subsegment` only when a step could skip a voxel
(`step_size > min_voxel_extent / 2`), and records `Densified` in the sidecar.

Normalization lives **only** in the data product; the figure sets `vmax` only.
A second display-time normalization is forbidden — it would let two scales drift
and the colorbar stop describing the data.

See `csttool/extract/modules/density.py` and the plan §2.4/§5.2 for the full
specification.

---

## Why the profile band is the IQR, and why the centre line stays the mean

Every profile in the report is a per-node mean over the streamlines that
contributed to it, drawn as a bare line. A line cannot distinguish a consensus
from an average over dissent: two hemispheres whose means separate while their
spreads overlap everywhere do not support the difference the lines imply. So each
line is now backed by the per-node **interquartile range across contributing
streamlines**.

The centre line stays the **mean**, deliberately. `compute_localized_metrics`
derives the twelve regional values — and therefore every regional laterality
index — from that exact array. Switching the centre to the median would silently
change every published regional metric, which is a different change wearing a
rendering change's clothes. Where the mean and the IQR diverge visibly, that
divergence is itself the finding (a skewed per-node distribution), not a defect
in the choice of centre.

The band is computed in the **same pass** that produces the profile: the metric
block now derives both from `qc_stats.profile_matrix` instead of recomputing the
profile with `compute_tract_profile`. That is a pure refactor, and it is enforced
as one — `matrix.mean(axis=0)` must reproduce `compute_tract_profile` *exactly*,
not approximately. Exactness matters because scalar maps are stored `float32`:
promoting to `float64` inside the shared resampler moved the profile by ~5e-8,
invisible in a plot and a changed published value everywhere else.

Only the IQR is drawn. An earlier prototype carried 5–95 bands as well; four
translucent bands per panel proved unreadable, so there are two, each with its
own quartile edges stroked in its own hue so the blue/orange overlap stays
traceable to a hemisphere.

## Why the bootstrap SE is conditional on the retained bundle

The report's laterality indices are differences of two means quoted without any
statement of uncertainty, which leaves a reader unable to tell either case from
noise. Every headline mean now carries a nonparametric bootstrap standard error
in the JSON and CSV (not in the report tables — see below).

What that SE means is narrow and must stay narrow. It is the variability of the
reported mean **conditional on the bundle that was retained**: how much the number
would move if a different subset of *those* streamlines had been sampled. It does
**not** include tracking, seeding, registration, preprocessing or acquisition
variability. A re-run of tractography does not produce a resample of this bundle;
it produces a different bundle. Read as pipeline reproducibility, the SE would be
badly over-confident, so the JSON carries a mandatory `metrics.uncertainty.scope`
sentence saying exactly this, once, beside the numbers.

Two quantities deliberately have no SE. A streamline count *is* its own sample
size, so bootstrapping it is meaningless. Tract volume is a set-union voxel count,
not a mean over a resamplable population; a bootstrap of it estimates the
variability of a coverage statistic, which is a different claim that would be read
as an SE of a mean.

The SEs are not displayed in the report tables. That is a scope decision, not an
oversight: twelve regional cells plus the global rows would need a redesigned
column and a footnote for a value the reader is not yet asking for, and the page
budget had no room for either. The data is serialised, so the display can be added
later without recomputation.

## Why node homology is reported without a threshold

The twelve regional laterality indices assume node *i* is the same anatomical
level on both sides. Node *i* is `i/(n-1)` of the way along whatever was
reconstructed, so two hemispheres of different length put node *i* at different
heights — on one validation subject, 6.4 mm apart with a −7.9 mm mean length
difference. The regional table made that assumption silently; it now states the
measured offset beside the values that depend on it.

World Z is the headline rather than arc length. Arc-length mismatch is close to a
restatement of the length difference, since the parameterisation is arc-length
relative by construction, whereas Z answers the anatomical question directly — is
node *i* at the same height on both sides? — and for the CST, node Z is
near-monotone along the tract.

No threshold ships. Two subjects cannot establish one, and an unvalidated
threshold on a clinical report acquires downstream consumers immediately. So
there is no PASS/FAIL, no colour, no icon, no suppression of any regional value,
and — deliberately — **no `homology_flag` key in the JSON schema** for a consumer
to start depending on. Adding one later is not a breaking change; removing one
would be. Establishing a threshold needs ≥20 subjects and is future work.

## Why b-vector rotation accompanies motion correction

Motion correction resamples every DWI volume onto the reference volume's pose. The
diffusion-encoding direction recorded for volume *k*, however, is the one that applied
*before* the head moved. Ship the resampled volumes with the original `.bvec` and the
image and its gradient table describe different anatomies — which biases FA and MD and
tilts the principal eigenvector, silently, because nothing about the files looks wrong.
This is Leemans & Jones (2009); the fix is to apply the inverse of each volume's
estimated rotation to its b-vector.

Three conventions have to line up, and each one is settled from source rather than
assumed:

**Which direction the transforms point.** DIPY's `motion_correction` returns one
`AffineMap` affine per volume, operating in **world** coordinates and mapping
static-world → moving-world — the pull transform used for resampling, which numerically
equals that volume's forward head motion. Verified empirically as well as from the
docstring: planting a +10° world rotation into a synthetic volume and running the
correction returns `polar(reg_affine) ≈ Rz(+10°)`, not its inverse.

**How a b-vector responds.** `dipy.core.gradients.reorient_bvecs` applies the *inverse*
polar rotation of each affine to the corresponding non-b0 b-vector. That is exactly the
Leemans & Jones rule for a forward-motion affine, and it is DIPY's own convention,
pinned by DIPY's own test. csttool calls the primitive rather than re-deriving the
rotation direction — a sign error here is invisible in the output and doubles the
error instead of removing it.

**Which frame the b-vectors live in.** This is the part the library does not do for
you. DIPY's affines are world-space; b-vectors follow the DIPY/FSL convention and are
expressed in the frame of the **voxel axes**. A world rotation must therefore be
conjugated into that frame:

    g' = normalise( V⁻¹ · R_world⁻¹ · V · g )

It matters. For LAS-stored data (`V = diag(-1, 1, 1)`, the common dcm2niix output),
`V⁻¹ Rz(-θ) V = Rz(+θ)` — applying the world rotation directly would rotate the
b-vector the wrong way. For RAS+ data `V = I` and the conjugation is a no-op, which is
exactly why a test suite built only on RAS fixtures would never catch the error.

### Why `V` is the orientation only, not the full affine

`V = polar(image_affine[:3, :3])[0]` — the orthonormal rotation, with voxel scaling
discarded.

A b-vector is a physical unit direction whose components happen to be given in the
frame of the voxel axes. Voxel size is a property of the sampling grid, not of the
direction: `(1, 0, 0)` means "along the +i voxel axis in physical space" whether the
voxel is 1 mm or 6 mm. Using the full 3×3 would treat the b-vector as a displacement in
index space, which is a different object.

The decisive consequence is testable. With an axis-aligned affine the ground truth is
unambiguous — the corrected b-vector is `R_world⁻¹ g` — and it cannot depend on the
zooms. `A⁻¹ R⁻¹ A` violates that for anisotropic voxels; `V⁻¹ R⁻¹ V` satisfies it
exactly. The test suite asserts the invariance directly, and separately asserts that
the rejected formula would give a different answer, so the choice cannot be silently
reverted. DIPY takes the same position one level down: `reorient_bvecs` strips scale
from the motion affines by the same polar decomposition.

## Why external correction is declared, not verified

csttool skips its own preprocessing by default, so most runs operate on data another
tool has already corrected. That was previously an implicit assumption written into a
metadata string — `"Skipped (External Preprocessing Used)"` — which asserted, as fact,
something nobody had stated and csttool cannot check.

csttool cannot check it. Whether a DWI has been through TOPUP and EDDY is not
recoverable from the image and its sidecars: distortion correction leaves no signature
a tool can reliably detect, and BIDS metadata records the acquisition, not the
processing that followed. Any inference would be a guess presented as provenance.

So the mechanism is a declaration. `--input-corrected` records what the user says was
done, always as `{"declared": …, "verified_by_csttool": false, "source":
"user-declaration"}`. There is no code path that can set `verified_by_csttool` true —
a field that is always false is more honest than one that looks settable. The
declaration changes no processing decision.

Three design points follow:

**`unknown` is the default, and is not the same as `none`.** "Nobody said" and "the
user declared that nothing was done" are different states with different consequences
for how a result should be read. Collapsing them is what made the old string
misleading.

**A conflicting declaration warns; it does not block.** Declaring `topup-eddy` together
with `--perform-motion-correction` is usually a mistake, so it prints a high-visibility
warning and records it in the provenance. It does not error. An unverified declaration
must not be able to veto an explicit flag — a mistyped or over-broad declaration would
otherwise block a legitimate run, and running motion correction on already-corrected
data is undesirable rather than invalid (a user may legitimately want to measure the
residual motion an external tool left behind).

**Chronology is preserved.** In the stage ledger, declared external work is *prepended*
to csttool's own stages, because it happened before csttool received the data. A
thesis-style run serialises as external TOPUP/EDDY → csttool denoise → mask, so no
report can imply csttool denoised before corrections that preceded it.
