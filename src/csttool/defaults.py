"""
Pipeline defaults shared between the CLI and the library.

Deliberately dependency-free, and deliberately not inside `csttool.preprocess`: `denoise.py`
needs these values and is itself imported by `csttool/preprocess/__init__.py`, so a constant
living there could not be read from the modules that need it without a circular import. A leaf
module with no imports of its own can be read from anywhere — CLI parser construction, command
wrappers, or library code — regardless of import order.
"""

# Denoising method used when the caller does not choose one.
#
# MPPCA estimates its own noise level from the eigenvalue distribution of local PCA patches.
# Unlike nlmeans it therefore needs neither a receiver-coil count nor an assumption about the
# noise distribution being Gaussian or Rician, and unlike patch2self it does not need bvals.
# It also exposes no thread-count or seed parameter, so the multithreading non-determinism
# that affected nlmeans cannot arise. See docs/explanation/design-decisions.md.
#
# This is the single source of truth. The default was previously restated in eight places and
# a change applied to only one of them left every CLI command silently on nlmeans, so read it
# from here rather than repeating the literal.
DEFAULT_DENOISE_METHOD = "mppca"

# Tensor fit method for DTI.
#
# DIPY's TensorModel default is version-dependent (OLS in older versions, WLS in
# recent ones). Pinning this removes a silent reproducibility hazard and ensures
# the parameter sweep can include it. Options: 'OLS' (ordinary least squares,
# log-linear), 'WLS' (weighted least squares), 'NLLS' (non-linear least squares,
# reference standard), 'RT' (robust tensor fitting / RESTORE).
#
# WLS is the DIPY 1.9+ default and preserves backward compatibility with all
# previously reported csttool FA/MD values.
DEFAULT_FIT_METHOD = "WLS"

# Number of ODF peaks extracted per voxel for roi-seeded and bidirectional tracking.
#
# The whole-brain estimate_directions path pins npeaks=1 explicitly. The
# roi-seeded and bidirectional paths were previously leaving this unset, so they
# inherited DIPY's version-dependent default. Pinning npeaks=1 makes the behaviour
# consistent and reproducible across DIPY versions.
#
# >1 enables multi-peak tracking for crossing-fibre regions (corona radiata where
# CST crosses SLF/CC) — this is the planned research direction, not the current
# production setting.
DEFAULT_NPEAKS = 1

# --- B-value threshold for partitioning b=0 vs DWI volumes ---
#
# Hardcoded as <50 in seven places before AU20. A dataset with b=30 shells or a
# non-zero lowest b-value would be silently mis-partitioned. 50 s/mm² is the
# conventional DWI threshold; exposed on --b0-threshold for edge cases.
DEFAULT_B0_THRESHOLD = 50

# --- Whole-brain tracking ---
DEFAULT_FA_THRESHOLD = 0.2          # FA threshold for stopping and seeding
DEFAULT_SEED_DENSITY = 1            # seeds per voxel (whole-brain)
DEFAULT_STEP_SIZE = 0.5             # mm
DEFAULT_SH_ORDER = 6                # CSA ODF spherical harmonic order
DEFAULT_RNG_SEED = 42               # reproducibility seed
DEFAULT_SPHERE_NAME = "symmetric362"  # ODF sphere

# --- Peak extraction (whole-brain, estimate_directions) ---
DEFAULT_RELATIVE_PEAK_THRESHOLD = 0.8
DEFAULT_MIN_SEPARATION_ANGLE = 45   # degrees

# --- ROI-seeded / bidirectional extraction ---
DEFAULT_SEED_FA_THRESHOLD = 0.15    # FA threshold for roi-seeded/bidirectional
DEFAULT_SEED_DENSITY_EXTRACT = 2    # seeds per voxel for extraction methods
DEFAULT_MIN_LENGTH = 30.0           # mm (roi-seeded/bidirectional)
DEFAULT_MAX_LENGTH = 200.0          # mm (roi-seeded/bidirectional)

# --- Passthrough / endpoint extraction ---
DEFAULT_MIN_LENGTH_PASSTHROUGH = 20.0
DEFAULT_MAX_LENGTH_PASSTHROUGH = 200.0

# --- ROI dilation ---
DEFAULT_DILATE_BRAINSTEM = 2        # dilation iterations for brainstem ROI
DEFAULT_DILATE_MOTOR = 1            # dilation iterations for motor cortex ROI

# --- Midline exclusion (passthrough filtering) ---
DEFAULT_MIDLINE_TOLERANCE_MM = 8.0  # allowed medial excursion before exclusion

# --- Laterality index interpretation bands ---
DEFAULT_LI_SYMMETRIC = 0.05
DEFAULT_LI_MILD = 0.10
DEFAULT_LI_MODERATE = 0.20

# --- Bidirectional artifact index threshold ---
DEFAULT_ARTIFACT_INDEX_THRESHOLD = 0.20
