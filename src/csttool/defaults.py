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
