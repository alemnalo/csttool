"""AU31 edge case: misordered / malformed bvec/bval files.

This class is already caught at load by AU21's gradient_validation
(validate_bvals_bvecs: count mismatch, non-unit DWI bvecs, negative bvals,
no b0). AU31 pins that coverage here so the edge-case suite is self-
contained and a regression that bypasses the validator is caught.
"""

import numpy as np
import pytest

from csttool.preprocess.modules.gradient_validation import (
    GradientTableValidationError,
    validate_bvals_bvecs,
)


def test_count_mismatch_rejected():
    """bval/bvec with different row counts (the classic misorder) must fail."""
    bvals = np.array([0, 1000, 1000, 1000])
    bvecs = np.array([[0, 0, 0], [1, 0, 0], [0, 1, 0]])  # 3 rows vs 4 bvals
    with pytest.raises(GradientTableValidationError, match="count mismatch"):
        validate_bvals_bvecs(bvals, bvecs)


def test_non_unit_dwi_bvec_rejected():
    """A DWI row whose bvec is not unit-norm (e.g. a truncated/miswritten file)."""
    bvals = np.array([0, 1000])
    bvecs = np.array([[0, 0, 0], [2, 0, 0]])
    with pytest.raises(GradientTableValidationError, match="unit-norm"):
        validate_bvals_bvecs(bvals, bvecs)


def test_negative_bval_rejected():
    bvals = np.array([0, -5])
    bvecs = np.array([[0, 0, 0], [1, 0, 0]])
    with pytest.raises(GradientTableValidationError, match="negative"):
        validate_bvals_bvecs(bvals, bvecs)


def test_no_b0_rejected():
    """A bval file with no b0 volume cannot fit a tensor."""
    bvals = np.array([1000, 1000])
    bvecs = np.array([[1, 0, 0], [0, 1, 0]])
    with pytest.raises(GradientTableValidationError, match="No b=0"):
        validate_bvals_bvecs(bvals, bvecs)


def test_valid_table_still_passes():
    """Regression guard: the well-formed case must not be broken by edge-case
    hardening."""
    bvals = np.array([0, 1000, 1000, 1000])
    bvecs = np.array([[0, 0, 0], [1, 0, 0], [0, 1, 0], [0, 0, 1]])
    bv, bc = validate_bvals_bvecs(bvals, bvecs)
    assert bv.shape == (4,)
    assert bc.shape == (4, 3)
