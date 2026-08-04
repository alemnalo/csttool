"""AU31 edge case: single-direction DWI (1 b0 + 1 DWI).

Too few directions for any meaningful SH fit. estimate_directions already
force-reduces the SH order via validate_sh_order (which warns); AU31 pins
that behaviour so a regression that silently keeps a too-high SH order is
caught. The pipeline should run without crashing and produce a (degenerate)
direction field.
"""

import warnings

import numpy as np
import pytest
from dipy.core.gradients import gradient_table

from csttool.tracking.modules.estimate_directions import estimate_directions


def test_single_direction_reduces_sh_order_with_warning(
    single_direction_dwi_data, single_direction_bvals, single_direction_bvecs
):
    """With 1 DWI direction, SH order must reduce (to 2) and a warning emitted."""
    gtab = gradient_table(single_direction_bvals, bvecs=single_direction_bvecs)
    wm = np.zeros(single_direction_dwi_data.shape[:3], dtype=bool)
    wm[2:6, 2:6, 2:6] = True

    # validate_sh_order warns when the requested order exceeds what the
    # direction count supports.
    with warnings.catch_warnings(record=True) as w:
        warnings.simplefilter("always")
        csapeaks = estimate_directions(
            single_direction_dwi_data, gtab, wm, sh_order=6, verbose=False
        )
    sh_warnings = [x for x in w if issubclass(x.category, UserWarning)
                   and "Reducing to SH order" in str(x.message)]
    assert sh_warnings, "Expected an SH-order-reduction warning for 1 direction"
    # The direction field is still produced (degenerate but well-shaped).
    assert csapeaks.peak_dirs.shape[0] == single_direction_dwi_data.shape[0]


def test_single_direction_does_not_crash_pipeline(
    single_direction_dwi_data, single_direction_bvals, single_direction_bvecs
):
    """The whole fit -> directions path must not crash on a 2-volume DWI."""
    from csttool.tracking.modules.fit_tensors import fit_tensors

    gtab = gradient_table(single_direction_bvals, bvecs=single_direction_bvecs)
    mask = np.zeros(single_direction_dwi_data.shape[:3], dtype=bool)
    mask[2:6, 2:6, 2:6] = True

    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        _tf, fa, _md, _rd, _ad, wm = fit_tensors(
            single_direction_dwi_data, gtab, mask, verbose=False
        )
        csapeaks = estimate_directions(
            single_direction_dwi_data, gtab, wm, sh_order=2, verbose=False
        )
    assert fa.shape == single_direction_dwi_data.shape[:3]
    assert csapeaks.peak_dirs.shape[:3] == single_direction_dwi_data.shape[:3]
