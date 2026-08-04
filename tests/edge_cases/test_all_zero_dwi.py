"""AU31 edge case: all-zero DWI data.

An all-zero DWI legitimately yields FA=0 everywhere, an empty white-matter
mask, zero seeds and zero streamlines — a silent empty success. AU31 adds a
warning at fit_tensors when the white-matter mask is empty so the empty
output is not a mystery. (All-zero is a valid, if useless, input: warn, do
not fail.)
"""

import numpy as np
import pytest
from dipy.core.gradients import gradient_table

from csttool.tracking.modules.fit_tensors import fit_tensors
from csttool.tracking.modules.seed_and_stop import seed_and_stop
from csttool.tracking.modules.run_tractography import run_tractography
from csttool.tracking.modules.estimate_directions import estimate_directions


def _valid_gtab():
    bvals = np.array([0, 1000, 1000, 1000, 1000, 1000, 1000])
    bvecs = np.array([
        [0, 0, 0], [1, 0, 0], [-1, 0, 0],
        [0, 1, 0], [0, -1, 0], [0, 0, 1], [0, 0, -1],
    ], dtype=float)
    return gradient_table(bvals, bvecs=bvecs)


def test_all_zero_dwi_yields_empty_white_matter_with_warning(all_zero_dwi_data):
    """All-zero DWI -> FA all zero -> empty WM mask -> UserWarning."""
    gtab = _valid_gtab()
    mask = np.zeros(all_zero_dwi_data.shape[:3], dtype=bool)
    mask[2:6, 2:6, 2:6] = True

    with pytest.warns(UserWarning, match="No white-matter voxels"):
        _tenfit, fa, md, rd, ad, white_matter = fit_tensors(
            all_zero_dwi_data, gtab, mask, verbose=False
        )
    assert float(fa.max()) == 0.0
    assert int(white_matter.sum()) == 0


def test_all_zero_dwi_pipeline_produces_zero_streamlines(all_zero_dwi_data):
    """The full fit -> seeds -> track path on all-zero data yields zero
    streamlines without crashing (silent empty success, now warned upstream)."""
    gtab = _valid_gtab()
    mask = np.zeros(all_zero_dwi_data.shape[:3], dtype=bool)
    mask[2:6, 2:6, 2:6] = True
    affine = np.eye(4)

    with pytest.warns(UserWarning, match="No white-matter voxels"):
        _tf, fa, _md, _rd, _ad, wm = fit_tensors(
            all_zero_dwi_data, gtab, mask, verbose=False
        )
    seeds, stop = seed_and_stop(fa, affine, white_matter=wm, brain_mask=mask,
                                fa_thresh=0.2, verbose=False)
    assert len(seeds) == 0
    # estimate_directions on empty WM mask is fine; run_tractography on empty
    # seeds yields zero streamlines.
    csapeaks = estimate_directions(all_zero_dwi_data, gtab, wm,
                                   sh_order=2, verbose=False)
    streamlines = run_tractography(csapeaks, stop, seeds, affine,
                                   step_size=0.5, random_seed=42, verbose=False)
    assert len(streamlines) == 0
