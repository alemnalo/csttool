"""
Determinism tests for the ROI-seeded / bidirectional extraction path.

The rest of the determinism suite exercises only the whole-brain
`run_tractography` path, which has always received `random_seed`. These tests
cover `track_from_seeds` — the tracker behind `--extraction-method roi-seeded`
and `bidirectional` — which did not accept a seed at all until it was plumbed
through (audit finding AU6).

Note on what these can and cannot prove: with a deterministic peaks direction
getter, DIPY's `LocalTracking` never consumes the RNG that `random_seed` seeds,
so `test_unseeded_is_also_reproducible` documents that the pre-fix code was
already reproducible in practice. The seed is plumbed so the guarantee holds by
construction rather than by luck — if a probabilistic direction getter or
`randomize_forward_direction` is ever introduced, these tests keep the promise
honest instead of silently starting to fail.
"""

import numpy as np
import pytest

from dipy.direction import peaks_from_model
from dipy.data import default_sphere
from dipy.reconst.shm import CsaOdfModel
from dipy.tracking import utils
from dipy.tracking.stopping_criterion import ThresholdStoppingCriterion

from csttool.reproducibility import RunContext
from csttool.extract.modules.roi_seeded_tracking import track_from_seeds


def _fingerprint(streamlines):
    """Order-sensitive hash over every coordinate of every streamline."""
    if len(streamlines) == 0:
        return "EMPTY"
    return hash(np.concatenate([np.asarray(s) for s in streamlines]).tobytes())


@pytest.fixture(scope="module")
def seeded_tracking_inputs(request):
    """Direction getter, stopping criterion and seeds for the ROI-seeded tracker.

    Built from DIPY's `small_64D` set, matching the rest of this suite.
    """
    from dipy.data import get_fnames
    from dipy.io.image import load_nifti
    from dipy.io.gradients import read_bvals_bvecs
    from dipy.core.gradients import gradient_table
    from dipy.reconst.dti import TensorModel

    fraw, fbval, fbvec = get_fnames('small_64D')
    data, affine = load_nifti(fraw)
    bvals, bvecs = read_bvals_bvecs(fbval, fbvec)
    gtab = gradient_table(bvals, bvecs=bvecs)

    brain_mask = (data[..., 0] > 0).astype(bool)
    fa = np.clip(TensorModel(gtab).fit(data, mask=brain_mask).fa, 0, 1)
    fa = np.nan_to_num(fa)

    wm_mask = fa > 0.2
    peaks = peaks_from_model(
        CsaOdfModel(gtab, sh_order_max=4), data, default_sphere,
        relative_peak_threshold=0.5, min_separation_angle=25, mask=wm_mask,
    )
    stopping_criterion = ThresholdStoppingCriterion(fa, 0.2)
    seeds = utils.seeds_from_mask(wm_mask, affine, density=1)

    return {
        "peaks": peaks,
        "stopping_criterion": stopping_criterion,
        "seeds": seeds,
        "affine": affine,
    }


def _track(inputs, random_seed):
    return track_from_seeds(
        inputs["seeds"],
        inputs["peaks"],
        inputs["stopping_criterion"],
        inputs["affine"],
        step_size=0.5,
        random_seed=random_seed,
    )


def test_track_from_seeds_accepts_random_seed(seeded_tracking_inputs):
    """The tracker behind bidirectional/roi-seeded must take a seed at all.

    Guards AU6 directly: the parameter was absent, so the CLI's --rng-seed
    could not reach the flagship extraction methods.
    """
    import inspect
    assert "random_seed" in inspect.signature(track_from_seeds).parameters


def test_seeded_runs_are_bitwise_identical(seeded_tracking_inputs):
    """Same seed, same input -> identical streamlines."""
    a = _track(seeded_tracking_inputs, 42)
    b = _track(seeded_tracking_inputs, 42)

    assert len(a) > 0, "no streamlines produced - comparison would be vacuous"
    assert len(a) == len(b)
    assert _fingerprint(a) == _fingerprint(b)


def test_unseeded_is_also_reproducible(seeded_tracking_inputs):
    """random_seed=None is reproducible too, with a deterministic getter.

    This documents *why* AU6 was a latent defect rather than an active source of
    non-determinism: nothing in this path consumes LocalTracking's RNG. If this
    test ever fails, the tracker has gained a stochastic component and every
    unseeded caller has become non-reproducible.
    """
    a = _track(seeded_tracking_inputs, None)
    b = _track(seeded_tracking_inputs, None)

    assert len(a) > 0, "no streamlines produced - comparison would be vacuous"
    assert _fingerprint(a) == _fingerprint(b)


def test_run_context_seed_reaches_tracker(seeded_tracking_inputs):
    """The RunContext seed plumbs through, per decision D1."""
    ctx = RunContext(run_seed=42)
    a = _track(seeded_tracking_inputs, ctx.rng_tracking_seed())
    b = _track(seeded_tracking_inputs, 42)

    assert len(a) > 0
    assert _fingerprint(a) == _fingerprint(b)
