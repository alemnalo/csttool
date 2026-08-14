"""`--b0-threshold` plumbing (M2).

The flag was exposed on `preprocess`, `track` and `run` but consumed by none of
them: every gradient-table construction site read `DEFAULT_B0_THRESHOLD`
directly, so a non-default value was parsed and dropped. These tests pin one
authoritative threshold per execution, from the CLI down to `gtab.b0s_mask` and
the volumes masking actually uses.
"""

import argparse
from pathlib import Path
from unittest.mock import patch

import numpy as np
import nibabel as nib
import pytest

from csttool.defaults import DEFAULT_B0_THRESHOLD


def _write_dataset(directory: Path, stem: str, bvals: np.ndarray) -> Path:
    """Write a minimal DWI + FSL sidecars whose bvals are given."""
    directory.mkdir(parents=True, exist_ok=True)
    n = len(bvals)
    rng = np.random.default_rng(0)
    data = rng.random((6, 6, 6, n)).astype(np.float32) * 100

    bvecs = np.zeros((n, 3))
    dwi = np.asarray(bvals) > 0
    directions = np.array(
        [[1, 0, 0], [0, 1, 0], [0, 0, 1], [1, 1, 0], [1, 0, 1], [0, 1, 1]], float
    )
    directions /= np.linalg.norm(directions, axis=1, keepdims=True)
    bvecs[dwi] = [directions[i % len(directions)] for i in range(int(dwi.sum()))]

    nib.save(nib.Nifti1Image(data, np.eye(4)), directory / f"{stem}.nii.gz")
    np.savetxt(directory / f"{stem}.bval", np.asarray(bvals)[None, :], fmt="%g")
    np.savetxt(directory / f"{stem}.bvec", bvecs.T, fmt="%.8f")
    return directory / f"{stem}.nii.gz"


# ---------------------------------------------------------------------------
# T2.1 / T2.2 — the threshold given is the threshold used
# ---------------------------------------------------------------------------

@pytest.mark.parametrize(
    "threshold, expected_b0_indices",
    [(20, [0]), (DEFAULT_B0_THRESHOLD, [0, 1])],
)
def test_load_dataset_honours_b0_threshold(tmp_path, threshold, expected_b0_indices):
    from csttool.preprocess.modules.load_dataset import load_dataset

    bvals = np.array([0, 30, 1000, 1000, 1000])
    _write_dataset(tmp_path, "sub", bvals)

    _, gtab, _, _ = load_dataset(str(tmp_path), "sub", b0_threshold=threshold)

    assert np.where(gtab.b0s_mask)[0].tolist() == expected_b0_indices


@pytest.mark.parametrize(
    "threshold, expected_b0_indices",
    [(20, [0]), (DEFAULT_B0_THRESHOLD, [0, 1])],
)
def test_get_gtab_for_preproc_honours_b0_threshold(
    tmp_path, threshold, expected_b0_indices
):
    from csttool.cli.utils import get_gtab_for_preproc

    bvals = np.array([0, 30, 1000, 1000, 1000])
    nii = _write_dataset(tmp_path, "sub", bvals)

    gtab = get_gtab_for_preproc(nii, b0_threshold=threshold)

    assert np.where(gtab.b0s_mask)[0].tolist() == expected_b0_indices


def test_threshold_above_every_bval_fails_loudly(tmp_path):
    """A threshold that leaves no DWI is a user error, not a silent no-op."""
    from csttool.preprocess.modules.load_dataset import load_dataset
    from csttool.preprocess.modules.gradient_validation import (
        GradientTableValidationError,
    )

    _write_dataset(tmp_path, "sub", np.array([0, 30, 1000, 1000, 1000]))

    # Below every b-value: no b0 at all -> the existing hard failure.
    with pytest.raises(GradientTableValidationError, match="No b=0 volumes"):
        load_dataset(str(tmp_path), "sub", b0_threshold=-1)


# ---------------------------------------------------------------------------
# T2.3 — the < / <= boundary is now decided in exactly one place
# ---------------------------------------------------------------------------

def test_masking_uses_gtab_b0s_mask_at_the_exact_boundary():
    """A volume at exactly b == threshold is a b0 to the gradient table.

    Masking used to re-threshold with a strict `<` against the module-level
    constant while the validator built `b0s_mask` with `<=`. At the *default*
    threshold the two happened to agree, because `gradient_table` rewrites
    every sub-threshold b-value to exactly 0 — so this case alone was never
    reachable. Masking now reads `gtab.b0s_mask`, which makes the agreement
    structural rather than coincidental; the reachable half of the defect is
    covered by the next test.
    """
    from dipy.core.gradients import gradient_table
    from csttool.preprocess.modules import background_segmentation as bs_mod

    bvals = np.array([0, 50, 1000, 1000])
    bvecs = np.array([[0, 0, 0], [0, 0, 0], [1, 0, 0], [0, 1, 0]], float)
    gtab = gradient_table(bvals, bvecs=bvecs, b0_threshold=50)
    assert np.where(gtab.b0s_mask)[0].tolist() == [0, 1]

    data = np.random.default_rng(0).random((6, 6, 6, 4))

    with patch.object(bs_mod, "median_otsu") as mock_otsu:
        mock_otsu.return_value = (data, np.ones((6, 6, 6), bool))
        bs_mod.background_segmentation(data, gtab)

    vol_idx = mock_otsu.call_args.kwargs["vol_idx"]
    assert list(vol_idx) == [0, 1]


def test_masking_follows_a_non_default_threshold():
    """A gtab built with a tighter threshold narrows the masking volumes too."""
    from dipy.core.gradients import gradient_table
    from csttool.preprocess.modules import background_segmentation as bs_mod

    bvals = np.array([0, 30, 1000, 1000])
    # b=30 is a DWI under a threshold of 20, so it carries a real direction.
    bvecs = np.array([[0, 0, 0], [0, 0, 1], [1, 0, 0], [0, 1, 0]], float)
    gtab = gradient_table(bvals, bvecs=bvecs, b0_threshold=20)

    data = np.random.default_rng(0).random((6, 6, 6, 4))
    with patch.object(bs_mod, "median_otsu") as mock_otsu:
        mock_otsu.return_value = (data, np.ones((6, 6, 6), bool))
        bs_mod.background_segmentation(data, gtab)

    assert list(mock_otsu.call_args.kwargs["vol_idx"]) == [0]


# ---------------------------------------------------------------------------
# T2.4 — CLI dispatch on each of the three commands
# ---------------------------------------------------------------------------

def test_cmd_preprocess_forwards_threshold(tmp_path):
    from csttool.cli.commands.preprocess import cmd_preprocess

    nii = _write_dataset(tmp_path, "sub", np.array([0, 30, 1000, 1000]))
    args = argparse.Namespace(
        nifti=nii, dicom=None, out=tmp_path / "out", coil_count=4,
        target_voxel_size=None, b0_threshold=17.5,
    )

    with patch("csttool.preprocess.run_preprocessing") as mock_run:
        mock_run.return_value = None
        cmd_preprocess(args)

    assert mock_run.call_args.kwargs["b0_threshold"] == 17.5


def test_cmd_track_forwards_threshold(tmp_path):
    from csttool.cli.commands import track as track_mod

    nii = _write_dataset(tmp_path, "sub", np.array([0, 30, 1000, 1000]))
    args = argparse.Namespace(
        nifti=nii, subject_id="sub", out=tmp_path / "out", b0_threshold=17.5,
    )

    with patch.object(track_mod, "get_gtab_for_preproc") as mock_gtab:
        mock_gtab.side_effect = RuntimeError("stop after gtab construction")
        with pytest.raises(RuntimeError, match="stop after"):
            track_mod.cmd_track(args)

    assert mock_gtab.call_args.kwargs["b0_threshold"] == 17.5


def test_defaults_resolve_to_fifty_when_flag_absent(tmp_path):
    """T2.6: absent flag must behave exactly as before this milestone."""
    from csttool.cli.commands.preprocess import cmd_preprocess

    nii = _write_dataset(tmp_path, "sub", np.array([0, 30, 1000, 1000]))
    args = argparse.Namespace(
        nifti=nii, dicom=None, out=tmp_path / "out", coil_count=4,
        target_voxel_size=None,
    )

    with patch("csttool.preprocess.run_preprocessing") as mock_run:
        mock_run.return_value = None
        cmd_preprocess(args)

    assert mock_run.call_args.kwargs["b0_threshold"] == DEFAULT_B0_THRESHOLD


# ---------------------------------------------------------------------------
# T2.5 — `run` uses one threshold for both sub-commands
# ---------------------------------------------------------------------------

def test_run_forwards_one_threshold_to_both_subcommands(tmp_path):
    from csttool.cli.commands import run as run_mod

    nii = _write_dataset(tmp_path, "sub", np.array([0, 30, 1000, 1000]))
    out = tmp_path / "out"
    out.mkdir()
    args = argparse.Namespace(
        nifti=nii, dicom=None, out=out, subject_id="sub",
        preprocess=True, b0_threshold=17.5, continue_on_error=False,
    )

    with patch.object(run_mod, "cmd_preprocess") as mock_pre, \
            patch.object(run_mod, "cmd_track") as mock_track:
        mock_pre.return_value = {"preprocessed_path": nii, "stem": "sub"}
        mock_track.return_value = None
        run_mod.cmd_run(args)

    assert mock_pre.call_args[0][0].b0_threshold == 17.5
    assert mock_track.call_args[0][0].b0_threshold == 17.5


def test_no_hardcoded_threshold_remains_in_the_preprocess_path():
    """Every construction site must read the execution's threshold, not the constant."""
    import inspect
    from csttool.preprocess.modules import background_segmentation as bs_mod

    src = inspect.getsource(bs_mod)
    assert "DEFAULT_B0_THRESHOLD" not in src
