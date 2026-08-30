"""``--brain-mask`` must replace automatic background segmentation, exactly.

csttool estimates its own brain mask with median Otsu. Every voxel that mask
drops gets no tensor fit, so FA there is exactly zero — on CDMD that truncated
substantial inferior brainstem tissue that the dataset's own DWI-space mask,
and FSL dtifit with an independent mask, both retain.

``--brain-mask PATH`` substitutes a supplied mask for the automatic one. These
tests pin the two halves of that contract:

  * with a mask, ``background_segmentation`` is *not* called at all (a mask
    that merely re-masked the automatic result would not fix anything), and
  * without one, the automatic path is byte-for-byte what it was.

Validation is deliberately loud: a mask on the wrong grid is a user error, not
something to silently resample away.
"""

import argparse
from unittest.mock import MagicMock, patch

import nibabel as nib
import numpy as np
import pytest

from csttool.tracking.modules.brain_mask import (
    apply_brain_mask,
    load_external_brain_mask,
    resolve_brain_mask,
)

SHAPE = (6, 7, 8)
AFFINE = np.diag([2.0, 2.037037, 2.037037, 1.0])


def _write_mask(path, data, affine=None):
    affine = AFFINE if affine is None else affine
    nib.save(nib.Nifti1Image(np.asarray(data), affine), str(path))
    return path


def _dwi(n_vols=5):
    rng = np.random.default_rng(0)
    return rng.random(SHAPE + (n_vols,)) + 1.0


@pytest.fixture
def valid_mask_file(tmp_path):
    mask = np.zeros(SHAPE, dtype=np.int16)
    mask[1:5, 1:6, 1:7] = 1
    return _write_mask(tmp_path / "mask.nii.gz", mask)


# ---------------------------------------------------------------------------
# The external mask is used, and the automatic path is not run
# ---------------------------------------------------------------------------

def test_external_mask_is_used(valid_mask_file):
    data = _dwi()
    expected = np.zeros(SHAPE, dtype=bool)
    expected[1:5, 1:6, 1:7] = True

    masked_data, brain_mask, info = resolve_brain_mask(
        data, gtab=None, affine=AFFINE, brain_mask_path=valid_mask_file
    )

    assert brain_mask.dtype == np.bool_
    np.testing.assert_array_equal(brain_mask, expected)
    assert info['brain_mask_source'] == 'external'
    assert info['brain_mask_path'] == str(valid_mask_file)

    # masked_data must be data zeroed outside the mask, unchanged inside.
    np.testing.assert_array_equal(masked_data[expected], data[expected])
    assert np.all(masked_data[~expected] == 0)


def test_external_mask_skips_background_segmentation(valid_mask_file):
    """The point of the flag: median Otsu must not run at all."""
    with patch("csttool.preprocess.modules.background_segmentation."
               "background_segmentation") as mock_bg:
        resolve_brain_mask(_dwi(), gtab=None, affine=AFFINE,
                           brain_mask_path=valid_mask_file)
    assert not mock_bg.called


def test_no_mask_preserves_automatic_fallback():
    data = _dwi()
    auto_masked = data * 3.0
    auto_mask = np.ones(SHAPE, dtype=bool)
    gtab = MagicMock()

    with patch("csttool.preprocess.modules.background_segmentation."
               "background_segmentation",
               return_value=(auto_masked, auto_mask)) as mock_bg:
        masked_data, brain_mask, info = resolve_brain_mask(
            data, gtab=gtab, affine=AFFINE, brain_mask_path=None
        )

    mock_bg.assert_called_once_with(data, gtab)
    np.testing.assert_array_equal(masked_data, auto_masked)
    np.testing.assert_array_equal(brain_mask, auto_mask)
    assert info['brain_mask_source'] == 'automatic_background_segmentation'
    assert info['brain_mask_path'] is None


# ---------------------------------------------------------------------------
# Validation: fail loudly, never resample
# ---------------------------------------------------------------------------

def test_shape_mismatch_raises(tmp_path):
    bad = _write_mask(tmp_path / "bad_shape.nii.gz",
                      np.ones((6, 7, 9), dtype=np.int16))
    with pytest.raises(ValueError, match="does not match the DWI spatial shape"):
        load_external_brain_mask(bad, SHAPE + (5,), AFFINE)


def test_affine_mismatch_raises(tmp_path):
    shifted = AFFINE.copy()
    shifted[0, 3] += 2.0  # one voxel of translation
    bad = _write_mask(tmp_path / "bad_affine.nii.gz",
                      np.ones(SHAPE, dtype=np.int16), affine=shifted)
    with pytest.raises(ValueError, match="affine is not compatible"):
        load_external_brain_mask(bad, SHAPE + (5,), AFFINE)


def test_negligible_affine_difference_is_accepted(tmp_path):
    """float32 header round-tripping must not reject a matching grid."""
    jittered = AFFINE.copy()
    jittered[1, 3] += 1e-6
    ok = _write_mask(tmp_path / "jitter.nii.gz",
                     np.ones(SHAPE, dtype=np.int16), affine=jittered)
    mask = load_external_brain_mask(ok, SHAPE + (5,), AFFINE)
    assert mask.all()


def test_empty_mask_raises(tmp_path):
    empty = _write_mask(tmp_path / "empty.nii.gz", np.zeros(SHAPE, dtype=np.int16))
    with pytest.raises(ValueError, match="no nonzero voxels"):
        load_external_brain_mask(empty, SHAPE + (5,), AFFINE)


def test_non_finite_mask_raises(tmp_path):
    data = np.ones(SHAPE, dtype=np.float32)
    data[0, 0, 0] = np.nan
    bad = _write_mask(tmp_path / "nan.nii.gz", data)
    with pytest.raises(ValueError, match="non-finite"):
        load_external_brain_mask(bad, SHAPE + (5,), AFFINE)


def test_four_dimensional_mask_raises(tmp_path):
    bad = _write_mask(tmp_path / "mask4d.nii.gz",
                      np.ones(SHAPE + (1,), dtype=np.int16))
    with pytest.raises(ValueError, match="must be 3D"):
        load_external_brain_mask(bad, SHAPE + (5,), AFFINE)


def test_missing_mask_raises(tmp_path):
    with pytest.raises(FileNotFoundError):
        load_external_brain_mask(tmp_path / "nope.nii.gz", SHAPE + (5,), AFFINE)


# ---------------------------------------------------------------------------
# Binarisation
# ---------------------------------------------------------------------------

def test_non_binary_values_are_binarised(tmp_path):
    """0/255 masks and float probability maps: any nonzero value is inside."""
    data = np.zeros(SHAPE, dtype=np.float32)
    data[0] = 255.0
    data[1] = 0.4
    data[2] = -1.0   # nonzero, therefore inside
    data[3] = 1e-8
    path = _write_mask(tmp_path / "graded.nii.gz", data)

    mask = load_external_brain_mask(path, SHAPE + (5,), AFFINE)

    assert mask.dtype == np.bool_
    assert mask[0].all() and mask[1].all() and mask[2].all() and mask[3].all()
    assert not mask[4:].any()


def test_apply_brain_mask_broadcasts_over_volumes():
    data = _dwi(n_vols=3)
    mask = np.zeros(SHAPE, dtype=bool)
    mask[2:4] = True
    masked = apply_brain_mask(data, mask)
    assert masked.shape == data.shape
    np.testing.assert_array_equal(masked[2:4], data[2:4])
    assert np.all(masked[:2] == 0)


# ---------------------------------------------------------------------------
# CLI plumbing: --brain-mask must survive every hop to cmd_track
# ---------------------------------------------------------------------------

@pytest.mark.parametrize("subcommand", ["run", "track"])
def test_cli_defaults_brain_mask_to_none(subcommand, tmp_path, monkeypatch):
    """Absent flag must parse to None, so existing invocations are unchanged."""
    import csttool.cli as cli

    captured = {}

    def fake_func(args):
        captured['args'] = args
        return None

    argv = ["csttool", subcommand, "--nifti", str(tmp_path / "x.nii.gz"),
            "--out", str(tmp_path / "out")]
    monkeypatch.setattr("sys.argv", argv)
    with patch.object(cli, "cmd_run", fake_func), \
            patch.object(cli, "cmd_track", fake_func):
        try:
            cli.main()
        except SystemExit:
            pass

    assert 'args' in captured, "CLI never dispatched to the command function"
    assert getattr(captured['args'], 'brain_mask', 'MISSING') is None


@pytest.mark.parametrize("subcommand", ["run", "track"])
def test_cli_accepts_brain_mask_path(subcommand, tmp_path, monkeypatch):
    import csttool.cli as cli
    from pathlib import Path

    mask = tmp_path / "m.nii.gz"
    captured = {}

    def fake_func(args):
        captured['args'] = args
        return None

    argv = ["csttool", subcommand, "--nifti", str(tmp_path / "x.nii.gz"),
            "--out", str(tmp_path / "out"), "--brain-mask", str(mask)]
    monkeypatch.setattr("sys.argv", argv)
    with patch.object(cli, "cmd_run", fake_func), \
            patch.object(cli, "cmd_track", fake_func):
        try:
            cli.main()
        except SystemExit:
            pass

    assert captured['args'].brain_mask == Path(mask)


def _run_cmd_track(tmp_path, **arg_overrides):
    """Drive cmd_track with every heavy stage mocked; return the mocks we assert on."""
    from csttool.cli.commands import track as track_mod

    nii = tmp_path / "sub-001_desc-preproc_dwi.nii.gz"
    nii.write_bytes(b"")  # only existence is checked; load_nifti is mocked

    args = argparse.Namespace(
        nifti=nii, subject_id="sub-001", out=tmp_path / "out",
        fa_thr=0.2, seed_density=1, step_size=0.5, sh_order=6,
        rng_seed=42, verbose=False,
    )
    for key, value in arg_overrides.items():
        setattr(args, key, value)

    zeros = np.zeros(SHAPE, dtype=float)

    with patch.object(track_mod, "load_nifti") as mock_load, \
            patch.object(track_mod, "get_gtab_for_preproc") as mock_gtab, \
            patch("csttool.preprocess.modules.background_segmentation."
                  "background_segmentation") as mock_bg, \
            patch.object(track_mod, "fit_tensors") as mock_fit, \
            patch.object(track_mod, "estimate_directions"), \
            patch.object(track_mod, "seed_and_stop") as mock_seed, \
            patch.object(track_mod, "run_tractography") as mock_track, \
            patch.object(track_mod, "validate_sh_order", return_value=6), \
            patch.object(track_mod, "save_tracking_outputs") as mock_save:

        mock_load.return_value = (_dwi(), AFFINE, MagicMock())
        mock_gtab.return_value = MagicMock()
        mock_bg.return_value = (np.zeros(SHAPE + (5,)), np.ones(SHAPE, dtype=bool))
        mock_fit.return_value = (MagicMock(), zeros, zeros, zeros, zeros,
                                 np.ones(SHAPE, dtype=bool))
        mock_seed.return_value = (np.zeros((1, 3)), MagicMock())
        mock_track.return_value = []
        mock_save.return_value = {
            "tractogram": "t.trk", "fa_map": "fa.nii.gz", "md_map": "md.nii.gz",
        }

        result = track_mod.cmd_track(args)

    return result, mock_bg, mock_fit, mock_save


def test_cmd_track_without_brain_mask_uses_background_segmentation(tmp_path):
    result, mock_bg, _, mock_save = _run_cmd_track(tmp_path)
    assert result is not None
    assert mock_bg.called
    params = mock_save.call_args.kwargs["tracking_params"]
    assert params["brain_mask_source"] == "automatic_background_segmentation"
    assert params["brain_mask_path"] is None


def test_cmd_track_with_brain_mask_bypasses_segmentation(tmp_path, valid_mask_file):
    result, mock_bg, mock_fit, mock_save = _run_cmd_track(
        tmp_path, brain_mask=valid_mask_file
    )
    assert result is not None
    assert not mock_bg.called, "median Otsu ran despite an external mask"

    # The external mask, not the mocked automatic one, reached the tensor fit.
    fitted_mask = mock_fit.call_args.args[2]
    assert fitted_mask.sum() == 4 * 5 * 6

    params = mock_save.call_args.kwargs["tracking_params"]
    assert params["brain_mask_source"] == "external"
    assert params["brain_mask_path"] == str(valid_mask_file)


def test_cmd_track_reports_invalid_mask_and_returns_none(tmp_path):
    bad = _write_mask(tmp_path / "bad.nii.gz", np.ones((3, 3, 3), dtype=np.int16))
    result, mock_bg, _, _ = _run_cmd_track(tmp_path, brain_mask=bad)
    assert result is None
    assert not mock_bg.called


def test_cmd_run_forwards_brain_mask_to_cmd_track():
    """The flag must survive the cmd_run -> cmd_track Namespace hop."""
    import inspect
    from csttool.cli.commands import run as run_mod

    source = inspect.getsource(run_mod.cmd_run)
    assert "brain_mask=getattr(args, 'brain_mask', None)" in source


# ---------------------------------------------------------------------------
# load_and_mask keeps the same contract
# ---------------------------------------------------------------------------

def test_load_and_mask_accepts_brain_mask_path():
    import inspect
    from csttool.tracking.modules.load_and_mask import load_and_mask

    sig = inspect.signature(load_and_mask)
    assert sig.parameters["brain_mask_path"].default is None


def test_load_and_mask_uses_external_mask(tmp_path, valid_mask_file):
    from csttool.tracking.modules.load_and_mask import load_and_mask

    data = _dwi()
    nii = nib.Nifti1Image(data, AFFINE)

    with patch("csttool.preprocess.modules.load_dataset.load_dataset",
               return_value=(nii, MagicMock(bvals=np.zeros(5)), None, None)), \
            patch("csttool.preprocess.modules.background_segmentation."
                  "background_segmentation") as mock_bg:
        _d, _a, _i, _g, masked_data, brain_mask = load_and_mask(
            str(tmp_path), "dwi", brain_mask_path=valid_mask_file
        )

    assert not mock_bg.called
    assert brain_mask.sum() == 4 * 5 * 6
    assert np.all(masked_data[~brain_mask] == 0)
