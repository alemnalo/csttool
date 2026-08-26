"""``--npeaks`` must reach whole-brain CSA direction estimation, not just the
roi-seeded and bidirectional paths.

The CLI has always accepted ``--npeaks``, but the whole-brain path -- the one
``--extraction-method passthrough`` uses -- pinned ``npeaks=1`` inside
``estimate_directions`` and never read the CLI value. The flag was accepted and
silently discarded, so a run with ``--npeaks 2`` produced a direction field
identical to the default.

The value is lost or preserved at four separate hops, and a test at only one of
them would not have caught the original defect:

    csttool run --npeaks N        cli/__init__.py   (p_run; p_track had no flag)
      -> cmd_run                  builds track_args -- omitted npeaks
      -> cmd_track                called estimate_directions without npeaks
      -> estimate_directions      had no npeaks parameter
      -> peaks_from_model         npeaks=1 hard-coded

These tests walk each hop, and assert the default still resolves to 1 at every
one of them: 1 remains the production default and >1 is experimental only.
"""

import argparse
import inspect
from unittest.mock import MagicMock, patch

import numpy as np
import pytest

import csttool.cli as cli
from csttool.defaults import DEFAULT_NPEAKS


# ---------------------------------------------------------------------------
# The default itself
# ---------------------------------------------------------------------------

def test_production_default_is_one():
    """1 is the production default. Changing it is a separate, evidence-gated decision."""
    assert DEFAULT_NPEAKS == 1


# ---------------------------------------------------------------------------
# Hop 4: estimate_directions -> peaks_from_model
# ---------------------------------------------------------------------------

def _estimate_with(npeaks, gtab, **kwargs):
    """Call estimate_directions with peaks_from_model mocked; return its kwargs."""
    from csttool.tracking.modules.estimate_directions import estimate_directions

    data = np.zeros((2, 2, 2, 7), dtype=float)
    data[..., 0] = 100.0
    data[..., 1:] = 50.0
    mask = np.ones((2, 2, 2), dtype=bool)

    # estimate_directions imports peaks_from_model at call time, so the source
    # module is the patch target.
    with patch("dipy.direction.peaks_from_model") as mock_peaks:
        mock_peaks.return_value = MagicMock()
        if npeaks is _SENTINEL:
            estimate_directions(data, gtab, mask, sh_order=2, verbose=False, **kwargs)
        else:
            estimate_directions(data, gtab, mask, sh_order=2, npeaks=npeaks,
                                verbose=False, **kwargs)

    assert mock_peaks.called, "peaks_from_model was never reached"
    return mock_peaks.call_args.kwargs


_SENTINEL = object()


def test_estimate_directions_defaults_to_one(synthetic_gtab):
    """Omitting npeaks must behave exactly as the previous hard-coded value did."""
    kwargs = _estimate_with(_SENTINEL, synthetic_gtab)
    assert kwargs["npeaks"] == 1


@pytest.mark.parametrize("npeaks", [1, 2, 3, 5])
def test_estimate_directions_forwards_npeaks(npeaks, synthetic_gtab):
    """The value reaches DIPY unchanged -- this is the hop that was hard-coded."""
    kwargs = _estimate_with(npeaks, synthetic_gtab)
    assert kwargs["npeaks"] == npeaks


def test_estimate_directions_changes_nothing_else(synthetic_gtab):
    """npeaks must not disturb the other peaks_from_model arguments.

    The experiment's premise is that only npeaks varies between runs; if passing
    it perturbed the sphere, the thresholds or the mask, the comparison would be
    confounded.
    """
    from csttool.defaults import (
        DEFAULT_MIN_SEPARATION_ANGLE,
        DEFAULT_RELATIVE_PEAK_THRESHOLD,
    )

    one = _estimate_with(1, synthetic_gtab)
    five = _estimate_with(5, synthetic_gtab)

    for key in ("relative_peak_threshold", "min_separation_angle"):
        assert one[key] == five[key], f"{key} changed with npeaks"
    assert one["relative_peak_threshold"] == DEFAULT_RELATIVE_PEAK_THRESHOLD
    assert one["min_separation_angle"] == DEFAULT_MIN_SEPARATION_ANGLE
    assert one["sphere"] is not None
    assert np.array_equal(one["mask"], five["mask"])


def test_no_hardcoded_npeaks_remains_in_whole_brain_estimation():
    """Guards the regression directly: the literal pin must not come back."""
    from csttool.tracking.modules import estimate_directions as mod

    src = inspect.getsource(mod)
    assert "npeaks=1" not in src, (
        "estimate_directions pins npeaks again; the CLI value would be discarded"
    )
    assert "npeaks=npeaks" in src


# ---------------------------------------------------------------------------
# Hop 3: cmd_track -> estimate_directions, and the provenance record
# ---------------------------------------------------------------------------

def _run_cmd_track(tmp_path, **arg_overrides):
    """Drive cmd_track with every heavy stage mocked.

    Returns (estimate_directions_spy, save_tracking_outputs_spy).
    """
    from csttool.cli.commands import track as track_mod

    nii = tmp_path / "sub-001_desc-preproc_dwi.nii.gz"
    nii.write_bytes(b"")  # only existence is checked; load_nifti is mocked

    args = argparse.Namespace(
        nifti=nii,
        subject_id="sub-001",
        out=tmp_path / "out",
        fa_thr=0.2,
        seed_density=1,
        step_size=0.5,
        sh_order=6,
        rng_seed=42,
        verbose=False,
    )
    for key, value in arg_overrides.items():
        setattr(args, key, value)

    shape = (2, 2, 2)
    zeros = np.zeros(shape, dtype=float)

    with patch.object(track_mod, "load_nifti") as mock_load, \
            patch.object(track_mod, "get_gtab_for_preproc") as mock_gtab, \
            patch("csttool.preprocess.modules.background_segmentation."
                  "background_segmentation") as mock_bg, \
            patch.object(track_mod, "fit_tensors") as mock_fit, \
            patch.object(track_mod, "estimate_directions") as mock_est, \
            patch.object(track_mod, "seed_and_stop") as mock_seed, \
            patch.object(track_mod, "run_tractography") as mock_track, \
            patch.object(track_mod, "validate_sh_order", return_value=6), \
            patch.object(track_mod, "save_tracking_outputs") as mock_save:

        mock_load.return_value = (np.zeros(shape + (7,)), np.eye(4), MagicMock())
        mock_gtab.return_value = MagicMock()
        mock_bg.return_value = (np.zeros(shape + (7,)), np.ones(shape, dtype=bool))
        mock_fit.return_value = (MagicMock(), zeros, zeros, zeros, zeros,
                                 np.ones(shape, dtype=bool))
        mock_est.return_value = MagicMock()
        mock_seed.return_value = (np.zeros((1, 3)), MagicMock())
        mock_track.return_value = []
        mock_save.return_value = {
            "tractogram": "t.trk", "fa_map": "fa.nii.gz", "md_map": "md.nii.gz",
        }

        track_mod.cmd_track(args)

    assert mock_est.called, "estimate_directions was never reached"
    return mock_est, mock_save


def test_cmd_track_defaults_to_one(tmp_path):
    """An args Namespace without npeaks must still resolve to the default."""
    mock_est, _ = _run_cmd_track(tmp_path)
    assert mock_est.call_args.kwargs["npeaks"] == DEFAULT_NPEAKS


@pytest.mark.parametrize("npeaks", [1, 2, 3, 5])
def test_cmd_track_forwards_npeaks(npeaks, tmp_path):
    mock_est, _ = _run_cmd_track(tmp_path, npeaks=npeaks)
    assert mock_est.call_args.kwargs["npeaks"] == npeaks


@pytest.mark.parametrize("npeaks", [1, 2, 5])
def test_cmd_track_records_npeaks_in_provenance(npeaks, tmp_path):
    """The value used must be recoverable from the run's own outputs.

    tracking_params is written into the tracking report JSON and copied into
    pipeline_metadata['tracking'] by cmd_run, so an experimental run can later
    establish which npeaks produced it.
    """
    _, mock_save = _run_cmd_track(tmp_path, npeaks=npeaks)
    params = mock_save.call_args.kwargs["tracking_params"]
    assert params["npeaks"] == npeaks


def test_cmd_track_provenance_defaults_to_one(tmp_path):
    _, mock_save = _run_cmd_track(tmp_path)
    assert mock_save.call_args.kwargs["tracking_params"]["npeaks"] == DEFAULT_NPEAKS


# ---------------------------------------------------------------------------
# Hop 2: cmd_run -> cmd_track
# ---------------------------------------------------------------------------

def _run_cmd_run(tmp_path, **arg_overrides):
    """Drive cmd_run to the track step; return the Namespace cmd_track received."""
    from csttool.cli.commands import run as run_mod

    nii = tmp_path / "in.nii.gz"
    nii.write_bytes(b"")
    out = tmp_path / "out"
    out.mkdir()

    args = argparse.Namespace(
        nifti=nii, dicom=None, out=out, subject_id="sub-001",
        preprocess=False, continue_on_error=False,
    )
    for key, value in arg_overrides.items():
        setattr(args, key, value)

    with patch.object(run_mod, "cmd_track") as mock_track:
        mock_track.return_value = None
        run_mod.cmd_run(args)

    assert mock_track.called, "cmd_run never reached the track step"
    return mock_track.call_args[0][0]


def test_cmd_run_defaults_to_one(tmp_path):
    assert _run_cmd_run(tmp_path).npeaks == DEFAULT_NPEAKS


@pytest.mark.parametrize("npeaks", [1, 2, 3, 5])
def test_cmd_run_forwards_npeaks_to_track(npeaks, tmp_path):
    """This hop silently dropped the value: track_args simply omitted npeaks."""
    assert _run_cmd_run(tmp_path, npeaks=npeaks).npeaks == npeaks


# ---------------------------------------------------------------------------
# Hop 1: the CLI parsers
# ---------------------------------------------------------------------------

def _dispatched_args(command, func_attr, argv):
    """Run main() far enough to capture the Namespace the command is dispatched with."""
    captured = MagicMock()
    with patch.object(cli, func_attr, captured):
        with patch("sys.argv", ["csttool", command] + argv):
            cli.main()
    assert captured.called, f"'{command}' never dispatched; argv may be incomplete"
    return captured.call_args[0][0]


CLI_COMMANDS = [
    ("run", "cmd_run", ["--nifti", "in.nii.gz", "--out", "out"]),
    ("track", "cmd_track", ["--nifti", "in.nii.gz", "--out", "out"]),
]


@pytest.mark.parametrize("command,func_attr,argv", CLI_COMMANDS)
def test_cli_defaults_to_one(command, func_attr, argv):
    args = _dispatched_args(command, func_attr, argv)
    assert args.npeaks == DEFAULT_NPEAKS


@pytest.mark.parametrize("command,func_attr,argv", CLI_COMMANDS)
@pytest.mark.parametrize("npeaks", [1, 2, 3, 5])
def test_cli_accepts_explicit_npeaks(command, func_attr, argv, npeaks):
    """`csttool track` had no --npeaks at all before this change."""
    args = _dispatched_args(command, func_attr, argv + ["--npeaks", str(npeaks)])
    assert args.npeaks == npeaks


@pytest.mark.parametrize("command", ["run", "track"])
def test_help_text_does_not_claim_roi_seeded_only(command, capsys):
    """The help text said roi-seeded/bidirectional only, which is now wrong."""
    with patch("sys.argv", ["csttool", command, "--help"]):
        with pytest.raises(SystemExit):
            cli.main()
    help_text = capsys.readouterr().out
    assert "--npeaks" in help_text
    start = help_text.index("--npeaks")
    excerpt = help_text[start:start + 400].lower()
    assert "roi-seeded/bidirectional tracking" not in excerpt


# ---------------------------------------------------------------------------
# The roi-seeded / bidirectional paths must be untouched
# ---------------------------------------------------------------------------

def test_bidirectional_still_forwards_npeaks_to_peaks_from_model():
    from csttool.extract.modules import bidirectional_filtering as mod

    src = inspect.getsource(mod.extract_cst_bidirectional)
    assert "npeaks=npeaks" in src, "bidirectional no longer forwards npeaks"
    assert "npeaks" in inspect.signature(mod.extract_cst_bidirectional).parameters


def test_roi_seeded_still_forwards_npeaks_to_peaks_from_model():
    from csttool.extract.modules import roi_seeded_tracking as mod

    assert "npeaks" in inspect.signature(mod.extract_cst_roi_seeded).parameters
    src = inspect.getsource(mod)
    assert "npeaks=npeaks" in src, "roi-seeded no longer forwards npeaks"


def test_extract_command_still_reads_npeaks_from_args():
    """cmd_extract's roi-seeded/bidirectional branches must keep their getattr."""
    from csttool.cli.commands import extract as mod

    src = inspect.getsource(mod)
    assert src.count("getattr(args, 'npeaks', DEFAULT_NPEAKS)") >= 2
