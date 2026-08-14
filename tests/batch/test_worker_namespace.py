"""Batch worker namespace construction (M1).

`BatchConfig.preprocessing` is flattened onto an ``argparse.Namespace`` via
``asdict()``, but ``cmd_run`` reads ``args.preprocess``. Nothing ever set that
attribute, so every batch subject silently took the pass-through branch
regardless of the documented batch default. These tests pin the translation at
the worker boundary and the surrounding namespace contract.
"""

import argparse
import multiprocessing
from pathlib import Path
from unittest.mock import patch

import pytest

from csttool.batch.batch import (
    BatchConfig,
    SubjectSpec,
    _build_run_namespace,
    _run_subject_worker,
    compute_config_hash,
)


@pytest.fixture
def nifti_subject(tmp_path):
    return SubjectSpec(
        subject_id="sub-01",
        session_id=None,
        input_path=tmp_path / "sub-01_dwi.nii.gz",
        input_type="nifti",
    )


@pytest.fixture
def dicom_subject(tmp_path):
    return SubjectSpec(
        subject_id="sub-02",
        session_id="ses-01",
        input_path=tmp_path / "dicoms",
        input_type="dicom",
        series_uid="1.2.3.4",
    )


# ---------------------------------------------------------------------------
# T1.1 — the regression this milestone exists for
# ---------------------------------------------------------------------------

@pytest.mark.parametrize("preprocessing", [True, False])
def test_preprocessing_flag_reaches_cmd_run_attribute(
    tmp_path, nifti_subject, preprocessing
):
    """BatchConfig.preprocessing must arrive as the `preprocess` attribute."""
    config = BatchConfig(out=tmp_path / "out", preprocessing=preprocessing)
    ns = _build_run_namespace(nifti_subject, config, tmp_path / "work")

    assert ns.preprocess is preprocessing
    # The original field name is retained: it participates in config hashing
    # and must not be renamed.
    assert ns.preprocessing is preprocessing


# ---------------------------------------------------------------------------
# T1.2 — option forwarding and per-subject overrides
# ---------------------------------------------------------------------------

def test_denoise_method_forwarded(tmp_path, nifti_subject):
    config = BatchConfig(out=tmp_path / "out", denoise_method="patch2self")
    ns = _build_run_namespace(nifti_subject, config, tmp_path / "work")
    assert ns.denoise_method == "patch2self"


def test_subject_options_override_globals(tmp_path):
    config = BatchConfig(
        out=tmp_path / "out", denoise_method="mppca", preprocessing=True
    )
    subject = SubjectSpec(
        subject_id="sub-01",
        session_id=None,
        input_path=tmp_path / "sub-01_dwi.nii.gz",
        input_type="nifti",
        options={"denoise_method": "nlmeans", "preprocessing": False},
    )
    ns = _build_run_namespace(subject, config, tmp_path / "work")

    assert ns.denoise_method == "nlmeans"
    # A per-subject override of the documented global key must also reach
    # cmd_run through the translated attribute, not just the raw one.
    assert ns.preprocess is False


def test_batch_default_denoise_method_matches_shared_default(tmp_path):
    from csttool.defaults import DEFAULT_DENOISE_METHOD

    config = BatchConfig(out=tmp_path / "out")
    assert config.denoise_method == DEFAULT_DENOISE_METHOD


# ---------------------------------------------------------------------------
# T1.3 — input type wiring
# ---------------------------------------------------------------------------

def test_nifti_input_attributes(tmp_path, nifti_subject):
    config = BatchConfig(out=tmp_path / "out")
    work = tmp_path / "work"
    ns = _build_run_namespace(nifti_subject, config, work)

    assert ns.nifti == nifti_subject.input_path
    assert ns.dicom is None
    assert ns.out == work
    assert ns.subject_id == "sub-01"
    assert ns.session_id is None


def test_dicom_input_attributes(tmp_path, dicom_subject):
    config = BatchConfig(out=tmp_path / "out")
    ns = _build_run_namespace(dicom_subject, config, tmp_path / "work")

    assert ns.dicom == dicom_subject.input_path
    assert ns.nifti is None
    assert ns.series is None
    assert ns.series_uid == "1.2.3.4"
    assert ns.session_id == "ses-01"


# ---------------------------------------------------------------------------
# T1.4 — runtime confirmation through the real worker entry point
# ---------------------------------------------------------------------------

@pytest.mark.parametrize("preprocessing", [True, False])
def test_worker_passes_preprocess_to_cmd_run(tmp_path, nifti_subject, preprocessing):
    config = BatchConfig(out=tmp_path / "out", preprocessing=preprocessing)
    captured = {}

    def fake_cmd_run(args):
        captured["args"] = args

    queue = multiprocessing.Queue()
    with patch("csttool.cli.commands.run.cmd_run", side_effect=fake_cmd_run):
        _run_subject_worker(nifti_subject, config, tmp_path / "sub.log", queue)

    result = queue.get(timeout=5)
    assert result.status == "success", result.error
    assert captured["args"].preprocess is preprocessing


# ---------------------------------------------------------------------------
# T1.5 — a batch-originated namespace actually reaches cmd_preprocess
# ---------------------------------------------------------------------------

def test_batch_namespace_reaches_cmd_preprocess(tmp_path, nifti_subject):
    """The end of the chain: batch preprocessing=True invokes cmd_preprocess."""
    from csttool.cli.commands.run import cmd_run

    config = BatchConfig(out=tmp_path / "out", preprocessing=True)
    args = _build_run_namespace(nifti_subject, config, tmp_path / "work")
    # cmd_run's import step needs an existing NIfTI; point at a real file.
    nifti = tmp_path / "sub-01_dwi.nii.gz"
    nifti.write_bytes(b"")
    args.nifti = nifti
    args.out = tmp_path / "work"
    args.out.mkdir(parents=True, exist_ok=True)

    with patch("csttool.cli.commands.run.cmd_preprocess") as mock_preprocess:
        mock_preprocess.return_value = None  # forces an early, harmless failure
        with patch("csttool.cli.commands.run.cmd_track", return_value=None):
            cmd_run(args)

    assert mock_preprocess.called
    preproc_ns = mock_preprocess.call_args[0][0]
    assert preproc_ns.denoise_method == config.denoise_method


# ---------------------------------------------------------------------------
# T1.6 — the never-functional CLI choice is gone
# ---------------------------------------------------------------------------

def test_batch_rejects_denoise_method_none(tmp_path):
    """`none` was offered but never implemented; denoise() raises on it.

    Before M1 the choice was unreachable (batch preprocessing never ran). Now
    that preprocessing is reachable it would fail every subject, so the choice
    is removed rather than silently accepted.
    """
    from csttool.cli import main

    argv = [
        "csttool", "batch", "--bids-dir", str(tmp_path),
        "--out", str(tmp_path / "out"), "--denoise-method", "none",
    ]
    with patch("sys.argv", argv), patch("csttool.cli.cmd_batch") as mock_batch:
        with pytest.raises(SystemExit) as exc:
            main()
    assert exc.value.code == 2
    assert not mock_batch.called


@pytest.mark.parametrize("method", ["nlmeans", "patch2self", "mppca"])
def test_batch_accepts_supported_denoise_methods(tmp_path, method):
    from csttool.cli import main

    argv = [
        "csttool", "batch", "--bids-dir", str(tmp_path),
        "--out", str(tmp_path / "out"), "--denoise-method", method,
    ]
    with patch("sys.argv", argv), patch("csttool.cli.cmd_batch") as mock_batch:
        main()
    assert mock_batch.call_args[0][0].denoise_method == method


# ---------------------------------------------------------------------------
# T1.7 — resume compatibility
# ---------------------------------------------------------------------------

def test_config_hash_is_stable_for_a_fixed_config(tmp_path):
    """The hash feeds `_done.json` resume checks; the field set must not drift."""
    config = BatchConfig(
        out=Path("/fixed/out"),
        denoise_method="mppca",
        preprocessing=True,
        generate_pdf=False,
    )
    # Value recorded from the pre-M1 implementation: existing `_done.json`
    # markers must stay comparable after the worker-boundary fix.
    assert compute_config_hash(config) == (
        "4c2757bf8b9652b54a0db7a8366e2c1526d2c62b10886f2009c378d6299213ec"
    )
