"""External-preprocessing declaration (M4).

csttool skips its own preprocessing by default and its metadata asserted
"Skipped (External Preprocessing Used)" — a claim about the input that nobody
had made and csttool cannot check. `--input-corrected` turns that assumption
into an explicit, ordered, *declared* record. Every value is a user
declaration; nothing here is ever marked verified.
"""

import argparse
import json
import sys
from pathlib import Path
from unittest.mock import patch

import numpy as np
import nibabel as nib
import pytest

from csttool.cli import main


def run_cli(args):
    with patch.object(sys, "argv", ["csttool"] + args):
        try:
            main()
            return True
        except SystemExit as e:
            return e.code == 0


@pytest.fixture
def mocked_pipeline(tmp_path):
    """`run` with every stage mocked, so only the metadata plumbing is exercised."""
    nifti = tmp_path / "test.nii.gz"
    nifti.touch()
    out = tmp_path / "out"

    with patch("csttool.cli.commands.run.cmd_preprocess") as m_pre, \
            patch("csttool.cli.commands.run.cmd_import") as m_imp, \
            patch("csttool.cli.commands.run.cmd_track") as m_track, \
            patch("csttool.cli.commands.run.cmd_extract") as m_ext, \
            patch("csttool.cli.commands.run.cmd_metrics") as m_met:
        m_imp.return_value = {"nifti_path": str(nifti)}
        m_pre.return_value = {"preprocessed_path": str(out / "preproc.nii.gz")}
        m_track.return_value = {
            "tractogram_path": str(out / "t.trk"),
            "fa_path": str(out / "fa.nii.gz"),
            "md_path": str(out / "md.nii.gz"),
        }
        m_ext.return_value = {
            "cst_left_path": str(out / "l.trk"),
            "cst_right_path": str(out / "r.trk"),
            "stats": {"cst_total_count": 100},
        }
        m_met.return_value = {"success": True}
        yield {
            "nifti": nifti, "out": out, "preprocess": m_pre,
            "track": m_track, "metrics": m_met,
        }


def preprocessing_metadata(mocked):
    return mocked["metrics"].call_args[0][0].pipeline_metadata["preprocessing"]


# ---------------------------------------------------------------------------
# T4.1 — the flag exists and defaults to `unknown`
# ---------------------------------------------------------------------------

def test_default_declaration_is_unknown(mocked_pipeline):
    assert run_cli([
        "run", "--nifti", str(mocked_pipeline["nifti"]),
        "--out", str(mocked_pipeline["out"] / "d"), "--skip-check",
    ])
    meta = preprocessing_metadata(mocked_pipeline)
    assert meta["external_correction"]["declared"] == "unknown"


def test_unknown_value_is_rejected(tmp_path):
    nifti = tmp_path / "t.nii.gz"
    nifti.touch()
    with patch.object(sys, "argv", [
        "csttool", "run", "--nifti", str(nifti), "--out", str(tmp_path / "o"),
        "--input-corrected", "definitely-not-a-choice",
    ]):
        with pytest.raises(SystemExit) as exc:
            main()
    assert exc.value.code == 2


# ---------------------------------------------------------------------------
# T4.2 — recorded on the pass-through branch, never as verified
# ---------------------------------------------------------------------------

def test_declaration_recorded_on_passthrough(mocked_pipeline):
    assert run_cli([
        "run", "--nifti", str(mocked_pipeline["nifti"]),
        "--out", str(mocked_pipeline["out"] / "p"), "--skip-check",
        "--input-corrected", "topup-eddy",
    ])
    meta = preprocessing_metadata(mocked_pipeline)

    assert meta["external_correction"] == {
        "declared": "topup-eddy",
        "verified_by_csttool": False,
        "source": "user-declaration",
    }
    # Factual status: csttool skipped its own preprocessing. What happened
    # before csttool saw the data lives in `external_correction`, as a
    # declaration — the old status string asserted it as fact.
    assert meta["status"] == "Skipped"
    assert meta["performed_by_csttool"] is False


def test_status_no_longer_asserts_unverified_external_preprocessing(mocked_pipeline):
    """Regression lock, intentionally re-contracted in M4."""
    assert run_cli([
        "run", "--nifti", str(mocked_pipeline["nifti"]),
        "--out", str(mocked_pipeline["out"] / "s"), "--skip-check",
    ])
    meta = preprocessing_metadata(mocked_pipeline)
    assert "External Preprocessing Used" not in json.dumps(meta)
    assert meta["external_correction"]["declared"] == "unknown"
    assert meta["external_correction"]["verified_by_csttool"] is False


# ---------------------------------------------------------------------------
# T4.3 — recorded on the executed branch too
# ---------------------------------------------------------------------------

def test_declaration_forwarded_to_preprocess_and_recorded(mocked_pipeline):
    assert run_cli([
        "run", "--nifti", str(mocked_pipeline["nifti"]),
        "--out", str(mocked_pipeline["out"] / "e"), "--skip-check",
        "--preprocess", "--input-corrected", "eddy-only",
    ])

    preproc_ns = mocked_pipeline["preprocess"].call_args[0][0]
    assert preproc_ns.input_corrected == "eddy-only"

    meta = preprocessing_metadata(mocked_pipeline)
    assert meta["status"] == "Executed"
    assert meta["performed_by_csttool"] is True
    assert meta["external_correction"]["declared"] == "eddy-only"
    assert meta["external_correction"]["verified_by_csttool"] is False


# ---------------------------------------------------------------------------
# T4.4 — conflict policy: warn loudly, record it, keep going
# ---------------------------------------------------------------------------

@pytest.mark.parametrize("declaration", ["topup-eddy", "eddy-only"])
def test_double_motion_correction_warns_but_does_not_block(
    mocked_pipeline, capsys, declaration
):
    """The declaration is unverified, so it must not veto an explicit flag.

    A hard error would let a mistyped declaration block a legitimate run. The
    combination is usually undesirable, not intrinsically invalid, so it warns
    and records the warning instead.
    """
    assert run_cli([
        "run", "--nifti", str(mocked_pipeline["nifti"]),
        "--out", str(mocked_pipeline["out"] / "c"), "--skip-check",
        "--preprocess", "--perform-motion-correction",
        "--input-corrected", declaration,
    ])

    captured = capsys.readouterr().out
    assert "double motion correction" in captured.lower()

    meta = preprocessing_metadata(mocked_pipeline)
    assert any("motion correction" in w.lower() for w in meta["warnings"])
    # It ran anyway.
    assert mocked_pipeline["preprocess"].called


def test_no_conflict_warning_for_none_declaration(mocked_pipeline, capsys):
    assert run_cli([
        "run", "--nifti", str(mocked_pipeline["nifti"]),
        "--out", str(mocked_pipeline["out"] / "n"), "--skip-check",
        "--preprocess", "--perform-motion-correction",
        "--input-corrected", "none",
    ])
    assert "double motion correction" not in capsys.readouterr().out.lower()
    assert preprocessing_metadata(mocked_pipeline)["warnings"] == []


def test_conflict_warning_also_fires_on_standalone_preprocess(tmp_path, capsys):
    from csttool.cli.commands.preprocess import cmd_preprocess

    nifti = tmp_path / "sub.nii.gz"
    nib.save(nib.Nifti1Image(np.zeros((4, 4, 4, 2), np.float32), np.eye(4)), nifti)
    args = argparse.Namespace(
        nifti=nifti, dicom=None, out=tmp_path / "out", coil_count=4,
        target_voxel_size=None, perform_motion_correction=True,
        input_corrected="topup-eddy",
    )
    with patch("csttool.preprocess.run_preprocessing") as mock_run:
        mock_run.return_value = None
        cmd_preprocess(args)

    assert "double motion correction" in capsys.readouterr().out.lower()
    assert mock_run.call_args.kwargs["external_correction"] == "topup-eddy"


# ---------------------------------------------------------------------------
# T4.5 — advisory on the undeclared pass-through path
# ---------------------------------------------------------------------------

def test_advisory_printed_when_undeclared(mocked_pipeline, capsys):
    assert run_cli([
        "run", "--nifti", str(mocked_pipeline["nifti"]),
        "--out", str(mocked_pipeline["out"] / "a"), "--skip-check",
    ])
    out = capsys.readouterr().out.lower()
    assert "--input-corrected" in out
    assert "no preprocessing" in out or "assumes" in out


def test_no_advisory_when_external_correction_declared(mocked_pipeline, capsys):
    assert run_cli([
        "run", "--nifti", str(mocked_pipeline["nifti"]),
        "--out", str(mocked_pipeline["out"] / "b"), "--skip-check",
        "--input-corrected", "topup-eddy",
    ])
    assert "--input-corrected" not in capsys.readouterr().out


# ---------------------------------------------------------------------------
# T4.6 — batch reaches it through the documented per-subject manifest options
# ---------------------------------------------------------------------------

def test_batch_subject_options_carry_the_declaration(tmp_path):
    from csttool.batch.batch import BatchConfig, SubjectSpec, _build_run_namespace

    subject = SubjectSpec(
        subject_id="sub-01", session_id=None,
        input_path=tmp_path / "sub-01_dwi.nii.gz", input_type="nifti",
        options={"input_corrected": "eddy-only"},
    )
    ns = _build_run_namespace(subject, BatchConfig(out=tmp_path / "o"), tmp_path / "w")
    assert ns.input_corrected == "eddy-only"


# ---------------------------------------------------------------------------
# Chronology: a declaration describes what happened *before* csttool
# ---------------------------------------------------------------------------

def test_declaration_is_recorded_in_the_preprocessing_report(tmp_path):
    from csttool.preprocess import run_preprocessing

    in_dir = tmp_path / "in"
    in_dir.mkdir()
    rng = np.random.default_rng(0)
    data = rng.random((8, 8, 6, 4)).astype(np.float32) * 100
    nib.save(nib.Nifti1Image(data, np.eye(4)), in_dir / "sub.nii.gz")
    np.savetxt(in_dir / "sub.bval", np.array([[0, 1000, 1000, 1000]]), fmt="%g")
    np.savetxt(
        in_dir / "sub.bvec",
        np.array([[0, 1, 0, 0], [0, 0, 1, 0], [0, 0, 0, 1]], float),
        fmt="%.8f",
    )

    run_preprocessing(
        input_dir=in_dir, output_dir=tmp_path / "out", filename="sub",
        denoise_method="mppca", external_correction="topup-eddy",
    )

    report = json.loads(
        (tmp_path / "out" / "sub_dwi_preproc_nomc_report.json").read_text()
    )
    declared = report["processing_params"]["external_correction"]
    assert declared == {
        "declared": "topup-eddy",
        "verified_by_csttool": False,
        "source": "user-declaration",
    }
