"""Preprocessing provenance ledger (M5).

The preprocessing report carried eight keys — timestamp, shape, dtype, voxel
size and five processing flags — with no versions, no geometry log, no
gradient-transform record, no warnings and no way to tell "not requested" from
"requested but failed". The ledger is an ordered, machine-readable list of what
actually happened, in the order it happened, with externally declared work
placed before csttool's own stages.
"""

import json

import numpy as np
import nibabel as nib
import pytest

STATUS_VOCABULARY = {
    "executed",
    "not_requested",
    "failed_continued",
    "declared_external",
}
REQUIRED_STAGE_KEYS = {
    "stage",
    "performed_by",
    "requested",
    "status",
    "method",
    "backend",
    "parameters",
    "input_geometry",
    "output_geometry",
    "gradient_transform",
    "warnings",
    "skip_reason",
}


@pytest.fixture
def dataset(tmp_path):
    in_dir = tmp_path / "in"
    in_dir.mkdir()
    rng = np.random.default_rng(3)
    shape = (12, 12, 8)
    n = 7
    data = rng.random((*shape, n)).astype(np.float32) * 100 + 50

    bvals = np.array([0.0] + [1000.0] * (n - 1))
    dirs = np.array(
        [[1, 0, 0], [0, 1, 0], [0, 0, 1], [1, 1, 0], [1, 0, 1], [0, 1, 1]], float
    )
    dirs /= np.linalg.norm(dirs, axis=1, keepdims=True)
    bvecs = np.vstack([np.zeros(3), dirs])

    affine = np.diag([2.0, 2.0, 2.0, 1.0])
    nib.save(nib.Nifti1Image(data, affine), in_dir / "sub.nii.gz")
    np.savetxt(in_dir / "sub.bval", bvals[None, :], fmt="%g")
    np.savetxt(in_dir / "sub.bvec", bvecs.T, fmt="%.8f")
    return in_dir


def run(dataset, out, **kwargs):
    from csttool.preprocess import run_preprocessing

    kwargs.setdefault("denoise_method", "mppca")
    run_preprocessing(input_dir=dataset, output_dir=out, filename="sub", **kwargs)
    # The suffix depends on whether motion correction *succeeded*, which the
    # caller does not always know (a failed run falls back to _nomc).
    reports = list(out.glob("sub_dwi_preproc_*_report.json"))
    assert len(reports) == 1, [p.name for p in out.glob("*.json")]
    return json.loads(reports[0].read_text())


def stage(report, name):
    matches = [s for s in report["stages"] if s["stage"] == name]
    assert len(matches) == 1, f"expected exactly one {name!r} stage"
    return matches[0]


# ---------------------------------------------------------------------------
# T5.1 — schema
# ---------------------------------------------------------------------------

def test_report_carries_schema_version_provenance_and_stages(dataset, tmp_path):
    report = run(dataset, tmp_path / "out")

    assert report["schema_version"] == 1
    assert "dipy" in report["provenance"]["dependencies"]
    assert report["provenance"]["python_version"]
    assert isinstance(report["stages"], list) and report["stages"]

    for entry in report["stages"]:
        assert REQUIRED_STAGE_KEYS <= set(entry), (
            f"{entry['stage']} missing {REQUIRED_STAGE_KEYS - set(entry)}"
        )
        assert entry["status"] in STATUS_VOCABULARY


def test_versions_live_in_the_shared_provenance_block_only(dataset, tmp_path):
    """No per-stage version duplication — one block, one source of truth."""
    report = run(dataset, tmp_path / "out")
    for entry in report["stages"]:
        assert "version" not in entry
        assert "dependencies" not in entry


# ---------------------------------------------------------------------------
# T5.2 — chronology
# ---------------------------------------------------------------------------

def test_declared_external_work_precedes_csttool_stages(dataset, tmp_path):
    report = run(dataset, tmp_path / "out", external_correction="topup-eddy")

    first = report["stages"][0]
    assert first["stage"] == "external_correction"
    assert first["performed_by"] == "external"
    assert first["status"] == "declared_external"
    assert first["parameters"]["declared"] == "topup-eddy"
    assert first["parameters"]["verified_by_csttool"] is False

    assert all(s["performed_by"] == "csttool" for s in report["stages"][1:])
    # The thesis order serialises as it happened: external correction, then
    # csttool's denoise, then masking.
    names = [s["stage"] for s in report["stages"]]
    assert names.index("external_correction") < names.index("denoise")
    assert names.index("denoise") < names.index("mask")


def test_no_external_stage_when_nothing_is_declared(dataset, tmp_path):
    report = run(dataset, tmp_path / "out")
    assert all(s["stage"] != "external_correction" for s in report["stages"])
    assert report["stages"][0]["stage"] == "load"


@pytest.mark.parametrize("declaration", ["unknown", "none"])
def test_declaring_no_external_work_adds_no_stage(dataset, tmp_path, declaration):
    """`none` means nothing happened; it must not appear as a stage that did.

    Found on a real run: `--input-corrected none` produced a stage with
    status `declared_external` and a skip_reason reading "Performed outside
    csttool, before the data was received" — for a declaration stating that
    no external correction was performed. The stage list records steps that
    occurred; the declaration is recorded regardless, in processing_params.
    """
    report = run(dataset, tmp_path / "out", external_correction=declaration)

    assert all(s["stage"] != "external_correction" for s in report["stages"])
    assert report["processing_params"]["external_correction"] == {
        "declared": declaration,
        "verified_by_csttool": False,
        "source": "user-declaration",
    }


def test_stage_order_matches_execution_order(dataset, tmp_path):
    report = run(dataset, tmp_path / "out", apply_gibbs_correction=True)
    names = [s["stage"] for s in report["stages"]]
    assert names == [
        "load", "reslice", "denoise", "mask", "gibbs", "motion_correction", "save"
    ]


# ---------------------------------------------------------------------------
# T5.3 — status vocabulary
# ---------------------------------------------------------------------------

def test_optional_stages_are_marked_not_requested(dataset, tmp_path):
    report = run(dataset, tmp_path / "out")

    for name in ("reslice", "gibbs", "motion_correction"):
        entry = stage(report, name)
        assert entry["requested"] is False
        assert entry["status"] == "not_requested"
        assert entry["skip_reason"]

    for name in ("load", "denoise", "mask"):
        assert stage(report, name)["status"] == "executed"


def test_requested_optional_stage_is_marked_executed(dataset, tmp_path):
    report = run(dataset, tmp_path / "out", apply_gibbs_correction=True)
    entry = stage(report, "gibbs")
    assert entry["requested"] is True
    assert entry["status"] == "executed"
    assert entry["backend"] == "dipy"


def test_denoise_stage_records_method_and_parameters(dataset, tmp_path):
    report = run(dataset, tmp_path / "out", denoise_method="patch2self")
    entry = stage(report, "denoise")
    assert entry["method"] == "patch2self"
    assert entry["parameters"]["b0_threshold"] == 50


# ---------------------------------------------------------------------------
# T5.4 — gradient transform log
# ---------------------------------------------------------------------------

def test_gradient_transform_is_not_required_without_motion_correction(
    dataset, tmp_path
):
    report = run(dataset, tmp_path / "out")
    gt = stage(report, "motion_correction")["gradient_transform"]
    assert gt["status"] == "not_required"
    assert gt["n_volumes_rotated"] == 0


def test_gradient_transform_records_the_rotation(dataset, tmp_path):
    report = run(dataset, tmp_path / "out", apply_motion_correction=True)
    entry = stage(report, "motion_correction")
    assert entry["status"] == "executed"
    gt = entry["gradient_transform"]
    assert gt["status"] == "bvecs_rotated"
    assert gt["n_volumes_rotated"] == 6  # all DWI volumes, b0 excluded
    assert gt["max_rotation_deg"] is not None
    save_gt = stage(report, "save")["gradient_transform"]
    assert save_gt["status"] == "bvecs_rotated"
    # The save stage reports the same rotation it wrote out, not a bare count.
    assert save_gt["n_volumes_rotated"] == gt["n_volumes_rotated"]
    assert save_gt["max_rotation_deg"] == gt["max_rotation_deg"]


def test_load_stage_reports_ras_reorientation_for_dicom_input(dataset, tmp_path):
    """The DICOM branch reorients image + b-vectors to RAS; the ledger says so.

    NIfTI input reorients nothing, so it reports `none`. Pinned in both
    directions because the two branches are otherwise indistinguishable in
    the report.
    """
    from csttool.preprocess.modules import load_dataset as ld

    report = run(dataset, tmp_path / "nifti_in")
    assert stage(report, "load")["gradient_transform"]["status"] == "none"
    assert stage(report, "load")["parameters"]["source"] == "nifti"

    # Same dataset, but make the loader take its DICOM branch.
    import csttool.preprocess.preprocess as preproc_mod

    monkey = tmp_path / "dicom_out"
    orig = preproc_mod.is_dicom_directory
    try:
        preproc_mod.is_dicom_directory = lambda p: True
        report = run(dataset, monkey)
    finally:
        preproc_mod.is_dicom_directory = orig

    assert stage(report, "load")["gradient_transform"]["status"] == "reoriented_to_ras"
    assert stage(report, "load")["parameters"]["source"] == "dicom"


# ---------------------------------------------------------------------------
# T5.5 — failed but continued
# ---------------------------------------------------------------------------

def test_failed_motion_correction_is_recorded_as_failed_continued(
    dataset, tmp_path, monkeypatch
):
    import csttool.preprocess.preprocess as preproc_mod

    def boom(*args, **kwargs):
        raise RuntimeError("registration blew up")

    monkeypatch.setattr(preproc_mod, "perform_motion_correction", boom)

    report = run(dataset, tmp_path / "out", apply_motion_correction=True)
    entry = stage(report, "motion_correction")

    assert entry["requested"] is True
    assert entry["status"] == "failed_continued"
    assert any("registration blew up" in w for w in entry["warnings"])
    assert entry["gradient_transform"]["status"] == "none"
    # The pair that shipped is the uncorrected one, and the ledger says so.
    assert entry["output_geometry"] == entry["input_geometry"]


# ---------------------------------------------------------------------------
# T5.6 — geometry
# ---------------------------------------------------------------------------

def test_reslice_stage_records_changed_geometry(dataset, tmp_path):
    report = run(dataset, tmp_path / "out", target_voxel_size=(3.0, 3.0, 3.0))
    entry = stage(report, "reslice")

    assert entry["status"] == "executed"
    assert entry["input_geometry"]["zooms"] == pytest.approx([2.0, 2.0, 2.0])
    assert entry["output_geometry"]["zooms"] == pytest.approx([3.0, 3.0, 3.0])
    assert entry["input_geometry"]["shape"] != entry["output_geometry"]["shape"]
    assert entry["input_geometry"]["axis_codes"] == "RAS"
    # Reslicing does not touch the physical gradient frame.
    assert entry["gradient_transform"]["status"] == "not_required"


def test_geometry_is_identity_for_stages_that_do_not_move_data(dataset, tmp_path):
    report = run(dataset, tmp_path / "out")
    entry = stage(report, "denoise")
    assert entry["input_geometry"] == entry["output_geometry"]


# ---------------------------------------------------------------------------
# T5.7 — legacy keys retained
# ---------------------------------------------------------------------------

def test_legacy_report_keys_are_unchanged(dataset, tmp_path):
    report = run(dataset, tmp_path / "out")
    for key in ("timestamp", "filename_stem", "data_shape", "data_dtype",
                "voxel_size", "processing_params"):
        assert key in report

    params = report["processing_params"]
    for key in ("denoise_method", "gibbs_correction", "motion_correction",
                "resliced", "target_voxel_size"):
        assert key in params


def test_save_preprocessed_without_the_new_kwargs_writes_the_old_report(tmp_path):
    """The ledger is additive: omitting it leaves today's report shape."""
    from csttool.preprocess.modules.save_preprocessed import save_preprocessed

    paths = save_preprocessed(
        data=np.zeros((4, 4, 4, 2), np.float32),
        affine=np.eye(4),
        output_dir=tmp_path,
        filename_stem="x",
        processing_params={"denoise_method": "mppca"},
    )
    report = json.loads(paths["report"].read_text())
    assert "stages" not in report
    assert "provenance" not in report
    assert "schema_version" not in report
    assert report["processing_params"] == {"denoise_method": "mppca"}
