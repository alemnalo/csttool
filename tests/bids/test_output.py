"""Unit tests for csttool.bids.output helper functions and QC naming."""

import pytest
from csttool.bids.output import sanitize_bids_label, parse_dicom_age
from csttool.cli.commands.run import _resolve_qc_name, _QC_NAMES


# ---------------------------------------------------------------------------
# sanitize_bids_label
# ---------------------------------------------------------------------------

class TestSanitizeBidsLabel:
    def test_spaces_replaced(self):
        assert " " not in sanitize_bids_label("my label")

    def test_slashes_replaced(self):
        result = sanitize_bids_label("path/to/series")
        assert "/" not in result

    def test_numbers_first_prepended(self):
        result = sanitize_bids_label("3Tesla")
        assert result[0].isalpha()

    def test_consecutive_separators_collapsed(self):
        result = sanitize_bids_label("a  b--c")
        assert "--" not in result
        assert "  " not in result

    def test_max_len_respected(self):
        result = sanitize_bids_label("a" * 50)
        assert len(result) <= 20

    def test_max_len_custom(self):
        result = sanitize_bids_label("abcdefghij", max_len=5)
        assert len(result) <= 5

    def test_empty_string_returns_x(self):
        result = sanitize_bids_label("")
        assert result == "x"

    def test_all_symbols_returns_x(self):
        result = sanitize_bids_label("---")
        assert result[0].isalpha()

    def test_alphanumeric_unchanged(self):
        assert sanitize_bids_label("DWI01") == "DWI01"

    def test_leading_trailing_hyphens_stripped(self):
        result = sanitize_bids_label("!hello!")
        assert not result.startswith("-")
        assert not result.endswith("-")


# ---------------------------------------------------------------------------
# parse_dicom_age
# ---------------------------------------------------------------------------

class TestParseDicomAge:
    def test_years(self):
        assert parse_dicom_age("034Y") == 34.0

    def test_months(self):
        assert parse_dicom_age("018M") == pytest.approx(1.5, rel=1e-3)

    def test_days(self):
        result = parse_dicom_age("002D")
        assert result is not None
        assert result < 0.01

    def test_weeks(self):
        result = parse_dicom_age("004W")
        assert result is not None
        assert result < 0.1

    def test_empty_string_returns_none(self):
        assert parse_dicom_age("") is None

    def test_none_returns_none(self):
        assert parse_dicom_age(None) is None

    def test_invalid_format_returns_none(self):
        assert parse_dicom_age("notanage") is None

    def test_lowercase_unit(self):
        assert parse_dicom_age("025y") == 25.0

    def test_zero_years(self):
        assert parse_dicom_age("000Y") == 0.0


# ---------------------------------------------------------------------------
# _resolve_qc_name
# ---------------------------------------------------------------------------

class TestResolveQcName:
    def test_known_preproc_suffix(self):
        stage, label = _resolve_qc_name("sub001_brain_mask_qc.png")
        assert stage == "preproc"
        assert label == "brainmask"

    def test_known_tracking_suffix(self):
        stage, label = _resolve_qc_name("subject_tensor_maps.png")
        assert stage == "tracking"
        assert label == "tensormaps"

    def test_known_extraction_suffix(self):
        stage, label = _resolve_qc_name("sub001_registration_qc.png")
        assert stage == "extraction"
        assert label == "registration"

    def test_known_metrics_suffix(self):
        stage, label = _resolve_qc_name("sub001_tractogram_qc_sagittal.png")
        assert stage == "metrics"
        assert label == "tractogram-sagittal"

    def test_unknown_falls_back_to_misc(self):
        stage, label = _resolve_qc_name("something_unexpected.png")
        assert stage == "misc"
        assert "something_unexpected" in label

    def test_all_known_suffixes_resolve(self):
        for suffix in _QC_NAMES:
            stage, label = _resolve_qc_name(f"sub001{suffix}")
            assert stage != "misc", f"suffix {suffix!r} unexpectedly fell through to misc"


# ---------------------------------------------------------------------------
# write_derivative_sidecar - extra scientific keys (visualization refactor M2/M3)
# ---------------------------------------------------------------------------
from csttool.bids.output import write_derivative_sidecar  # noqa: E402


class TestWriteDerivativeSidecarExtra:
    def test_extra_merged_after_standard_keys(self, tmp_path):
        nii = tmp_path / "sub-01_desc-V1_dwimap.nii.gz"
        nii.write_bytes(b"")  # only the stem is read; no image needed
        sidecar = write_derivative_sidecar(
            nii, sources=["bids::sub-01/dwi/sub-01_dwi.nii.gz"],
            description="V1 world-frame field",
            extra={"VectorFrame": "world-RAS", "AffineDeterminant": -1.0},
        )
        import json
        data = json.loads(sidecar.read_text())
        assert data["VectorFrame"] == "world-RAS"
        assert data["AffineDeterminant"] == -1.0
        # Standard keys still present.
        assert data["Sources"] and data["SpatialReference"] == "native"
        assert data["Description"] == "V1 world-frame field"

    def test_extra_collision_with_reserved_key_raises(self, tmp_path):
        nii = tmp_path / "x.nii.gz"
        nii.write_bytes(b"")
        with pytest.raises(ValueError):
            write_derivative_sidecar(
                nii, sources=[], description="d",
                extra={"Description": "should collide"},
            )

    def test_extra_none_is_noop(self, tmp_path):
        nii = tmp_path / "y.nii.gz"
        nii.write_bytes(b"")
        sidecar = write_derivative_sidecar(nii, sources=[], description="d")
        import json
        data = json.loads(sidecar.read_text())
        assert "VectorFrame" not in data
        assert data["Description"] == "d"


# ---------------------------------------------------------------------------
# New QC names + NIfTI survival across the BIDS reorganiser (M6: §13.2 P-1..P-5)
# ---------------------------------------------------------------------------



class TestNewQcNames:
    def test_dec_fa_resolves_to_tracking(self):
        stage, label = _resolve_qc_name("sub001_dec_fa.png")
        assert (stage, label) == ("tracking", "decfa")

    def test_cst_density_resolves_to_extraction(self):
        stage, label = _resolve_qc_name("sub001_cst_density.png")
        assert (stage, label) == ("extraction", "density")

    def test_cst_over_fa_resolves_to_metrics(self):
        stage, label = _resolve_qc_name("sub001_cst_over_fa.png")
        assert (stage, label) == ("metrics", "cstoverfa")

    def test_new_names_not_misc(self):
        for suffix in ("_dec_fa.png", "_cst_density.png", "_cst_over_fa.png"):
            assert suffix in _QC_NAMES


class TestCliFlags:
    def test_p3_track_has_save_visualizations(self):
        from unittest.mock import patch
        from csttool.cli import main
        with patch('csttool.cli.cmd_track') as mock_cmd:
            with patch('sys.argv', ['csttool', 'track', '--nifti', 'x.nii.gz',
                                    '--out', 'o', '--save-visualizations']):
                main()
        assert mock_cmd.called
        assert mock_cmd.call_args.args[0].save_visualizations is True

    def test_p3_metrics_has_save_visualizations(self):
        from unittest.mock import patch
        from csttool.cli import main
        with patch('csttool.cli.cmd_metrics') as mock_cmd:
            with patch('sys.argv', ['csttool', 'metrics', '--cst-left', 'l.trk',
                                    '--cst-right', 'r.trk', '--out', 'o',
                                    '--save-visualizations']):
                main()
        assert mock_cmd.called
        assert mock_cmd.call_args.args[0].save_visualizations is True

    def test_track_default_is_false(self):
        from unittest.mock import patch
        from csttool.cli import main
        with patch('csttool.cli.cmd_track') as mock_cmd:
            with patch('sys.argv', ['csttool', 'track', '--nifti', 'x.nii.gz',
                                    '--out', 'o']):
                main()
        assert mock_cmd.call_args.args[0].save_visualizations is False


class TestBidsReorgNiftiSurvival:
    """P-1: the new V1 + density NIfTIs must survive the rmtree at run.py:836."""

    def _setup_fake_tree(self, out_root, stem="sub-001"):
        # Minimal stage-dir layout that _write_bids_derivatives consumes.
        trk_scalar = out_root / "tracking" / "scalar_maps"
        ext_scalar = out_root / "extraction" / "scalar_maps"
        trk_viz = out_root / "tracking" / "visualizations"
        ext_viz = out_root / "extraction" / "visualizations"
        met_viz = out_root / "metrics" / "visualizations"
        for d in (trk_scalar, ext_scalar, trk_viz, ext_viz, met_viz):
            d.mkdir(parents=True, exist_ok=True)
        # V1 + sidecar in tracking/scalar_maps
        import nibabel as nib
        import numpy as np
        aff = np.diag([2.0, 2.0, 2.0, 1.0])
        v1 = nib.Nifti1Image(np.zeros((4, 4, 4, 1, 3), np.float32), aff)
        v1.header.set_intent('vector')
        v1_path = trk_scalar / f"{stem}_v1.nii.gz"
        nib.save(v1, v1_path)
        import json
        (v1_path.with_suffix("").with_suffix(".json")).write_text(json.dumps({
            "VectorFrame": "world-RAS", "Description": "V1"}))
        # density + sidecar in extraction/scalar_maps
        dens = nib.Nifti1Image(np.zeros((4, 4, 4), np.float32), aff)
        dens_path = ext_scalar / f"{stem}_cst_density.nii.gz"
        nib.save(dens, dens_path)
        (dens_path.with_suffix("").with_suffix(".json")).write_text(json.dumps({
            "Denominator": 10, "Description": "density"}))
        # The three new PNGs in their stage visualizations/
        for vdir, name in [(trk_viz, "dec_fa"), (ext_viz, "cst_density"),
                           (met_viz, "cst_over_fa")]:
            (vdir / f"{stem}_{name}.png").write_bytes(b"\x89PNG fake")
        return v1_path, dens_path

    def test_v1_and_density_survive_reorg(self, tmp_path):
        from csttool.cli.commands.run import _write_bids_derivatives
        out_root = tmp_path / "work"
        out_root.mkdir()
        stem = "sub-001"
        v1_src, dens_src = self._setup_fake_tree(out_root, stem)
        assert v1_src.exists() and dens_src.exists()

        bids_out = tmp_path / "derivatives"
        import argparse
        args = argparse.Namespace(out=out_root, bids_in=None, raw_bids=None)
        # Minimal scalar-map inputs (FA) so the scalar block runs; others None.
        import nibabel as nib
        import numpy as np
        fa_path = out_root / "tracking" / "scalar_maps" / f"{stem}_fa.nii.gz"
        nib.save(nib.Nifti1Image(np.zeros((4, 4, 4), np.float32),
                                 np.diag([2, 2, 2, 1.0])), fa_path)

        _write_bids_derivatives(
            args, subject_id=stem, session_id=None, bids_out=bids_out,
            step_results={}, tractogram_path=None, fa_path=fa_path,
            md_path=None, rd_path=None, ad_path=None,
            cst_left_path=None, cst_right_path=None, preproc_path=None,
            pipeline_metadata={}, verbose=False,
        )

        # P-1: both new NIfTIs exist under <bids>/sub-001/dwi/.
        dwi = bids_out / "sub-001" / "dwi"
        v1_bids = sorted(dwi.glob("*desc-V1_dwimap.nii.gz"))
        dens_bids = sorted(dwi.glob("*desc-CSTdensity_dwimap.nii.gz"))
        assert v1_bids, "V1 NIfTI did not survive BIDS reorg (rmtree deleted it)"
        assert dens_bids, "density NIfTI did not survive BIDS reorg"
        # Sidecars survived too.
        assert v1_bids[0].with_suffix("").with_suffix(".json").exists()
        assert dens_bids[0].with_suffix("").with_suffix(".json").exists()
        # P-2: the three new PNGs landed under figures/ with the right labels.
        figs = list((bids_out / "sub-001" / "figures").glob("*.png"))
        names = [f.name for f in figs]
        assert any("stage-tracking_qc-decfa" in n for n in names)
        assert any("stage-extraction_qc-density" in n for n in names)
        assert any("stage-metrics_qc-cstoverfa" in n for n in names)
        assert not any("stage-misc" in n for n in names)
        # Stage dirs are gone (rmtree ran).
        assert not (out_root / "tracking").exists()
        assert not (out_root / "extraction").exists()

    def test_p5_sidecars_valid_json_with_required_keys(self, tmp_path):
        from csttool.cli.commands.run import _write_bids_derivatives
        out_root = tmp_path / "w"; out_root.mkdir()
        self._setup_fake_tree(out_root, "sub-002")
        import nibabel as nib
        import numpy as np
        fa_path = out_root / "tracking" / "scalar_maps" / "sub-002_fa.nii.gz"
        nib.save(nib.Nifti1Image(np.zeros((4, 4, 4), np.float32),
                                 np.diag([2, 2, 2, 1.0])), fa_path)
        bids_out = tmp_path / "d"
        import argparse
        args = argparse.Namespace(out=out_root, bids_in=None, raw_bids=None)
        _write_bids_derivatives(
            args, "sub-002", None, bids_out, {}, None, fa_path,
            None, None, None, None, None, None, {}, False)
        import json
        v1_sc = sorted((bids_out / "sub-002" / "dwi").glob("*desc-V1_dwimap.json"))[0]
        d_sc = sorted((bids_out / "sub-002" / "dwi").glob("*desc-CSTdensity_dwimap.json"))[0]
        v1 = json.loads(v1_sc.read_text()); d = json.loads(d_sc.read_text())
        assert v1["VectorFrame"] == "world-RAS"
        assert d["Denominator"] == 10
