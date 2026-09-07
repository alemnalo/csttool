"""Tests that the registration template is identified, recorded and displayed.

Which template served as the moving image is a licensing fact, not a tuning
detail: the pipeline silently falls back from FSL-licensed FMRIB58_FA to the
permissively licensed bundled MNI152. These tests pin that the choice is
resolved once, carried into the report JSON, and named on the QC figure --
so a published figure can always be attributed from the run artifacts alone.
"""

import json
from unittest.mock import patch

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pytest

from csttool.extract.modules.endpoint_filtering import save_extraction_report
from csttool.extract.modules.registration import describe_registration_template
from csttool.extract.modules import visualizations as viz


class TestDescribeRegistrationTemplate:
    def test_fmrib58_carries_fsl_license_and_source(self):
        info = describe_registration_template(
            "fmrib58_fa", path="/data/FMRIB58_FA_1mm.nii.gz"
        )
        assert info["name"] == "FMRIB58_FA"
        assert info["modality"] == "FA"
        assert info["tier"] == "user-fetched"
        assert info["license"] == "FSL non-commercial"
        assert "fmrib.ox.ac.uk" in info["source_url"]
        assert info["path"] == "/data/FMRIB58_FA_1mm.nii.gz"
        assert info["fallback_reason"] is None

    def test_bundled_mni152_carries_icbm_license(self):
        info = describe_registration_template("mni152_t1")
        assert info["name"] == "MNI152_T1_1mm"
        assert info["modality"] == "T1"
        assert info["tier"] == "bundled"
        assert info["license"] == "BSD-like (ICBM)"
        assert "mcgill" in info["source_url"].lower()

    def test_the_two_templates_are_distinguishable(self):
        """The whole point: the report must not conflate them."""
        fa = describe_registration_template("fmrib58_fa")
        t1 = describe_registration_template("mni152_t1")
        assert fa["license"] != t1["license"]
        assert fa["display_name"] != t1["display_name"]
        assert fa["sha256"] != t1["sha256"]

    def test_user_supplied_template_has_no_manifest_claims(self):
        info = describe_registration_template("user_supplied", path="/tmp/mine.nii.gz")
        assert info["tier"] == "user-supplied"
        assert info["license"] is None
        assert info["sha256"] is None
        assert info["path"] == "/tmp/mine.nii.gz"

    def test_fallback_reason_is_preserved(self):
        info = describe_registration_template(
            "mni152_t1", fallback_reason="FMRIB58_FA not installed"
        )
        assert info["fallback_reason"] == "FMRIB58_FA not installed"

    def test_descriptor_is_json_serialisable(self):
        for kind in ("fmrib58_fa", "mni152_t1", "user_supplied"):
            json.dumps(describe_registration_template(kind))

    def test_unknown_kind_raises(self):
        with pytest.raises(KeyError):
            describe_registration_template("not_a_template")


class TestExtractionReportRecordsTemplate:
    @staticmethod
    def _cst_result():
        return {"stats": {"n_left": 2, "n_right": 3}}

    @staticmethod
    def _paths():
        return {"cst_left": "l.trk", "cst_right": "r.trk", "cst_combined": "c.trk"}

    def _read(self, path):
        return json.loads(path.read_text())

    def test_template_is_recorded_when_registration_passed(self, tmp_path):
        tpl = describe_registration_template("fmrib58_fa", path="/data/fa.nii.gz")
        out = save_extraction_report(
            self._cst_result(), self._paths(), tmp_path, "sub-01",
            registration={"template": tpl},
        )
        report = self._read(out)
        assert report["registration"]["template"]["name"] == "FMRIB58_FA"
        assert report["registration"]["template"]["license"] == "FSL non-commercial"

    def test_fallback_is_recorded_distinctly(self, tmp_path):
        tpl = describe_registration_template(
            "mni152_t1", fallback_reason="FMRIB58_FA not installed"
        )
        out = save_extraction_report(
            self._cst_result(), self._paths(), tmp_path, "sub-02",
            registration={"template": tpl},
        )
        template = self._read(out)["registration"]["template"]
        assert template["name"] == "MNI152_T1_1mm"
        assert template["fallback_reason"] == "FMRIB58_FA not installed"

    def test_report_omits_registration_when_not_supplied(self, tmp_path):
        """Backward compatibility: callers that pass nothing get the old shape."""
        out = save_extraction_report(
            self._cst_result(), self._paths(), tmp_path, "sub-03"
        )
        assert "registration" not in self._read(out)

    def test_report_omits_registration_when_template_missing(self, tmp_path):
        out = save_extraction_report(
            self._cst_result(), self._paths(), tmp_path, "sub-04",
            registration={"jacobian_stats": {}},
        )
        assert "registration" not in self._read(out)


class TestRegistrationQcNamesTheTemplate:
    """The QC panel used to hardcode 'MNI template' whatever was registered."""

    @staticmethod
    def _volume():
        vol = np.zeros((16, 16, 16), dtype=np.float32)
        vol[4:12, 4:12, 4:12] = 1.0
        return vol

    def _titles_for(self, tmp_path, **kwargs):
        """Capture the figure at save time and read back its text artists."""
        captured = {}

        def _capture(fig, path, *a, **kw):
            captured["titles"] = [ax.get_title() for ax in fig.axes]
            captured["suptitle"] = fig._suptitle.get_text() if fig._suptitle else ""
            captured["legend"] = [
                t.get_text() for lg in fig.legends for t in lg.get_texts()
            ]
            return path

        affine = np.eye(4)
        affine[:3, 3] = [-8, -8, -8]
        with patch.object(viz._style, "save_figure", side_effect=_capture):
            viz.plot_registration_comparison(
                self._volume(), self._volume(), tmp_path,
                subject_id="sub-01", affine=affine, verbose=False, **kwargs
            )
        plt.close("all")
        return captured

    def test_named_template_appears_in_panel_title(self, tmp_path):
        got = self._titles_for(tmp_path, template_label="FMRIB58 FA template")
        assert "FMRIB58 FA template (warped)" in got["titles"]
        assert "MNI template (warped)" not in got["titles"]

    def test_named_template_appears_in_suptitle_and_legend(self, tmp_path):
        got = self._titles_for(tmp_path, template_label="FMRIB58 FA template")
        assert "FMRIB58 FA template → Subject Space" in got["suptitle"]
        assert any("Warped FMRIB58 FA template" in t for t in got["legend"])

    def test_default_label_is_generic_for_callers_that_do_not_know(self, tmp_path):
        got = self._titles_for(tmp_path)
        assert "MNI template (warped)" in got["titles"]
