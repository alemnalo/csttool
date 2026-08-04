"""Unit tests for the centralized report formatting helpers.

These are the single source of truth for precision, units, and sign convention
used in both the global and regional tables (promoted out of the closures that
used to live inside save_html_report).
"""

import pytest

from csttool.metrics.modules.reports import (
    format_mean_sd,
    format_med_range,
    format_li,
    format_localized,
    li_class,
    _build_global_metrics,
    _build_regional_metrics,
)


class TestFormatMeanSD:
    def test_fa_dimensionless(self):
        assert format_mean_sd(0.487, 0.080) == "0.487 ± 0.080"

    def test_diffusivity_scaled(self):
        # md ~ 8e-4 mm^2/s -> 0.80 x10^-3
        assert format_mean_sd(8.0e-4, 6.0e-5, is_diffusivity=True) == "0.80 ± 0.06"

    def test_fa_three_decimals(self):
        assert format_mean_sd(0.5, 0.1).count(".") == 2  # "0.500 ± 0.100"


class TestFormatMedRange:
    def test_fa(self):
        assert format_med_range(0.487, 0.312, 0.721) == "0.487 (0.312-0.721)"

    def test_diffusivity(self):
        assert (
            format_med_range(7.9e-4, 6.1e-4, 9.5e-4, is_diffusivity=True)
            == "0.79 (0.61-0.95)"
        )


class TestFormatLI:
    def test_positive_signed(self):
        assert format_li(0.043) == "+0.043"

    def test_negative_keeps_minus(self):
        assert format_li(-0.052) == "-0.052"

    def test_zero(self):
        assert format_li(0.0) == "0.000"

    def test_rounds_to_three(self):
        assert format_li(0.0436) == "+0.044"


class TestLIClass:
    def test_positive_is_left(self):
        assert li_class(0.04) == "li-left"

    def test_negative_is_right(self):
        assert li_class(-0.04) == "li-right"

    def test_zero(self):
        assert li_class(0.0) == "li-zero"


class TestFormatLocalized:
    def _asym(self, val):
        return {"fa_plic": {"laterality_index": val}}

    def test_fa_three_decimals(self):
        left = {"fa": {"plic": 0.487}}
        right = {"fa": {"plic": 0.491}}
        assert (
            format_localized(left, right, self._asym(0.004), "fa", "plic")
            == "0.487 / 0.491 / +0.004"
        )

    def test_diffusivity_two_decimals(self):
        left = {"md": {"plic": 7.9e-4}}
        right = {"md": {"plic": 8.1e-4}}
        asym = {"md_plic": {"laterality_index": -0.013}}
        assert (
            format_localized(left, right, asym, "md", "plic")
            == "0.79 / 0.81 / -0.013"
        )

    def test_absent_scalar(self):
        assert format_localized({}, {}, {}, "fa", "plic") == "-"


def _comparison_with_median():
    left = {
        "morphology": {
            "n_streamlines": 100, "mean_length": 85.0, "median_length": 84.0,
            "std_length": 12.0, "min_length": 40.0, "max_length": 130.0,
            "tract_volume": 12000.0,
        },
        "fa": {"mean": 0.487, "std": 0.080, "median": 0.487, "min": 0.31, "max": 0.72,
               "n_streamlines": 100},
    }
    right = {
        "morphology": {
            "n_streamlines": 110, "mean_length": 88.0, "median_length": 87.0,
            "std_length": 11.0, "min_length": 42.0, "max_length": 135.0,
            "tract_volume": 13000.0,
        },
        "fa": {"mean": 0.491, "std": 0.080, "median": 0.491, "min": 0.32, "max": 0.73,
               "n_streamlines": 110},
    }
    asym = {
        "streamline_count": {"laterality_index": -0.05},
        "volume": {"laterality_index": -0.04},
        "mean_length": {"laterality_index": -0.02},
        "fa": {"laterality_index": -0.004},
    }
    return {"left": left, "right": right, "asymmetry": asym}


class TestLengthRowMedian:
    """median_length is a genuine statistic; a mean is never shown as a median."""

    def test_length_row_shows_median_when_present(self):
        rows = _build_global_metrics(
            _comparison_with_median()["left"],
            _comparison_with_median()["right"],
            _comparison_with_median()["asymmetry"],
        )
        length_row = next(r for r in rows if r["label"] == "Length (mm)")
        # median (min-max) of the genuine median (1 decimal, mm), not the mean.
        assert length_row["left_med_range"] == "84.0 (40.0-130.0)"
        assert length_row["right_med_range"] == "87.0 (42.0-135.0)"

    def test_length_row_dash_when_median_absent(self):
        comp = _comparison_with_median()
        # Legacy morphology without median_length.
        for side in ("left", "right"):
            comp[side]["morphology"].pop("median_length", None)
        rows = _build_global_metrics(comp["left"], comp["right"], comp["asymmetry"])
        length_row = next(r for r in rows if r["label"] == "Length (mm)")
        # Em dash, never the mean dressed as a median.
        assert length_row["left_med_range"] == "—"
        assert length_row["right_med_range"] == "—"

    def test_streamlines_and_volume_carry_no_median(self):
        rows = _build_global_metrics(
            _comparison_with_median()["left"],
            _comparison_with_median()["right"],
            _comparison_with_median()["asymmetry"],
        )
        stream = next(r for r in rows if r["label"] == "Streamlines")
        vol = next(r for r in rows if r["label"] == "Volume (cm³)")
        assert stream["left_med_range"] == "—"
        assert vol["left_med_range"] == "—"

    def test_li_class_assigned_by_sign(self):
        rows = _build_global_metrics(
            _comparison_with_median()["left"],
            _comparison_with_median()["right"],
            _comparison_with_median()["asymmetry"],
        )
        fa_row = next(r for r in rows if r["label"] == "FA")
        # asymmetry.fa.laterality_index == -0.004 -> li-right (orange).
        assert fa_row["li_class"] == "li-right"
        assert fa_row["li"] == "-0.004"
