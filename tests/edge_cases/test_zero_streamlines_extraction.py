"""AU31 edge case: zero-streamlines input to extraction.

A whole-brain tractogram that is empty (or fully filtered out) must produce a
well-formed empty result, not a ZeroDivisionError or IndexError. Verified
against the running code: extract_cst_passthrough already guards
extraction_rate with ``if len(streamlines) > 0``; AU31 pins that behaviour
and the all-length-filtered case so a regression is caught.
"""

import numpy as np
import pytest
from dipy.tracking.streamline import Streamlines

from csttool.extract.modules.passthrough_filtering import extract_cst_passthrough


def _empty_masks():
    return {
        "brainstem": np.zeros((6, 6, 6), dtype=bool),
        "motor_left": np.zeros((6, 6, 6), dtype=bool),
        "motor_right": np.zeros((6, 6, 6), dtype=bool),
    }


def test_empty_tractogram_returns_well_formed_zero_result():
    """Empty input -> zero counts, extraction_rate 0, all dict keys present."""
    result = extract_cst_passthrough(
        [], _empty_masks(), np.eye(4), verbose=False
    )
    assert isinstance(result, dict)
    assert set(result.keys()) >= {"cst_left", "cst_right", "cst_combined", "stats"}
    assert result["stats"]["cst_left_count"] == 0
    assert result["stats"]["cst_right_count"] == 0
    assert result["stats"]["cst_total_count"] == 0
    assert result["stats"]["extraction_rate"] == 0
    assert len(result["cst_left"]) == 0


def test_all_length_filtered_out_returns_well_formed_zero_result():
    """A non-empty tractogram fully removed by the length filter must not crash
    in the hemisphere-distribution diagnostic (which iterates the filtered set)."""
    short = [np.array([[0, 0, 0], [1, 1, 1], [2, 2, 2]], dtype=float)]
    result = extract_cst_passthrough(
        short, _empty_masks(), np.eye(4),
        min_length=30.0, max_length=200.0, verbose=False,
    )
    assert result["stats"]["after_length_filter"] == 0
    assert result["stats"]["cst_total_count"] == 0
    assert result["stats"]["extraction_rate"] == 0.0
