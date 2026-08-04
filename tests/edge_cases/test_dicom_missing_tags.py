"""AU31 edge case: DICOM with missing tags.

DICOM stubs with only the magic bytes (no series-level metadata tags) must
not crash the ingest analysis. analyze_series already reads every attribute
defensively via ``getattr(ds, tag, default)``, so missing tags silently
default to ''/0/[] and the series is classified as unsuitable for
tractography. AU31 pins that behaviour and documents it as intentional
(missing tags are a corrupt/incomplete-DICOM edge case; defaulting rather
than crashing is the deliberate choice, surfaced as a warning).
"""

import pytest

from csttool.ingest.modules.analyze_series import analyze_series
from csttool.ingest.modules.scan_study import scan_study


def test_analyze_series_missing_tags_does_not_crash(missing_tag_dicom_dir):
    """A directory of DICOM stubs with no metadata tags must not raise."""
    analysis = analyze_series(missing_tag_dicom_dir, verbose=False)
    # Every defaulted attribute is the empty/zero default, not a crash.
    assert analysis.series_description == ""
    assert analysis.modality == ""
    assert analysis.image_type == []
    assert analysis.uid == ""


def test_missing_tags_classified_unsuitable(missing_tag_dicom_dir):
    """A tag-less series has no diffusion info, so it must be unsuitable."""
    analysis = analyze_series(missing_tag_dicom_dir, verbose=False)
    assert analysis.b_values == []
    assert analysis.suitable_for_tractography is False


def test_scan_study_finds_no_tag_dicoms(missing_tag_dicom_dir):
    """scan_study must still discover the stub DICOMs (magic-byte detection)
    even though they carry no metadata."""
    series = scan_study(missing_tag_dicom_dir)
    assert len(series) == 1
    assert series[0]["n_files"] == 3
