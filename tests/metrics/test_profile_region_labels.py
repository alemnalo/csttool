"""
The profile figure's anatomical labels must describe the regions the report tabulates.

`plot_tract_profiles` and `plot_stacked_profiles` drew 'Pontine Level' / 'PLIC' /
'Precentral Gyrus' at 0 / 50 / 100% of tract position, annotating the axis ticks. Read as
point landmarks that is defensible — the tract does start at the pons and end at the
precentral gyrus, and each label did land inside the region it names. But the report uses
those same three names for *bin averages* over 0-35 / 35-70 / 70-100%, so the figure and
the table said "pontine" while meaning a point and a range respectively. Two of the three
labels sat at the extreme edge of their range.

The labels are now centred in their regions and the bands are shaded, so the figure states
which stretch of tract each tabulated number covers. This is a legibility fix, not a
correctness one: no label previously pointed at the wrong region.

These tests deliberately do not read `TRACT_REGIONS` to assert positions, which would be
tautological now that the figure derives its labels from it. Instead they recover the bins
empirically by probing `compute_localized_metrics` with one-hot profiles, so re-hardcoding
a position fails the tests even if the constants stay put.
"""

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import pytest

from csttool.metrics.modules.unilateral_analysis import compute_localized_metrics
from csttool.metrics.modules.visualizations import (
    _REGION_DISPLAY_NAMES,
    _label_tract_regions,
)

PROBE_N = 200


def _region_at_percent(pct):
    """Which region does compute_localized_metrics average this tract position into?

    Recovered by black-box probe rather than read from the region constants, so this stays
    an independent check on where the reported numbers actually come from.
    """
    index = min(int(pct / 100.0 * PROBE_N), PROBE_N - 1)
    profile = [0.0] * PROBE_N
    profile[index] = 1.0

    responding = [name for name, value in compute_localized_metrics(profile).items() if value > 0]
    assert len(responding) == 1, f"position {pct}% falls in {responding}, expected exactly one region"
    return responding[0]


def _region_extent(region):
    """Empirically recover (first, last) whole-percent positions binned into `region`."""
    inside = [p for p in range(101) if _region_at_percent(p) == region]
    assert inside, f"no tract position maps to '{region}'"
    return inside[0], inside[-1]


def _drawn_labels(wrap=False):
    """Return {display text: x position} for the labels the figure actually draws."""
    fig, ax = plt.subplots()
    try:
        _label_tract_regions(ax, wrap=wrap)
        return {t.get_text(): t.get_position()[0] for t in ax.texts}
    finally:
        plt.close(fig)


def test_probe_recovers_three_distinct_regions():
    """Guard: the probe must actually discriminate, or every test below is vacuous."""
    found = {_region_at_percent(p) for p in range(0, 101, 5)}

    assert found == set(_REGION_DISPLAY_NAMES), f"probe recovered {found}"


@pytest.mark.parametrize("region,display", sorted(_REGION_DISPLAY_NAMES.items()))
def test_label_sits_over_the_region_it_names(region, display):
    """Each label falls within the stretch of tract whose value is reported under that name.

    This held before the labels were re-centred too. It is kept because it is the property
    that actually matters: it fails if the bin bounds are ever changed without moving the
    labels to match.
    """
    labels = _drawn_labels()

    assert display in labels, f"'{display}' is not drawn; labels are {sorted(labels)}"
    assert _region_at_percent(labels[display]) == region, (
        f"'{display}' is drawn at {labels[display]}%, which the report bins as "
        f"'{_region_at_percent(labels[display])}', not '{region}'"
    )


@pytest.mark.parametrize("region,display", sorted(_REGION_DISPLAY_NAMES.items()))
def test_label_is_centred_in_its_region_not_at_an_edge(region, display):
    """Labels mark the middle of a range, not a landmark at the tract's start or end.

    This is what changed: 'Pontine Level' sat at 0% and 'Precentral Gyrus' at 100%, the
    extreme edges of ranges covering 0-35% and 70-100%.
    """
    first, last = _region_extent(region)
    x = _drawn_labels()[display]

    assert abs(x - (first + last) / 2.0) <= 1.0, (
        f"'{display}' is drawn at {x}%, not centred in its {first}-{last}% region"
    )


def test_wrapped_labels_are_positioned_identically():
    """The stacked plot wraps label text onto two lines; that must not move it."""
    plain, wrapped = _drawn_labels(wrap=False), _drawn_labels(wrap=True)

    for text, x in plain.items():
        assert wrapped[text.replace(' ', '\n')] == x
