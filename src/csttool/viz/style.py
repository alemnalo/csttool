"""
style.py - csttool shared visualization style.

Single source of truth for typography, figure defaults, canonical colours/colormaps,
the savefig policy, and small presentation-only helpers (colorbars).

This module holds **presentation policy only** - no coordinate/geometry logic
(that lives in ``viz.geometry``) and no pipeline logic.

Usage
-----
    from csttool.viz import style
    style.apply_house_style()          # once, before creating figures
    ...
    style.save_figure(fig, path)       # uniform savefig policy
"""

import matplotlib as mpl
import matplotlib.pyplot as plt

# ---------------------------------------------------------------------------
# Canonical colours
# ---------------------------------------------------------------------------
# Hemisphere colours - used for Left/Right EVERYWHERE (streamlines, bars, profiles).
LEFT = "#1f77b4"    # blue
RIGHT = "#ff7f0e"   # orange

# ROI overlay colours (categorical).
MOTOR_LEFT = LEFT
MOTOR_RIGHT = RIGHT
BRAINSTEM = "#2ca02c"   # green

# Validation overlays (candidate vs reference) - a DIFFERENT axis of meaning than
# L/R, so deliberately distinct hues to avoid colliding with hemisphere colours.
CANDIDATE = "#d62728"   # red
REFERENCE = "#9467bd"   # purple

# Neutral colour for whole-brain streamline projections (carries no L/R meaning).
NEUTRAL_STREAMLINE = "#555555"

# ---------------------------------------------------------------------------
# Colormaps
# ---------------------------------------------------------------------------
# Continuous quantitative scalar maps (FA/MD): perceptually reasonable, colorbar-friendly.
FA_CMAP = "gray"          # FA shown as anatomy-like background
MD_CMAP = "inferno"       # MD magnitude (sequential, perceptually uniform)
# Residual / |difference| maps (denoising, unringing): sequential + colorbar.
RESIDUAL_CMAP = "magma"
# Signed deformation (Jacobian): diverging around 1.0.
JACOBIAN_CMAP = "RdBu_r"

ANATOMY_BG = "gray"       # grayscale anatomical backgrounds (b0, FA-as-background)

# ---------------------------------------------------------------------------
# rcParams (house style)
# ---------------------------------------------------------------------------
RCPARAMS = {
    "figure.dpi": 150,
    "savefig.dpi": 200,
    "savefig.bbox": "tight",
    "font.size": 12,
    "axes.titlesize": 13,
    "axes.labelsize": 12,
    "xtick.labelsize": 11,
    "ytick.labelsize": 11,
    "legend.fontsize": 11,
    "axes.spines.top": False,
    "axes.spines.right": False,
}

# Savefig policy applied by ``save_figure`` (kept explicit so image figures that call
# ``fig.savefig`` directly stay consistent even if rcParams are not applied).
SAVEFIG_DPI = 200
SAVEFIG_FACECOLOR = "white"


def apply_house_style():
    """Apply the shared rcParams to the current matplotlib session (idempotent)."""
    mpl.rcParams.update(RCPARAMS)


def save_figure(fig, path, **kwargs):
    """Save a figure with the uniform csttool policy (dpi=200, tight, white bg).

    Parameters
    ----------
    fig : matplotlib.figure.Figure
    path : str or pathlib.Path
    **kwargs : forwarded to ``Figure.savefig`` (override policy if needed).
    """
    params = dict(dpi=SAVEFIG_DPI, bbox_inches="tight", facecolor=SAVEFIG_FACECOLOR)
    params.update(kwargs)
    fig.savefig(path, **params)
    return path


def add_scalar_colorbar(fig, mappable, ax, label, orientation="vertical", **kwargs):
    """Attach a labelled colorbar to a continuous scalar-map image.

    Use for quantitative maps whose values are interpretable (FA, MD, Jacobian,
    residual/difference magnitude). Do NOT use for binary masks or categorical
    ROI overlays.

    Parameters
    ----------
    fig : matplotlib.figure.Figure
    mappable : the object returned by ``imshow`` / ``pcolormesh``.
    ax : Axes or list of Axes the colorbar should steal space from.
    label : str, colorbar label (include units where meaningful).
    orientation : "vertical" | "horizontal".
    """
    cbar_kw = dict(fraction=0.046, pad=0.04)
    cbar_kw.update(kwargs)
    cbar = fig.colorbar(mappable, ax=ax, orientation=orientation, **cbar_kw)
    cbar.set_label(label)
    return cbar


def hemisphere_legend_handles(left_label="Left", right_label="Right"):
    """Return Line2D handles for a consistent Left/Right streamline legend."""
    return [
        plt.Line2D([0], [0], color=LEFT, linewidth=2, label=left_label),
        plt.Line2D([0], [0], color=RIGHT, linewidth=2, label=right_label),
    ]
