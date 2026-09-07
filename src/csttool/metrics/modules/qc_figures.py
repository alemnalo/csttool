"""
qc_figures.py - the three standalone report QC prototype panels.

These are the **figure composition** layer (plan §4 layer 5): they create
``matplotlib.figure.Figure`` objects, lay panels out on them, draw via the
renderer primitives in ``csttool.viz.render``, save through
``csttool.viz.style.save_figure``, and return paths. They contain no numerical
science: every numerical input (V1, FA, density) is a persisted data product read
from disk, so the figure is reproducible from those products and no
pipeline-stage object is held at render time.

The three slice panels answer three different questions (plan §1):

* DEC-FA          — does the local diffusion direction field support the anatomy?
* CST density     — is the extracted bundle spatially coherent and symmetric?
* final CST over FA — does the extracted CST follow the expected course?

All three share the single deterministic slice from
``csttool.viz.geometry.select_qc_slice`` and a 90 mm x 90 mm prototype canvas at
dpi 200. Report-integration dimensions are deliberately **not** fixed here (M8's
concern); the prototypes are standalone at 90 mm. They are **not** embedded in
the PDF report until scientific review passes (plan §2.11, §9.4).

Four further panels answer the questions the slice panels cannot — whether the
*numbers* are trustworthy rather than whether the picture looks right:

* V1 angle          — are the streamlines following the local principal
  diffusion direction, or being pushed through crossing-fibre voxels?
* profile dispersion — is the mean profile a consensus of the bundle?
* sampling saturation — would a re-run of tracking give the same number?
* node homology      — does node *i* mean the same anatomical level on both sides?

These are plots rather than slices, so they use a landscape canvas and no slice
selection. Their arithmetic lives in ``qc_stats``; this module only composes it,
which is the same separation ``csttool.viz.render`` follows.
"""

import json
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.patches as mpatches
import matplotlib.pyplot as plt
import numpy as np
import nibabel as nib

from csttool.viz import geometry as _geo
from csttool.viz import render
from csttool.viz import style as _style
from csttool.viz.utils import viz_rng

from . import qc_stats
# The region bands and their boundary ticks are drawn by the same two helpers the
# report's profile matrix uses, so every along-tract x-axis in the tool derives
# its geometry from `TRACT_REGIONS` and they cannot drift apart.
from .visualizations import _shade_tract_regions, _label_tract_regions

# Prototype panel size in millimetres (90 mm square) and the saved DPI. The
# report's final dimensions are fixed only in M8 after review.
PROTOTYPE_SIZE_MM = (90.0, 90.0)
PROTOTYPE_DPI = 200

# Plot panels carry an x-axis, a y-axis and a legend rather than an image, so
# they get a landscape canvas instead of the 90 mm square.
PLOT_SIZE_MM = (140.0, 90.0)

# Angles at or below this are read as "the tracker agreed with the tensor".
# Drawn as a reference line on the V1-angle panel; not a pass/fail gate.
V1_ANGLE_REFERENCE_DEG = 20.0


def _figure(size_mm=PROTOTYPE_SIZE_MM):
    w, h = size_mm[0] / 25.4, size_mm[1] / 25.4
    return plt.figure(figsize=(w, h))


def _write_figure_sidecar(png_path, subject_id, panel, slice_index, provenance,
                          extra=None):
    sidecar = Path(png_path).with_suffix(".json")
    content = {
        "Subject": str(subject_id),
        "Panel": panel,
        "SliceIndex": int(slice_index),
        "SliceSelectionRule": provenance.get("rule"),
        "SliceSelectionProvenance": provenance,
    }
    if extra:
        content.update(extra)
    sidecar.write_text(json.dumps(content, indent=2) + "\n")
    return sidecar


def _select_slice(density, density_left, density_right, roi_masks, brain_mask,
                  affine):
    """Run the shared slice selection, tolerating missing optional inputs."""
    return _geo.select_qc_slice(
        density=density, density_left=density_left, density_right=density_right,
        roi_masks=roi_masks, brain_mask=brain_mask, affine=affine, view="coronal",
    )


# ---------------------------------------------------------------------------
# Panel 1: DEC-FA
# ---------------------------------------------------------------------------
def plot_dec_fa_panel(v1_path, fa_path, output_dir, subject_id,
                     *, brain_mask_path=None, roi_masks=None,
                     density=None, density_left=None, density_right=None):
    """DEC-FA panel: world-frame V1 colour-encoded by FA over a coronal slice.

    Reads the stored ``{stem}_v1.nii.gz`` (world RAS+ frame) and the FA map; does
    not re-derive V1. The colour is ``|V1_world| * clip(FA, 0, 1)`` — the
    documented ``dipy.reconst.dti.color_fa`` formula applied after rotation
    (plan §5.1.2, §8.1). No gamma, no percentile stretch: brightness is FA, so a
    dark panel is a real finding.

    Parameters
    ----------
    v1_path : path
        ``{stem}_v1.nii.gz`` with ``NIFTI_INTENT_VECTOR``.
    fa_path : path
        FA NIfTI, same grid.
    brain_mask_path : path, optional
        Brain mask for the always-available slice fallback (level 4). If None,
        a mask derived from FA > 0 is used so the figure still renders.
    roi_masks : list of ndarray, optional
        For the level-3 slice fallback.
    density, density_left, density_right : ndarray, optional
        For the level-1/2 slice selection.

    Returns
    -------
    path to the saved PNG. A JSON sidecar records the slice index and rule.
    """
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    v1_img = nib.load(str(v1_path))
    fa_img = nib.load(str(fa_path))
    affine = fa_img.affine
    v1 = v1_img.get_fdata(dtype=np.float32).squeeze(axis=3) if v1_img.ndim == 5 \
        else v1_img.get_fdata()
    fa = fa_img.get_fdata().astype(np.float32)

    # DEC: |V1_world| * clip(FA, 0, 1). Identical to color_fa(FA, evecs_world)
    # evaluated on the stored world-frame V1 (plan §5.1.2, §8.1).
    dec = np.abs(np.asarray(v1)) * np.clip(fa, 0, 1)[..., None]
    dec = np.clip(dec, 0.0, 1.0).astype(np.float32)

    # Slice selection (brain mask from FA > 0 if none supplied).
    brain_mask = (fa > 0).astype(np.uint8)
    if brain_mask_path is not None:
        brain_mask = nib.load(str(brain_mask_path)).get_fdata().astype(np.uint8)
    idx, prov = _select_slice(density, density_left, density_right, roi_masks,
                              brain_mask, affine)

    fig = _figure()
    ax = fig.add_axes([0.02, 0.02, 0.86, 0.86])
    render.render_rgb_slice(ax, dec, affine, "coronal", idx)
    render.add_direction_legend(ax, loc="lower right")
    ax.set_title("DEC-FA", fontsize=10, fontweight="bold")
    # Caption: the slice rule is mandatory disclosure (plan §6.4).
    fig.text(0.5, 0.005,
             f"coronal slice {idx} · rule: {prov['rule']}",
             ha="center", fontsize=6.5)
    out_path = output_dir / f"{subject_id}_dec_fa.png"
    _style.save_figure(fig, out_path, dpi=PROTOTYPE_DPI)
    plt.close(fig)
    _write_figure_sidecar(out_path, subject_id, "DEC-FA", idx, prov)
    print(f"✓ DEC-FA panel saved: {out_path}")
    return out_path


# ---------------------------------------------------------------------------
# Panel 2: CST density
# ---------------------------------------------------------------------------
def plot_cst_density_panel(density_path, fa_path, output_dir, subject_id,
                           *, brain_mask_path=None, roi_masks=None,
                           density=None, density_left=None, density_right=None):
    """CST density panel: sequential density over grayscale FA, coronal.

    Normalization lives only in the data product (§6.8): the figure sets
    ``vmax`` only — ``vmax = 99th percentile of non-zero density`` (or 1.0 if
    empty) — and prints it in the colorbar label so the scale is never anonymous.
    One shared bilateral scale (left and right on one map by construction, §2.4).
    """
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    dens_img = nib.load(str(density_path))
    fa_img = nib.load(str(fa_path))
    affine = fa_img.affine
    density = dens_img.get_fdata().astype(np.float32) if density is None else np.asarray(density, dtype=np.float32)
    fa = fa_img.get_fdata().astype(np.float32)

    # The denominator (total bilateral streamline count) is recorded in the
    # density sidecar; read it so the caption can state n=... (plan §8.2).
    dens_sidecar = Path(density_path).with_suffix("")
    dens_sidecar = dens_sidecar.parent / (dens_sidecar.name.replace(".nii", "") + ".json")
    n_total = None
    if dens_sidecar.exists():
        try:
            n_total = json.loads(dens_sidecar.read_text()).get("Denominator")
        except Exception:
            n_total = None

    brain_mask = (fa > 0).astype(np.uint8)
    if brain_mask_path is not None:
        brain_mask = nib.load(str(brain_mask_path)).get_fdata().astype(np.uint8)
    idx, prov = _select_slice(density, density_left, density_right, roi_masks,
                              brain_mask, affine)

    # vmax: 99th percentile of non-zero density (or 1.0 if empty). Print it.
    nz = density[density > 0]
    vmax = float(np.percentile(nz, 99)) if nz.size else 1.0

    fig = _figure()
    ax_img = fig.add_axes([0.02, 0.02, 0.74, 0.86])
    ax_cbar = fig.add_axes([0.80, 0.10, 0.04, 0.70])

    render.render_scalar_slice(ax_img, fa, affine, "coronal", idx,
                               cmap=_style.ANATOMY_BG, norm=plt.Normalize(0, 1))
    im = render.render_density_overlay(ax_img, density, affine, "coronal", idx,
                                       cmap=_style.DENSITY_CMAP, vmax=vmax)
    ax_img.set_title("CST density", fontsize=10, fontweight="bold")

    cbar = fig.colorbar(im, cax=ax_cbar)
    cbar.set_label(f"fraction of bundle streamlines (vmax={vmax:.4f})",
                   fontsize=7)
    cbar.ax.tick_params(labelsize=6)

    cap_n = f" · n={n_total}" if n_total is not None else ""
    fig.text(0.5, 0.005,
             f"coronal slice {idx} · rule: {prov['rule']}{cap_n}",
             ha="center", fontsize=6.5)
    out_path = output_dir / f"{subject_id}_cst_density.png"
    _style.save_figure(fig, out_path, dpi=PROTOTYPE_DPI)
    plt.close(fig)
    _write_figure_sidecar(out_path, subject_id, "CST-density", idx, prov,
                          extra={"vmax": vmax})
    print(f"✓ CST density panel saved: {out_path}")
    return out_path


# ---------------------------------------------------------------------------
# Panel 3: final CST over FA
# ---------------------------------------------------------------------------
def plot_cst_over_fa_panel(cst_left_path, cst_right_path, fa_path,
                           output_dir, subject_id, *,
                           brain_mask_path=None, roi_masks=None,
                           density=None, density_left=None, density_right=None,
                           max_streamlines=500, thickness_mm=_geo.DEFAULT_SLAB_MM,
                           seed=None):
    """Final CST over FA: left/right streamlines over grayscale FA, coronal.

    Left = ``style.LEFT`` (#1f77b4), right = ``style.RIGHT`` (#ff7f0e) —
    unchanged from the legacy triptych. FA on a fixed ``Normalize(0, 1)`` so the
    panel is directly comparable to the legacy triptych during migration.
    Streamlines subsampled via ``viz_rng`` (seeded from ``VIZ_SEED`` by default).
    """
    from dipy.io.streamline import load_tractogram

    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    fa_img = nib.load(str(fa_path))
    affine = fa_img.affine
    fa = fa_img.get_fdata().astype(np.float32)

    brain_mask = (fa > 0).astype(np.uint8)
    if brain_mask_path is not None:
        brain_mask = nib.load(str(brain_mask_path)).get_fdata().astype(np.uint8)
    idx, prov = _select_slice(density, density_left, density_right, roi_masks,
                              brain_mask, affine)

    def _load(path):
        try:
            sft = load_tractogram(str(path), 'same')
            return list(sft.streamlines)
        except Exception:
            return []

    left = _load(cst_left_path)
    right = _load(cst_right_path)

    rng = viz_rng(seed)
    fig = _figure()
    # Room reserved below the image for the key. It used to sit inside the
    # panel at "upper right", directly over the anatomy the panel exists to
    # show - and over the superior end of whichever bundle reached that corner.
    ax = fig.add_axes([0.02, 0.13, 0.86, 0.73])
    render.render_scalar_slice(ax, fa, affine, "coronal", idx,
                               cmap=_style.ANATOMY_BG, norm=plt.Normalize(0, 1))
    render.render_streamline_overlay(ax, left, affine, "coronal", idx,
                                     color=_style.LEFT,
                                     thickness_mm=thickness_mm,
                                     max_streamlines=max_streamlines, rng=rng)
    render.render_streamline_overlay(ax, right, affine, "coronal", idx,
                                     color=_style.RIGHT,
                                     thickness_mm=thickness_mm,
                                     max_streamlines=max_streamlines, rng=rng)
    ax.set_title("CST over FA", fontsize=10, fontweight="bold")
    # Hemisphere legend + counts.
    leg = ax.legend(handles=_style.hemisphere_legend_handles(
        left_label=f"Left ({len(left)})", right_label=f"Right ({len(right)})"),
        loc="upper center", bbox_to_anchor=(0.5, -0.02), ncol=2,
        fontsize=6, frameon=False)
    fig.text(0.5, 0.005,
             f"coronal slice {idx} · slab {thickness_mm:.1f} mm · "
             f"rule: {prov['rule']} · drawn L={min(len(left), max_streamlines)} "
             f"R={min(len(right), max_streamlines)}",
             ha="center", fontsize=6.0)
    out_path = output_dir / f"{subject_id}_cst_over_fa.png"
    _style.save_figure(fig, out_path, dpi=PROTOTYPE_DPI)
    plt.close(fig)
    _write_figure_sidecar(out_path, subject_id, "CST-over-FA", idx, prov,
                          extra={"slab_mm": float(thickness_mm),
                                 "streamlines_left": len(left),
                                 "streamlines_right": len(right),
                                 "max_streamlines": int(max_streamlines)})
    print(f"✓ CST-over-FA panel saved: {out_path}")
    return out_path


# ---------------------------------------------------------------------------
# Trust-chain panels: plots, not slices
# ---------------------------------------------------------------------------
def _write_plot_sidecar(png_path, subject_id, panel, question, extra=None):
    """Sidecar for the slice-less plot panels.

    Records the question the panel answers alongside the numbers behind it, so
    the figure's claim is recoverable without re-running the pipeline. The slice
    panels use ``_write_figure_sidecar`` instead; these have no slice to declare.
    """
    sidecar = Path(png_path).with_suffix(".json")
    content = {
        "Subject": str(subject_id),
        "Panel": panel,
        "Question": question,
    }
    if extra:
        content.update(_jsonable(extra))
    sidecar.write_text(json.dumps(content, indent=2) + "\n")
    return sidecar


def _jsonable(value):
    """Recursively convert numpy scalars/arrays so ``json.dumps`` accepts them."""
    if isinstance(value, dict):
        return {k: _jsonable(v) for k, v in value.items()}
    if isinstance(value, (list, tuple)):
        return [_jsonable(v) for v in value]
    if isinstance(value, np.ndarray):
        return _jsonable(value.tolist())
    if isinstance(value, np.integer):
        return int(value)
    if isinstance(value, np.floating):
        return float(value)
    return value


def _load_streamlines(path):
    """Load a tractogram in RASMM, returning [] for a missing or empty file."""
    from dipy.io.streamline import load_tractogram
    if path is None or not Path(path).exists():
        return []
    try:
        return list(load_tractogram(str(path), 'same').streamlines)
    except Exception:
        return []


def _empty_panel(ax, message, clear_ticks=True):
    """Placeholder that states why a panel is blank instead of drawing nothing.

    ``clear_ticks=False`` for an axes that shares its x with a populated one —
    clearing there would strip the ticks from the sibling too.
    """
    ax.text(0.5, 0.5, message, ha="center", va="center", fontsize=8,
            style="italic", transform=ax.transAxes)
    if clear_ticks:
        ax.set_xticks([])
        ax.set_yticks([])


def _draw_band(ax, x, lower, upper, color, alpha, label=None):
    ax.fill_between(x, lower, upper, color=color, alpha=alpha, linewidth=0,
                    label=label)


# ---------------------------------------------------------------------------
# QC-1: tissue plausibility of the sampled voxels
# ---------------------------------------------------------------------------
def plot_tissue_plausibility_panel(fa_path, md_path, density_path, output_dir,
                                   subject_id, *, density_left=None,
                                   density_right=None, bins=qc_stats.TISSUE_BINS):
    """Are the reported scalars coming from voxels that behave like white matter?

    Joint FA-MD histogram over every voxel the CST visits, each voxel weighted
    by its CST density, so the cloud describes the population the published
    means average over. The free-water box (FA below 0.2 *and* MD above
    2.0e-3 mm²/s) is outlined; the fraction of the density mass inside it is the
    number to read, per hemisphere when the per-hemisphere density volumes are
    supplied.

    This is the only panel that questions the *input* to the metrics rather than
    the tractography: a bundle can be anatomically perfect and still report an
    elevated MD because a third of its mass sits in voxels adjacent to the
    ventricles.

    ``density_left`` / ``density_right`` are in-memory arrays because the
    pipeline persists only the bilateral density sum; when they are omitted the
    panel reports the bilateral fraction alone.
    """
    from matplotlib.colors import LogNorm

    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    fa = nib.load(str(fa_path)).get_fdata()
    md = nib.load(str(md_path)).get_fdata()
    density = nib.load(str(density_path)).get_fdata()

    stats = qc_stats.tissue_plausibility(fa, md, density, bins=bins)
    per_hemisphere = {}
    for name, hemisphere_density in (("left", density_left),
                                     ("right", density_right)):
        if hemisphere_density is None:
            continue
        hemisphere = qc_stats.tissue_plausibility(fa, md, hemisphere_density,
                                                  bins=bins)
        per_hemisphere[name] = {
            "n_voxels": hemisphere["n_voxels"],
            "centroid_fa": hemisphere["centroid_fa"],
            "centroid_md": hemisphere["centroid_md"],
            "free_water_fraction": hemisphere["free_water_fraction"],
        }

    fig = _figure(PLOT_SIZE_MM)
    ax = fig.add_subplot(111)

    if stats["n_voxels"] == 0:
        _empty_panel(ax, "no voxel carries CST density")
    else:
        # MD is plotted in ×10⁻³ mm²/s to match the profile figures' axis.
        histogram = stats["histogram"].T          # rows = MD, columns = FA
        md_edges = stats["md_edges"] * 1000.0
        positive = histogram[histogram > 0]
        norm = LogNorm(vmin=max(positive.min(), positive.max() / 1e4),
                       vmax=positive.max())
        mesh = ax.pcolormesh(stats["fa_edges"], md_edges,
                             np.ma.masked_where(histogram <= 0, histogram),
                             cmap=_style.DENSITY_CMAP, norm=norm,
                             shading="flat")
        _style.add_scalar_colorbar(fig, mesh, ax, "CST density mass per bin")

        # Free-water box: the corner no intact white-matter voxel can occupy.
        ax.add_patch(mpatches.Rectangle(
            (0.0, stats["free_water_md_min"] * 1000.0),
            stats["free_water_fa_max"], md_edges[-1] - stats["free_water_md_min"] * 1000.0,
            facecolor="none", edgecolor="#d62728", linewidth=1.0,
            linestyle="--", zorder=3))
        # Label inside the box, at its empty top-left: outside it would land
        # under the legend, and lower down it would sit on the sparse bins.
        ax.text(0.01, md_edges[-1] - 0.03, "free water", fontsize=6.5,
                color="#d62728", ha="left", va="top")

        ax.plot(stats["centroid_fa"], stats["centroid_md"] * 1000.0,
                marker="o", markersize=6, markerfacecolor="white",
                markeredgecolor="black", markeredgewidth=1.2, zorder=4,
                label=f"weighted centroid "
                      f"({stats['centroid_fa']:.2f}, "
                      f"{stats['centroid_md'] * 1000.0:.2f})")
        ax.legend(loc="upper right", fontsize=6.5, framealpha=0.85)

        lines = [f"bilateral free-water mass: "
                 f"{stats['free_water_fraction'] * 100:.2f}%"]
        for name in ("left", "right"):
            if name in per_hemisphere:
                lines.append(f"{name}: "
                             f"{per_hemisphere[name]['free_water_fraction'] * 100:.2f}%")
        # Lower right: the low-MD/high-FA corner is empty for any real bundle,
        # and the upper left is the free-water box this text is describing.
        ax.text(0.98, 0.03, "\n".join(lines), transform=ax.transAxes,
                fontsize=6.5, va="bottom", ha="right",
                bbox=dict(boxstyle="round,pad=0.3", facecolor="white",
                          edgecolor="#999999", alpha=0.85))

    ax.set_xlabel("FA", fontsize=9)
    ax.set_ylabel("MD (×10⁻³ mm²/s)", fontsize=9)
    ax.set_title("Tissue plausibility of the sampled voxels", fontsize=10,
                 fontweight="bold")
    ax.tick_params(labelsize=8)
    fig.text(0.5, 0.005,
             f"every voxel with CST density > 0, weighted by that density · "
             f"n={stats['n_voxels']} voxels · log colour scale",
             ha="center", fontsize=6.0)
    fig.subplots_adjust(bottom=0.16)

    out_path = output_dir / f"{subject_id}_qc_tissue_plausibility.png"
    _style.save_figure(fig, out_path, dpi=PROTOTYPE_DPI)
    plt.close(fig)
    _write_plot_sidecar(
        out_path, subject_id, "tissue-plausibility",
        "Are the reported scalars coming from voxels that behave like white "
        "matter, or from voxels contaminated by CSF partial volume?",
        extra={"bins": int(bins),
               "n_voxels": stats["n_voxels"],
               "density_mass": stats["density_mass"],
               "centroid_fa": stats["centroid_fa"],
               "centroid_md": stats["centroid_md"],
               "fa_percentiles": stats["fa_percentiles"],
               "md_percentiles": stats["md_percentiles"],
               "free_water_fraction": stats["free_water_fraction"],
               "free_water_fa_max": stats["free_water_fa_max"],
               "free_water_md_min": stats["free_water_md_min"],
               "n_md_above_range": stats["n_md_above_range"],
               "per_hemisphere": per_hemisphere})
    print(f"✓ Tissue plausibility panel saved: {out_path}")
    return out_path


# ---------------------------------------------------------------------------
# QC-2: streamline tangent vs local V1
# ---------------------------------------------------------------------------
def plot_v1_angle_panel(cst_left_path, cst_right_path, v1_path,
                        output_dir, subject_id, *, n_points=20):
    """Are the streamlines following the local principal diffusion direction?

    Plots, per hemisphere, the distribution of the acute angle between the
    streamline tangent and the world-frame V1 sampled at each point, as a
    function of position along the tract: median line, interquartile band and
    5-95 band, over the anatomical region bands.

    The angle is the streamline's own account of whether the tensor supported
    the step it took. A bulge in the corona radiata is the CST/CC/SLF crossing
    announcing itself, and it identifies precisely which profile nodes are the
    least trustworthy — which no existing figure reports.

    Reads the stored ``*_desc-V1_dwimap.nii.gz``; the grid and affine come from
    that file, so no FA map is needed. Passing a voxel-frame eigenvector volume
    would silently produce wrong angles on an oblique acquisition, which is why
    the world-frame product exists.
    """
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    v1_img = nib.load(str(v1_path))
    affine = v1_img.affine
    v1 = v1_img.get_fdata(dtype=np.float32)

    left = _load_streamlines(cst_left_path)
    right = _load_streamlines(cst_right_path)

    angles_left, meta_left = qc_stats.tangent_v1_angles(
        left, v1, affine, n_points=n_points)
    angles_right, meta_right = qc_stats.tangent_v1_angles(
        right, v1, affine, n_points=n_points)

    fig = _figure(PLOT_SIZE_MM)
    axes = fig.subplots(2, 1, sharex=True, sharey=True)
    x = np.linspace(0, 100, n_points)

    for ax, angles, meta, color, name in (
            (axes[0], angles_left, meta_left, _style.LEFT, "Left"),
            (axes[1], angles_right, meta_right, _style.RIGHT, "Right")):
        _shade_tract_regions(ax, tick_suffix='%')
        ax.set_ylim(0, 90)
        ax.tick_params(labelsize=7)
        if angles.size == 0:
            # x is shared with the sibling panel, so leave its ticks alone.
            _empty_panel(ax, f"{name}: no streamline sampled the V1 field",
                         clear_ticks=False)
            ax.set_ylabel(f"{name}\nangle (deg)", fontsize=8)
            continue
        _draw_band(ax, x, np.percentile(angles, 5, axis=0),
                   np.percentile(angles, 95, axis=0), color, 0.15)
        _draw_band(ax, x, np.percentile(angles, 25, axis=0),
                   np.percentile(angles, 75, axis=0), color, 0.30)
        ax.plot(x, np.median(angles, axis=0), color=color, linewidth=1.8)
        ax.axhline(V1_ANGLE_REFERENCE_DEG, color="#555555", linestyle=":",
                   linewidth=0.9)
        ax.set_ylabel(f"{name}\nangle (deg)", fontsize=8)
        ax.text(0.99, 0.95,
                f"n={meta['n_contributing']} · median "
                f"{meta['median_angle_deg']:.1f}°",
                transform=ax.transAxes, ha="right", va="top", fontsize=6.5)

    axes[0].set_title("Streamline tangent vs local V1", fontsize=10,
                      fontweight="bold")
    _label_tract_regions(axes[1], y=-0.22)
    # labelpad clears the italic region names, which sit below the tick labels.
    axes[1].set_xlabel("Normalized tract position", fontsize=8, labelpad=16)
    fig.text(0.5, 0.005,
             f"acute angle (|cos θ| — eigenvector sign is arbitrary) · "
             f"median line, IQR and 5–95 bands · dotted line "
             f"{V1_ANGLE_REFERENCE_DEG:.0f}°",
             ha="center", fontsize=6.0)
    fig.subplots_adjust(bottom=0.18, hspace=0.12)

    out_path = output_dir / f"{subject_id}_qc_v1_angle.png"
    _style.save_figure(fig, out_path, dpi=PROTOTYPE_DPI)
    plt.close(fig)
    _write_plot_sidecar(
        out_path, subject_id, "V1-angle",
        "Are the streamlines following the local principal diffusion direction?",
        extra={"n_points": int(n_points),
               "reference_angle_deg": V1_ANGLE_REFERENCE_DEG,
               "left": meta_left, "right": meta_right})
    print(f"✓ V1 angle panel saved: {out_path}")
    return out_path


# ---------------------------------------------------------------------------
# QC-5: dispersion behind the mean profile
# ---------------------------------------------------------------------------
def plot_profile_dispersion_panel(cst_left_path, cst_right_path, scalar_path,
                                  output_dir, subject_id, *, scalar="fa",
                                  n_points=20):
    """Is the mean profile a consensus of the bundle, or an average of dissent?

    The report plots the mean profile as a bare line. This plots the population
    behind it: per-node median, interquartile band and 5-95 band, left and right
    on one axis. Narrow bands mean the mean describes the bundle; bands spanning
    FA 0.3-0.7 at one node mean it summarises a mixture, and a laterality index
    built from two such means compares two mixtures rather than two tracts.

    The contributing-streamline counts printed in the caption are the ones that
    survived profile sampling, not the ones extraction retained — the difference
    is the silent attrition recorded in the sidecar.
    """
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    scalar_img = nib.load(str(scalar_path))
    affine = scalar_img.affine
    scalar_map = scalar_img.get_fdata().astype(np.float32)

    # MD/RD/AD are stored in mm²/s (~8e-4); the axis is labelled ×10⁻³ mm²/s, so
    # scale to match the existing profile figures. FA is dimensionless.
    scale = 1.0 if scalar.lower() == "fa" else 1000.0
    unit = "" if scalar.lower() == "fa" else " (×10⁻³ mm²/s)"

    left = _load_streamlines(cst_left_path)
    right = _load_streamlines(cst_right_path)

    matrix_left, attrition_left = qc_stats.profile_matrix(
        left, scalar_map, affine, n_points=n_points)
    matrix_right, attrition_right = qc_stats.profile_matrix(
        right, scalar_map, affine, n_points=n_points)
    disp_left = qc_stats.profile_dispersion(matrix_left * scale)
    disp_right = qc_stats.profile_dispersion(matrix_right * scale)

    fig = _figure(PLOT_SIZE_MM)
    ax = fig.add_subplot(111)
    x = np.linspace(0, 100, n_points)

    if disp_left["n"] == 0 and disp_right["n"] == 0:
        _empty_panel(ax, "no streamline contributed to either profile")
    else:
        _shade_tract_regions(ax, tick_suffix='%')
        for disp, color, name in ((disp_left, _style.LEFT, "Left"),
                                  (disp_right, _style.RIGHT, "Right")):
            if disp["n"] == 0:
                continue
            _draw_band(ax, x, disp["p5"], disp["p95"], color, 0.12)
            _draw_band(ax, x, disp["p25"], disp["p75"], color, 0.28)
            ax.plot(x, disp["p50"], color=color, linewidth=1.8,
                    label=f"{name} median (n={disp['n']})")
        ax.legend(loc="best", fontsize=7)
        _label_tract_regions(ax, y=-0.13)

    ax.set_ylabel(f"{scalar.upper()}{unit}", fontsize=9)
    # labelpad clears the italic region names, which sit below the tick labels.
    ax.set_xlabel("Normalized tract position", fontsize=9, labelpad=16)
    ax.set_title(f"{scalar.upper()} profile dispersion", fontsize=10,
                 fontweight="bold")
    ax.tick_params(labelsize=8)
    ax.grid(True, alpha=0.25)
    fig.text(0.5, 0.005,
             "median line, interquartile band, 5–95 band across streamlines",
             ha="center", fontsize=6.0)
    fig.subplots_adjust(bottom=0.20)

    out_path = output_dir / f"{subject_id}_qc_profile_dispersion_{scalar.lower()}.png"
    _style.save_figure(fig, out_path, dpi=PROTOTYPE_DPI)
    plt.close(fig)
    _write_plot_sidecar(
        out_path, subject_id, "profile-dispersion",
        "Is the mean profile a consensus of the bundle, or an average over "
        "streamlines that disagree?",
        extra={"scalar": scalar.lower(), "n_points": int(n_points),
               "left_attrition": attrition_left,
               "right_attrition": attrition_right,
               "left_n_contributing": disp_left["n"],
               "right_n_contributing": disp_right["n"]})
    print(f"✓ Profile dispersion panel saved: {out_path}")
    return out_path


# ---------------------------------------------------------------------------
# QC-6: sampling saturation
# ---------------------------------------------------------------------------
def plot_sampling_saturation_panel(cst_left_path, cst_right_path, scalar_path,
                                   output_dir, subject_id, *, scalar="fa",
                                   fractions=qc_stats.DEFAULT_FRACTIONS,
                                   n_repeats=qc_stats.DEFAULT_REPEATS,
                                   seed=None):
    """Would a re-run of tracking give the same number?

    Subsamples each bundle at a grid of fractions and recomputes the headline
    length-unbiased mean, plotting the spread of the estimate against the number
    of streamlines. If the spread has flattened well before the attained N, the
    published mean is converged. If this subject sits on the steep part — the
    common case for a low-yield hemisphere — the mean is a sampling accident and
    the laterality index built from it is noise.

    The caption states the bootstrap standard error of the published mean at the
    full N, which the subsampling curve cannot show (at 100 % it has zero
    variance by construction).
    """
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    scalar_img = nib.load(str(scalar_path))
    affine = scalar_img.affine
    scalar_map = scalar_img.get_fdata().astype(np.float32)
    scale = 1.0 if scalar.lower() == "fa" else 1000.0
    unit = "" if scalar.lower() == "fa" else " (×10⁻³ mm²/s)"

    left = _load_streamlines(cst_left_path)
    right = _load_streamlines(cst_right_path)

    kwargs = dict(fractions=fractions, n_repeats=n_repeats)
    if seed is not None:
        kwargs["seed"] = seed
    stability_left = qc_stats.subsample_stability(left, scalar_map, affine, **kwargs)
    stability_right = qc_stats.subsample_stability(right, scalar_map, affine, **kwargs)

    fig = _figure(PLOT_SIZE_MM)
    ax = fig.add_subplot(111)

    captions = []
    drew = False
    for stability, color, name in ((stability_left, _style.LEFT, "Left"),
                                   (stability_right, _style.RIGHT, "Right")):
        if stability["n_streamlines"] == 0:
            captions.append(f"{name}: empty")
            continue
        drew = True
        counts = np.asarray(stability["counts"], dtype=float)
        ax.fill_between(counts,
                        np.asarray(stability["p5"]) * scale,
                        np.asarray(stability["p95"]) * scale,
                        color=color, alpha=0.20, linewidth=0)
        ax.plot(counts, np.asarray(stability["mean"]) * scale, color=color,
                linewidth=1.6, marker="o", markersize=3,
                label=f"{name} (N={stability['n_streamlines']})")
        ax.axhline(stability["full_estimate"] * scale, color=color,
                   linestyle="--", linewidth=0.8, alpha=0.6)
        boot = stability["bootstrap_full"]["std"] * scale
        captions.append(f"{name} bootstrap SE at full N: {boot:.4g}")

    if not drew:
        _empty_panel(ax, "no streamline yielded an in-bounds sample")
    else:
        ax.set_xscale("log")
        ax.legend(loc="best", fontsize=7)
        ax.grid(True, alpha=0.25, which="both")

    ax.set_xlabel("Streamlines in subsample (log scale)", fontsize=9)
    ax.set_ylabel(f"Mean {scalar.upper()}{unit}", fontsize=9)
    ax.set_title(f"Sampling saturation of mean {scalar.upper()}", fontsize=10,
                 fontweight="bold")
    ax.tick_params(labelsize=8)
    fig.text(0.5, 0.005,
             f"mean ± 5–95 band over {n_repeats} draws without replacement · "
             + " · ".join(captions),
             ha="center", fontsize=6.0)
    fig.subplots_adjust(bottom=0.18)

    out_path = output_dir / f"{subject_id}_qc_sampling_saturation.png"
    _style.save_figure(fig, out_path, dpi=PROTOTYPE_DPI)
    plt.close(fig)
    _write_plot_sidecar(
        out_path, subject_id, "sampling-saturation",
        "Is the headline mean converged with respect to the number of "
        "streamlines, i.e. would a re-run give the same number?",
        extra={"scalar": scalar.lower(), "n_repeats": int(n_repeats),
               "seed": qc_stats.DEFAULT_SEED if seed is None else seed,
               "left": {k: v for k, v in stability_left.items()
                        if k != "estimates"},
               "right": {k: v for k, v in stability_right.items()
                         if k != "estimates"}})
    print(f"✓ Sampling saturation panel saved: {out_path}")
    return out_path


# ---------------------------------------------------------------------------
# QC-7: silent attrition during profile construction
# ---------------------------------------------------------------------------
def plot_profile_attrition_panel(cst_left_path, cst_right_path, scalar_path,
                                 output_dir, subject_id, *, scalar="fa",
                                 n_points=20):
    """How much of the bundle was silently discarded when the profile was built?

    `compute_tract_profile` drops streamlines that are too short and points that
    fall outside the scalar grid without saying so. This audits the pipeline
    against itself: the four gates a streamline passes on the way to the
    published profile, per hemisphere, plus the per-point retention.

    The failure it exists to catch is *asymmetric* attrition. If 30 % of the
    right bundle never reached the profile, every right-hemisphere regional
    metric is computed on a different population than the left, and every
    regional laterality index is comparing unlike with unlike — which no other
    panel in the set would reveal.

    On a well-formed run the answer is "none": the streamlines were generated on
    the same grid the scalar map lives on, so the bars are equal by
    construction. The panel therefore states its verdict in words as well as in
    bars, so a flat funnel reads as a pass rather than as a broken figure.
    """
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    scalar_img = nib.load(str(scalar_path))
    affine = scalar_img.affine
    scalar_map = scalar_img.get_fdata().astype(np.float32)

    left = _load_streamlines(cst_left_path)
    right = _load_streamlines(cst_right_path)

    _, attrition_left = qc_stats.profile_matrix(left, scalar_map, affine,
                                                n_points=n_points)
    _, attrition_right = qc_stats.profile_matrix(right, scalar_map, affine,
                                                 n_points=n_points)
    funnel_left = qc_stats.attrition_funnel(attrition_left)
    funnel_right = qc_stats.attrition_funnel(attrition_right)

    fig = _figure(PLOT_SIZE_MM)
    ax_funnel, ax_points = fig.subplots(
        1, 2, gridspec_kw={"width_ratios": [3, 1]})

    labels = [stage["label"] for stage in funnel_left["stages"]]
    positions = np.arange(len(labels))[::-1]   # first gate at the top
    height = 0.36

    if funnel_left["n_input"] == 0 and funnel_right["n_input"] == 0:
        _empty_panel(ax_funnel, "both bundles are empty")
    else:
        for funnel, offset, color, name in (
                (funnel_left, +height / 2, _style.LEFT, "Left"),
                (funnel_right, -height / 2, _style.RIGHT, "Right")):
            counts = [stage["count"] for stage in funnel["stages"]]
            ax_funnel.barh(positions + offset, counts, height=height,
                           color=color, alpha=0.85,
                           label=f"{name} (n={funnel['n_input']})")
            for y, count in zip(positions + offset, counts):
                ax_funnel.text(count, y, f" {count}", va="center", ha="left",
                               fontsize=6.5)
        ax_funnel.set_yticks(positions)
        ax_funnel.set_yticklabels(labels, fontsize=7)
        # Headroom for the count annotations, which sit outside the bars, and a
        # clear band below the last group for the legend to sit in.
        ax_funnel.set_xlim(0, max(funnel_left["n_input"],
                                  funnel_right["n_input"]) * 1.18)
        ax_funnel.set_ylim(-1.15, len(labels) - 0.4)
        ax_funnel.legend(loc="lower right", fontsize=6.5, framealpha=1.0)

    ax_funnel.set_xlabel("Streamlines surviving the gate", fontsize=8)
    ax_funnel.set_title("Profile attrition audit", fontsize=10,
                        fontweight="bold")
    ax_funnel.tick_params(labelsize=7)
    ax_funnel.grid(True, axis="x", alpha=0.25)

    retentions = [(funnel_left, _style.LEFT, "L"), (funnel_right, _style.RIGHT, "R")]
    drawn = [(x, color, name) for x, color, name in retentions
             if x["point_retention"] is not None]
    if drawn:
        ax_points.bar([name for _, _, name in drawn],
                      [funnel["point_retention"] * 100 for funnel, _, _ in drawn],
                      color=[color for _, color, _ in drawn], alpha=0.85,
                      width=0.6)
        for index, (funnel, _, _) in enumerate(drawn):
            ax_points.text(index, funnel["point_retention"] * 100 + 1.5,
                           f"{funnel['point_retention'] * 100:.1f}%",
                           ha="center", fontsize=6.5)
        ax_points.set_ylim(0, 112)
        ax_points.axhline(100, color="#555555", linestyle=":", linewidth=0.9)
    else:
        _empty_panel(ax_points, "no point sampled")

    ax_points.set_ylabel("Points inside the scalar grid (%)", fontsize=8)
    ax_points.set_title("Per-point retention", fontsize=8)
    ax_points.tick_params(labelsize=7)

    if funnel_left["any_attrition"] or funnel_right["any_attrition"]:
        verdict = " · ".join(
            f"{name}: {funnel['n_contributing']}/{funnel['n_input']} streamlines, "
            f"{(funnel['point_retention'] or 0) * 100:.1f}% of points"
            for funnel, name in ((funnel_left, "Left"), (funnel_right, "Right")))
        verdict = f"ATTRITION — {verdict}"
    else:
        verdict = ("no attrition: every extracted streamline and every sampled "
                   "point reached the profile")
    fig.text(0.5, 0.005, verdict, ha="center", fontsize=6.5)
    fig.subplots_adjust(bottom=0.18, wspace=0.45)

    out_path = output_dir / f"{subject_id}_qc_profile_attrition.png"
    _style.save_figure(fig, out_path, dpi=PROTOTYPE_DPI)
    plt.close(fig)
    _write_plot_sidecar(
        out_path, subject_id, "profile-attrition",
        "How much of the bundle was silently discarded when the profile was "
        "built?",
        extra={"scalar": scalar.lower(), "n_points": int(n_points),
               "min_profile_points": qc_stats.MIN_PROFILE_POINTS,
               "left": funnel_left, "right": funnel_right})
    print(f"✓ Profile attrition panel saved: {out_path}")
    return out_path


# ---------------------------------------------------------------------------
# QC-8: node homology between hemispheres
# ---------------------------------------------------------------------------
def plot_node_homology_panel(cst_left_path, cst_right_path, output_dir,
                             subject_id, *, n_points=20):
    """Does node *i* mean the same anatomical level on both sides?

    The 20-node parameterisation is relative — node ``i`` is ``i/(n-1)`` of the
    way along whatever was reconstructed. Top axis: the mean world Z of each node
    for each hemisphere, with the divergence between them shaded. Bottom axis:
    the two streamline-length distributions.

    If the left bundle is systematically shorter, node 7 sits at a different
    anatomical height on each side, and every one of the twelve regional
    laterality indices in ``bilateral_metrics.json`` is confounded by geometry
    rather than microstructure. Nothing else in the figure set checks this.
    """
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    left = _load_streamlines(cst_left_path)
    right = _load_streamlines(cst_right_path)
    comparison = qc_stats.compare_node_geometry(left, right, n_points=n_points)
    geom_left, geom_right = comparison["left"], comparison["right"]

    fig = _figure(PLOT_SIZE_MM)
    ax_z, ax_len = fig.subplots(2, 1, gridspec_kw={"height_ratios": [2, 1]})
    nodes = np.arange(n_points)

    if geom_left["node_z"] is None and geom_right["node_z"] is None:
        _empty_panel(ax_z, "both bundles are empty")
    else:
        for geom, color, name in ((geom_left, _style.LEFT, "Left"),
                                  (geom_right, _style.RIGHT, "Right")):
            if geom["node_z"] is None:
                continue
            ax_z.plot(nodes, geom["node_z"], color=color, linewidth=1.6,
                      marker="o", markersize=3,
                      label=f"{name} (n={geom['n_streamlines']}, "
                            f"mean length {geom['length_mean']:.0f} mm)")
            ax_z.fill_between(nodes, geom["node_z"] - geom["node_z_std"],
                              geom["node_z"] + geom["node_z_std"],
                              color=color, alpha=0.15, linewidth=0)
        if comparison["z_difference_mm"] is not None:
            ax_z.fill_between(nodes, geom_left["node_z"], geom_right["node_z"],
                              color="#555555", alpha=0.12, linewidth=0,
                              label="L–R divergence")
        ax_z.legend(loc="best", fontsize=6.5)

    # Nodes are integers; the default locator would tick them at 2.5.
    ax_z.set_xticks(np.arange(0, n_points, max(1, n_points // 10)))
    ax_z.set_ylabel("Mean world Z (mm, superior +)", fontsize=8)
    ax_z.set_title("Node homology between hemispheres", fontsize=10,
                   fontweight="bold")
    ax_z.tick_params(labelsize=7)
    ax_z.grid(True, alpha=0.25)

    lengths = [g["lengths_mm"] for g in (geom_left, geom_right)
               if g["lengths_mm"].size]
    if lengths:
        bins = np.histogram_bin_edges(np.concatenate(lengths), bins=25)
        for geom, color, name in ((geom_left, _style.LEFT, "Left"),
                                  (geom_right, _style.RIGHT, "Right")):
            if not geom["lengths_mm"].size:
                continue
            ax_len.hist(geom["lengths_mm"], bins=bins, color=color, alpha=0.5,
                        label=name)
        ax_len.legend(loc="best", fontsize=6.5)
    else:
        _empty_panel(ax_len, "no streamline lengths")

    ax_len.set_xlabel("Streamline length (mm)", fontsize=8)
    ax_len.set_ylabel("Count", fontsize=8)
    ax_len.tick_params(labelsize=7)
    ax_z.set_xlabel(f"Profile node (0 = inferior, {n_points - 1} = superior)",
                    fontsize=8)

    max_z = comparison["max_abs_z_difference_mm"]
    verdict = ("node positions not comparable (empty bundle)" if max_z is None
               else f"max |L–R| node offset: {max_z:.1f} mm · "
                    f"mean length difference: "
                    f"{comparison['length_difference_mm']:+.1f} mm")
    fig.text(0.5, 0.005, verdict, ha="center", fontsize=6.5)
    fig.subplots_adjust(bottom=0.16, hspace=0.45)

    out_path = output_dir / f"{subject_id}_qc_node_homology.png"
    _style.save_figure(fig, out_path, dpi=PROTOTYPE_DPI)
    plt.close(fig)
    _write_plot_sidecar(
        out_path, subject_id, "node-homology",
        "Does profile node i mean the same anatomical level in both hemispheres?",
        extra={"n_points": int(n_points),
               "max_abs_z_difference_mm": max_z,
               "length_difference_mm": comparison["length_difference_mm"],
               "z_difference_mm": comparison["z_difference_mm"],
               "left_length_mean": geom_left["length_mean"],
               "right_length_mean": geom_right["length_mean"],
               "left_n_streamlines": geom_left["n_streamlines"],
               "right_n_streamlines": geom_right["n_streamlines"]})
    print(f"✓ Node homology panel saved: {out_path}")
    return out_path
