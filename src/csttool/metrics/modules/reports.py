"""
report.py

Report generation functions for CST metrics.

This module generates:
- JSON reports (complete machine-readable data)
- CSV summaries (metrics table for Excel/R analysis)
- PDF clinical reports (human-readable summary with visualizations)
"""

import json
import csv
import base64
from pathlib import Path
from datetime import datetime
import numpy as np

from jinja2 import Environment, FileSystemLoader

# Import version from package
from csttool import __version__

# Module-level template environment
_TEMPLATE_DIR = Path(__file__).parent / "templates"
_jinja_env = Environment(loader=FileSystemLoader(_TEMPLATE_DIR))


def _embed_image(path):
    """Embed image as base64 data URI.
    
    Parameters
    ----------
    path : str or Path or None
        Path to image file
        
    Returns
    -------
    str or None
        Base64 data URI or None if path invalid
    """
    if path is None:
        return None
    path = Path(path)
    if not path.exists():
        return None
    try:
        with open(path, 'rb') as f:
            data = base64.b64encode(f.read()).decode('utf-8')
        ext = path.suffix.lower()
        mime = 'image/png' if ext == '.png' else 'image/jpeg'
        return f'data:{mime};base64,{data}'
    except Exception:
        return None


# The reproducibility footer reports a *fixed* dependency subset rather than
# whatever happened to be importable, so the footer has a deterministic height
# and two reports are directly comparable line by line. Display names are the
# projects' own capitalisation.
_FOOTER_DEPENDENCIES = (
    ('numpy', 'NumPy'),
    ('scipy', 'SciPy'),
    ('dipy', 'DIPY'),
    ('nibabel', 'NiBabel'),
    ('matplotlib', 'Matplotlib'),
)

# Thread-limit environment variables, shortened for the footer summary.
_THREAD_VARS = (
    ('OMP_NUM_THREADS', 'OMP'),
    ('MKL_NUM_THREADS', 'MKL'),
    ('OPENBLAS_NUM_THREADS', 'OPENBLAS'),
    ('NUMEXPR_NUM_THREADS', 'NUMEXPR'),
    ('VECLIB_MAXIMUM_THREADS', 'VECLIB'),
    ('DIPY_NUM_THREADS', 'DIPY'),
)


def _truncate(text, limit):
    """Shorten a string to ``limit`` characters with an ellipsis character."""
    text = str(text)
    return text if len(text) <= limit else text[: limit - 1].rstrip() + '…'


def _short_python_version(raw):
    """``"3.12.3 (main, ...) [GCC 13.3.0]"`` -> ``"3.12.3"``."""
    if not raw:
        return 'N/A'
    return str(raw).split()[0]


def _short_platform(raw):
    """``"Linux-6.17.0-40-generic-x86_64-with-glibc2.39"`` -> ``"Linux 6.17.0-40-generic (x86_64)"``.

    ``platform.platform()`` returns one long hyphenated token. The OS name,
    kernel release and machine are the parts a reader needs; the libc suffix is
    noise in a one-line footer.
    """
    if not raw:
        return 'N/A'
    text = str(raw)
    head, sep, _ = text.partition('-with-')
    parts = (head if sep else text).split('-')
    if len(parts) >= 3:
        system, machine = parts[0], parts[-1]
        release = '-'.join(parts[1:-1])
        return _truncate(f"{system} {release} ({machine})", 40)
    return _truncate(text, 40)


def _build_report_provenance(raw_provenance):
    """Build a filtered provenance view for the human-readable report.

    The full provenance dict (including git commit and raw command line) is
    preserved in the JSON report. This function returns only the fields
    relevant to a scientific reader: hardware, threading environment, a fixed
    dependency subset, Python version, and platform — each pre-formatted as one
    short line so the footer is fixed-height.

    Parameters
    ----------
    raw_provenance : dict
        Full provenance dict from ``get_provenance_dict()`` (may be empty).

    Returns
    -------
    dict
        Filtered provenance with keys ``python_version``, ``platform``,
        ``machine``, ``hardware_str``, ``thread_env_str``, ``dependencies_str``.
        All values are strings; None / missing keys produce "N/A" placeholders.
    """
    if not raw_provenance:
        return {}

    hardware = raw_provenance.get('hardware', {}) or {}
    cpu_model = hardware.get('cpu_model') or 'N/A'
    cpu_count = hardware.get('cpu_count')
    ram = hardware.get('total_ram_gb')
    gpu = hardware.get('gpu')

    # Long CPU strings are truncated horizontally; the footer must not wrap.
    hardware_parts = [_truncate(cpu_model, 44)]
    if cpu_count is not None:
        hardware_parts.append(f"{cpu_count} logical cores")
    if ram is not None:
        hardware_parts.append(f"{ram:.1f} GB RAM")
    if gpu:
        gpu_str = ', '.join(gpu) if isinstance(gpu, list) else str(gpu)
        hardware_parts.append(_truncate(gpu_str, 30))

    # Summarised, not one line per raw variable: "Thread limits: unset" when the
    # process inherited no limits, otherwise only the variables actually set.
    thread_env = raw_provenance.get('thread_env', {}) or {}
    set_vars = [
        f"{short}={thread_env[var]}"
        for var, short in _THREAD_VARS
        if thread_env.get(var) is not None
    ]
    if not thread_env:
        thread_str = 'Thread limits: not recorded'
    elif not set_vars:
        thread_str = 'Thread limits: unset'
    else:
        unset = len(thread_env) - len(set_vars)
        thread_str = 'Threads: ' + ', '.join(set_vars)
        if unset:
            thread_str += ' (others unset)'

    deps = raw_provenance.get('dependencies', {}) or {}
    dep_parts = [
        f"{display} {deps.get(key, 'N/A')}" for key, display in _FOOTER_DEPENDENCIES
    ]

    return {
        'python_version': _short_python_version(raw_provenance.get('python_version')),
        'platform': _short_platform(raw_provenance.get('platform')),
        'machine': raw_provenance.get('machine') or 'N/A',
        'hardware_str': ' · '.join(hardware_parts) if hardware_parts else 'N/A',
        'thread_env_str': thread_str,
        'dependencies_str': 'Dependencies: ' + ' · '.join(dep_parts),
    }


# ---------------------------------------------------------------------------
# Metric formatting helpers (module-level, unit-tested).
#
# Single source of truth for the precision, units, and sign conventions used in
# both the global and regional tables. Promoted out of the closures that used
# to live inside save_html_report so the rules can be tested independently and
# cannot drift between the two tables.
# ---------------------------------------------------------------------------

# Scalars stored in SI units (mm^2/s, ~1e-3) and displayed scaled by 1000 as
# "x10^-3 mm^2/s". FA is dimensionless.
_DIFFUSIVITY_SCALARS = ('md', 'rd', 'ad')


def format_mean_sd(mean, std, is_diffusivity=False):
    """Format a mean +/- SD string.

    Diffusivity values are scaled by 1000 to match the report's ``x10^-3 mm^2/s``
    axis convention; FA is dimensionless (3 decimals).
    """
    if is_diffusivity:
        return f"{mean * 1000:.2f} ± {std * 1000:.2f}"
    return f"{mean:.3f} ± {std:.3f}"


def format_med_range(median, min_val, max_val, is_diffusivity=False, decimals=3):
    """Format a ``median (min-max)`` string.

    Diffusivity values are scaled by 1000 (2 decimals). FA uses 3 decimals.
    Length (mm) is non-diffusivity but reported at 1 decimal, so the caller
    passes ``decimals=1``.
    """
    if is_diffusivity:
        return f"{median * 1000:.2f} ({min_val * 1000:.2f}-{max_val * 1000:.2f})"
    return (
        f"{median:.{decimals}f} "
        f"({min_val:.{decimals}f}-{max_val:.{decimals}f})"
    )


def format_li(value):
    """Format a laterality index with an explicit sign, 3 decimals.

    Positive (left-dominant) values carry a leading ``+``; negative values keep
    their ``-``; zero renders as ``0.000``. One rule for both tables.
    """
    if value > 0:
        return f"+{value:.3f}"
    return f"{value:.3f}"


def format_localized(left, right, asym, scalar, region):
    """Format a regional ``L / R / LI`` cell string.

    FA uses 3 decimals; diffusivities use 2 decimals scaled by 1000. LI is
    signed via :func:`format_li`. Returns ``"-"`` when the scalar/region is
    absent.
    """
    if scalar not in left or region not in left[scalar]:
        return "-"
    l_val = left[scalar][region]
    r_val = right[scalar][region]
    li_key = f"{scalar}_{region}"
    li_val = asym.get(li_key, {}).get("laterality_index", 0.0)
    if scalar == "fa":
        return f"{l_val:.3f} / {r_val:.3f} / {format_li(li_val)}"
    return f"{l_val * 1000:.2f} / {r_val * 1000:.2f} / {format_li(li_val)}"


def li_class(value):
    """CSS class for an LI cell, by sign (blue/orange/neutral)."""
    if value > 0:
        return "li-left"
    if value < 0:
        return "li-right"
    return "li-zero"


def _scalar_med_range_block(side_metrics, scalar):
    """Build a ``median (min-max)`` cell for a scalar, with legacy fallback.

    Legacy data without ``median``/``min``/``max`` (older reports) falls back to
    a mean +/- 3*SD envelope rather than crashing, preserving the existing
    backward-compatible behaviour. A mean is never displayed as a median.
    """
    block = side_metrics[scalar]
    median = block.get("median", block["mean"])
    min_val = block.get("min", max(0.0, block["mean"] - 3 * block["std"]))
    max_val = block.get("max", min(1.0, block["mean"] + 3 * block["std"]))
    is_diff = scalar in _DIFFUSIVITY_SCALARS
    return format_med_range(median, min_val, max_val, is_diffusivity=is_diff)


def _build_global_metrics(left, right, asym):
    """Build the global metrics table rows (6-column).

    Streamlines and Volume carry no per-streamline distribution in this report,
    so their median columns render an explicit em dash. Length uses the genuine
    ``median_length`` when present; legacy morphology without it renders an em
    dash rather than substituting the mean (a mean is never shown as a median).
    """
    rows = []

    li_sc = asym["streamline_count"]["laterality_index"]
    rows.append({
        "label": "Streamlines",
        "left_mean_sd": str(left["morphology"]["n_streamlines"]),
        "left_med_range": "—",
        "right_mean_sd": str(right["morphology"]["n_streamlines"]),
        "right_med_range": "—",
        "li": format_li(li_sc),
        "li_class": li_class(li_sc),
    })

    li_vol = asym["volume"]["laterality_index"]
    rows.append({
        "label": "Volume (cm³)",
        "left_mean_sd": f"{left['morphology']['tract_volume'] / 1000.0:.2f}",
        "left_med_range": "—",
        "right_mean_sd": f"{right['morphology']['tract_volume'] / 1000.0:.2f}",
        "right_med_range": "—",
        "li": format_li(li_vol),
        "li_class": li_class(li_vol),
    })

    lm = left["morphology"]
    rm = right["morphology"]
    li_len = asym["mean_length"]["laterality_index"]
    # Length: genuine median when present (1 decimal, mm); em dash for legacy
    # data (never the mean dressed as a median).
    if "median_length" in lm and "median_length" in rm:
        left_len_med = format_med_range(
            lm["median_length"], lm["min_length"], lm["max_length"], decimals=1
        )
        right_len_med = format_med_range(
            rm["median_length"], rm["min_length"], rm["max_length"], decimals=1
        )
    else:
        left_len_med = "—"
        right_len_med = "—"
    rows.append({
        "label": "Length (mm)",
        "left_mean_sd": f"{lm['mean_length']:.1f} ± {lm['std_length']:.1f}",
        "left_med_range": left_len_med,
        "right_mean_sd": f"{rm['mean_length']:.1f} ± {rm['std_length']:.1f}",
        "right_med_range": right_len_med,
        "li": format_li(li_len),
        "li_class": li_class(li_len),
    })

    for scalar in ("fa", "md", "rd", "ad"):
        if scalar not in left or scalar not in right:
            continue
        is_diff = scalar in _DIFFUSIVITY_SCALARS
        li_val = asym[scalar]["laterality_index"]
        rows.append({
            "label": "FA" if scalar == "fa" else f"{scalar.upper()} (×10⁻³)",
            "left_mean_sd": format_mean_sd(
                left[scalar]["mean"], left[scalar]["std"], is_diffusivity=is_diff
            ),
            "left_med_range": _scalar_med_range_block(left, scalar),
            "right_mean_sd": format_mean_sd(
                right[scalar]["mean"], right[scalar]["std"], is_diffusivity=is_diff
            ),
            "right_med_range": _scalar_med_range_block(right, scalar),
            "li": format_li(li_val),
            "li_class": li_class(li_val),
        })

    return rows


def _build_regional_metrics(left, right, asym):
    """Build the regional metrics table rows (Pontine / PLIC / Precentral)."""
    rows = []
    for name, key in (("Pontine", "pontine"), ("PLIC", "plic"), ("Precentral", "precentral")):
        rows.append({
            "name": name,
            "fa": format_localized(left, right, asym, "fa", key),
            "md": format_localized(left, right, asym, "md", key),
            "rd": format_localized(left, right, asym, "rd", key),
            "ad": format_localized(left, right, asym, "ad", key),
        })
    return rows


def _build_methods_band(acquisition, processing, space, orientation_code, version):
    """Build the three compact methods-band columns as (label, value) row lists.

    Terminology is verified against the implementation: DTI tensor model
    (``fit_tensors.py``), CSA ODF direction model (``estimate_directions.py``),
    deterministic LocalTracking (``run_tractography.py``), and FA-mask seeding
    (``seed_and_stop.py``). "Anatomically constrained" is deliberately not used.
    """
    bvals = acquisition.get("b_values") if acquisition else None
    max_b = f"{bvals[-1]} s/mm²" if bvals else "N/A"
    res = acquisition.get("resolution_mm") if acquisition else None
    if res:
        res_str = f"{res[0]:.1f}×{res[1]:.1f}×{res[2]:.1f} mm"
    else:
        res_str = "N/A"
    fs = acquisition.get("field_strength_T") if acquisition else None
    fs_str = f"{fs:.1f} T" if fs else "N/A"
    te = acquisition.get("echo_time_ms") if acquisition else None
    te_str = f"{te:.1f} ms" if te else "N/A"
    ndir = acquisition.get("n_directions", "N/A") if acquisition else "N/A"
    acq_rows = [
        ("Max b-value", max_b),
        ("Directions", str(ndir)),
        ("Resolution", res_str),
        ("Field strength", fs_str),
        ("Echo time", te_str),
    ]

    tparams = (processing or {}).get("tracking_params", {}) or {}
    fit_method = tparams.get("fit_method", "WLS")
    sh_order = tparams.get("sh_order")
    dir_model = f"CSA ODF, SH {sh_order}" if sh_order is not None else "CSA ODF"
    seed_density = tparams.get("seed_density")
    seeding = f"FA mask, ×{seed_density}" if seed_density is not None else "FA mask"
    fa_thresh = tparams.get("fa_thresh")
    fa_str = str(fa_thresh) if fa_thresh is not None else "N/A"
    step_size = tparams.get("step_size")
    step_str = f"{step_size} mm" if step_size is not None else "N/A"
    pre = (processing or {}).get("preprocessing", {}) or {}
    motion = "Yes" if pre.get("motion_correction") else "No"
    proc_rows = [
        ("Scalar model", f"DTI ({fit_method})"),
        ("Direction model", dir_model),
        ("Tracking", "Deterministic"),
        ("Seeding", seeding),
        ("FA threshold", fa_str),
        ("Step size", step_str),
        ("Motion correction", motion),
    ]

    space_short = "Native" if (not space or "native" in space.lower()) else space
    extraction = (processing or {}).get("extraction", {}) or {}
    ex_method = extraction.get("method", "N/A")
    ai = extraction.get("artifact_index")
    if ai is not None:
        ai_str = f"{ai:.2f}"
    elif extraction and extraction.get("artifact_index_available") is False:
        # The diagnostic is only defined for bidirectional extraction. Naming the
        # method that was actually used says exactly that, in one cell, without
        # wrapping a sentence across the band. "Not applicable" is preserved as a
        # distinct meaning from a missing value (which renders a bare "N/A").
        ai_str = f"N/A for {ex_method} extraction"
    else:
        ai_str = "N/A"
    space_rows = [
        ("Space", space_short),
        ("Orientation", orientation_code or "N/A"),
        ("csttool", f"v{version}"),
        ("Extraction", ex_method),
        ("Artifact index", ai_str),
    ]

    return acq_rows, proc_rows, space_rows


def build_node_homology_view(node_homology):
    """Pre-format the node-homology status line's numbers for the template.

    The template does no arithmetic and applies no rule: it prints two numbers
    and a static sentence. There is deliberately **no** threshold, flag, colour
    or classification here — two subjects cannot establish one, and a
    ``homology_flag`` in the schema would immediately acquire downstream
    consumers depending on an unvalidated rule.

    Returns ``{'available': bool, 'max_abs_z_mm': str, 'length_diff_mm': str}``.
    ``available`` is False when either hemisphere is empty, in which case the
    template states that plainly and omits the explanatory sentence.
    """
    if not node_homology or node_homology.get("max_abs_z_difference_mm") is None:
        return {"available": False, "max_abs_z_mm": "—", "length_diff_mm": "—"}
    length_diff = node_homology.get("length_difference_mm")
    return {
        "available": True,
        "max_abs_z_mm": f"{node_homology['max_abs_z_difference_mm']:.1f}",
        # Signed: which side is longer is the whole content of the number.
        "length_diff_mm": "—" if length_diff is None else f"{length_diff:+.1f}",
    }


def build_report_context(
    comparison,
    visualization_paths,
    subject_id,
    version,
    space,
    metadata,
    fa_affine,
):
    """Assemble the full Jinja render context in one testable function.

    Parameters
    ----------
    fa_affine : ndarray or None
        FA-map affine, used to compute the orientation code dynamically. When
        None, the orientation renders as ``N/A``.
    """
    from csttool.viz.geometry import orientation_code as _orientation_code

    left = comparison["left"]
    right = comparison["right"]
    asym = comparison["asymmetry"]

    orient = _orientation_code(fa_affine) if fa_affine is not None else None
    acquisition = (metadata or {}).get("acquisition", {}) or {}
    processing = (metadata or {}).get("processing", {}) or {}
    acq_rows, proc_rows, space_rows = _build_methods_band(
        acquisition, processing, space, orient, version
    )

    # The 1x4 strip replaces the 1x3 triptych. The legacy key is still read so a
    # caller that has not migrated (or a replay of an older run's paths) still
    # renders a QC figure rather than a placeholder.
    qc_path = (visualization_paths.get("qc_strip")
               or visualization_paths.get("tractogram_qc_triptych"))

    return {
        "subject_id": subject_id,
        "date": datetime.now().strftime("%Y-%m-%d"),
        "version": version,
        "method_summary": "Deterministic CST tractography",
        "acq_rows": acq_rows,
        "proc_rows": proc_rows,
        "space_rows": space_rows,
        "metrics": _build_global_metrics(left, right, asym),
        "localized_metrics": _build_regional_metrics(left, right, asym),
        "node_homology": build_node_homology_view(comparison.get("node_homology")),
        "profile_matrix": _embed_image(visualization_paths.get("profile_matrix")),
        "qc_strip": _embed_image(qc_path),
        "provenance": _build_report_provenance((metadata or {}).get("provenance", {})),
    }


# The scope statement serialised beside every bootstrap SE. It is the sentence
# that stops the SE from being read as pipeline reproducibility, so it is
# mandatory and its wording is fixed.
UNCERTAINTY_SCOPE = (
    "Conditional on the retained bundle. Quantifies how much the reported mean "
    "would move if a different subset of THESE streamlines had been sampled. "
    "Does NOT include tracking, seeding, registration, preprocessing or "
    "acquisition variability; a re-run of tractography would produce a "
    "different bundle, not a resample of this one."
)


def build_uncertainty_block():
    """The ``metrics.uncertainty`` block describing every ``*_se`` in the JSON.

    Written once per report rather than repeated per value: the method, the
    resample count and the seed are identical for every SE, and the scope
    sentence needs to be read once.
    """
    from csttool.reproducibility.context import DEFAULT_SEED
    from .qc_stats import REPORT_BOOTSTRAP_REPEATS

    return {
        "method": "nonparametric bootstrap of the per-streamline mean",
        "n_resamples": REPORT_BOOTSTRAP_REPEATS,
        "seed": DEFAULT_SEED,
        "scope": UNCERTAINTY_SCOPE,
    }


def _se_cell(value):
    """A CSV cell for a standard error: the float, or an empty string.

    Never ``0.0`` for a missing SE — a zero standard error is a scientific
    claim (the mean is exactly determined) and must not be fabricated by a
    serialiser.
    """
    return "" if value is None else value


def save_json_report(comparison, output_dir, subject_id, metadata=None):
    """
    Save comprehensive metrics report as JSON.
    
    Parameters
    ----------
    comparison : dict
        Output from compare_bilateral_cst()
    output_dir : str or Path
        Output directory
    subject_id : str
        Subject identifier
    metadata : dict, optional
        Additional metadata including:
        - acquisition: dict with protocol, b_values, n_directions, resolution
        - processing: dict with denoising_method, tracking_method, etc.
        - qc_thresholds: dict with fa_threshold, min_length, max_length
        - provenance: dict from get_provenance_dict() (git, deps, hardware, etc.)
        
    Returns
    -------
    json_path : Path
        Path to saved JSON file
    """
    
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # Initialize metadata if not provided
    if metadata is None:
        metadata = {}
    
    # Build report with extended schema. The uncertainty block is attached to the
    # serialised copy rather than mutating the caller's comparison dict, which is
    # also handed to the figure and HTML paths.
    metrics = dict(comparison)
    metrics['uncertainty'] = build_uncertainty_block()

    report = {
        'subject_id': subject_id,
        'processing_date': datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
        'csttool_version': __version__,
        'acquisition': metadata.get('acquisition', {}),
        'processing': metadata.get('processing', {}),
        'qc_thresholds': metadata.get('qc_thresholds', {}),
        'provenance': metadata.get('provenance', {}),
        'metrics': metrics
    }
    
    # Save JSON
    json_path = output_dir / f"{subject_id}_bilateral_metrics.json"
    with open(json_path, 'w') as f:
        json.dump(report, f, indent=2)
    
    print(f"  ✓ JSON report saved: {json_path}")
    return json_path


def save_csv_summary(comparison, output_dir, subject_id):
    """
    Save metrics summary as CSV table.
    
    Creates a flat CSV file with key metrics suitable for group analysis.
    
    Parameters
    ----------
    comparison : dict
        Output from compare_bilateral_cst()
    output_dir : str or Path
        Output directory
    subject_id : str
        Subject identifier
        
    Returns
    -------
    csv_path : Path
        Path to saved CSV file
    """
    
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    
    left = comparison['left']
    right = comparison['right']
    asym = comparison['asymmetry']
    
    # Prepare data row
    data = {
        'subject_id': subject_id,
        'processing_date': datetime.now().strftime("%Y-%m-%d"),
        
        # Left morphology
        'left_n_streamlines': left['morphology']['n_streamlines'],
        'left_mean_length_mm': left['morphology']['mean_length'],
        'left_median_length_mm': left['morphology'].get('median_length', 0.0),
        'left_tract_volume_mm3': left['morphology']['tract_volume'],

        # Right morphology
        'right_n_streamlines': right['morphology']['n_streamlines'],
        'right_mean_length_mm': right['morphology']['mean_length'],
        'right_median_length_mm': right['morphology'].get('median_length', 0.0),
        'right_tract_volume_mm3': right['morphology']['tract_volume'],
        
        # Asymmetry
        'volume_laterality_index': asym['volume']['laterality_index'],
        'streamline_count_laterality_index': asym['streamline_count']['laterality_index'],
    }
    
    # Add FA if available
    if 'fa' in left:
        data.update({
            'left_fa_mean': left['fa']['mean'],
            'left_fa_std': left['fa']['std'],
            'right_fa_mean': right['fa']['mean'],
            'right_fa_std': right['fa']['std'],
            'fa_laterality_index': asym['fa']['laterality_index'],
            # Length-biased counterpart, preserved for bias auditing (AU10):
            'left_fa_mean_point_weighted': left['fa'].get('mean_point_weighted', 0.0),
            'right_fa_mean_point_weighted': right['fa'].get('mean_point_weighted', 0.0),
        })
    
    # Add MD if available
    if 'md' in left:
        data.update({
            'left_md_mean': left['md']['mean'],
            'left_md_std': left['md']['std'],
            'right_md_mean': right['md']['mean'],
            'right_md_std': right['md']['std'],
            'md_laterality_index': asym['md']['laterality_index'],
            'left_md_mean_point_weighted': left['md'].get('mean_point_weighted', 0.0),
            'right_md_mean_point_weighted': right['md'].get('mean_point_weighted', 0.0),
        })
    
    # Add RD if available
    if 'rd' in left:
        data.update({
            'left_rd_mean': left['rd']['mean'],
            'left_rd_std': left['rd']['std'],
            'right_rd_mean': right['rd']['mean'],
            'right_rd_std': right['rd']['std'],
            'rd_laterality_index': asym['rd']['laterality_index'],
            'left_rd_mean_point_weighted': left['rd'].get('mean_point_weighted', 0.0),
            'right_rd_mean_point_weighted': right['rd'].get('mean_point_weighted', 0.0),
        })
    
    # Add AD if available
    if 'ad' in left:
        data.update({
            'left_ad_mean': left['ad']['mean'],
            'left_ad_std': left['ad']['std'],
            'right_ad_mean': right['ad']['mean'],
            'right_ad_std': right['ad']['std'],
            'ad_laterality_index': asym['ad']['laterality_index'],
            'left_ad_mean_point_weighted': left['ad'].get('mean_point_weighted', 0.0),
            'right_ad_mean_point_weighted': right['ad'].get('mean_point_weighted', 0.0),
        })

    # Add localized metrics (pontine, plic, precentral) for each scalar
    regions = ['pontine', 'plic', 'precentral']
    scalars = ['fa', 'md', 'rd', 'ad']

    for scalar in scalars:
        if scalar in left and 'pontine' in left[scalar]:
            for region in regions:
                data[f'left_{scalar}_{region}'] = left[scalar].get(region, 0.0)
                data[f'right_{scalar}_{region}'] = right[scalar].get(region, 0.0)
                # Add LI for localized metric
                li_key = f'{scalar}_{region}'
                if li_key in asym:
                    data[f'{scalar}_{region}_laterality_index'] = asym[li_key]['laterality_index']

    # Bootstrap standard errors, appended at the end so a reader that indexes by
    # column position is unaffected. Regional SEs are deliberately not exported:
    # 24 more columns for a value the report does not display. A missing SE is an
    # empty cell, never 0.0 (see _se_cell).
    data['left_mean_length_se'] = _se_cell(left['morphology'].get('bootstrap_se_length'))
    data['right_mean_length_se'] = _se_cell(right['morphology'].get('bootstrap_se_length'))
    for scalar in scalars:
        if scalar in left and scalar in right:
            data[f'left_{scalar}_bootstrap_se'] = _se_cell(left[scalar].get('bootstrap_se'))
            data[f'right_{scalar}_bootstrap_se'] = _se_cell(right[scalar].get('bootstrap_se'))
            data[f'{scalar}_laterality_index_se'] = _se_cell(
                asym.get(scalar, {}).get('laterality_index_se')
            )

    # Node homology: the two headline scalars only. The 20-element per-node
    # difference arrays stay in the JSON — they have no place in a flat table.
    homology = comparison.get('node_homology') or {}
    data['node_homology_max_abs_z_diff_mm'] = _se_cell(
        homology.get('max_abs_z_difference_mm')
    )
    data['node_homology_length_diff_mm'] = _se_cell(
        homology.get('length_difference_mm')
    )

    # Save CSV
    csv_path = output_dir / f"{subject_id}_metrics_summary.csv"
    with open(csv_path, 'w', newline='') as f:
        writer = csv.DictWriter(f, fieldnames=data.keys())
        writer.writeheader()
        writer.writerow(data)
    
    print(f"  ✓ CSV summary saved: {csv_path}")
    return csv_path


def save_html_report(
    comparison,
    visualization_paths,
    output_dir,
    subject_id,
    version=None,
    space="Native Space",
    metadata=None,
    fa_affine=None,
):
    """
    Generate HTML report using Jinja2 template.
    
    Parameters
    ----------
    comparison : dict
        Output from compare_bilateral_cst()
    visualization_paths : dict
        Paths to generated visualizations. Expected keys: ``profile_matrix``
        (the 2x2 along-tract figure) and ``tractogram_qc_triptych`` (the 1x3
        QC composite). Both are optional and render as placeholders if absent.
    output_dir : str or Path
        Output directory
    subject_id : str
        Subject identifier
    version : str, optional
        csttool version (defaults to package version)
    space : str
        Space declaration (e.g., "Native Space")
    metadata : dict, optional
        Acquisition, processing, and provenance metadata
    fa_affine : ndarray, optional
        FA-map affine used to compute the orientation code dynamically. When
        None, orientation renders as ``N/A``.
        
    Returns
    -------
    html_path : Path
        Path to saved HTML file
    """
    if version is None:
        version = __version__

    if metadata is None:
        metadata = {}

    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    context = build_report_context(
        comparison=comparison,
        visualization_paths=visualization_paths,
        subject_id=subject_id,
        version=version,
        space=space,
        metadata=metadata,
        fa_affine=fa_affine,
    )

    # Load template and CSS
    template = _jinja_env.get_template("report.html.j2")
    css = (_TEMPLATE_DIR / "report.css").read_text()

    # Render template with context
    html_content = template.render(css=css, **context)

    html_path = output_dir / f"{subject_id}_report.html"
    html_path.write_text(html_content, encoding="utf-8")

    print(f"  ✓ HTML report saved: {html_path}")
    return html_path

def html_to_pdf(html_file, pdf_file):
    """Convert HTML to single-page A4 PDF.
    
    Parameters
    ----------
    html_file : Path
        Path to the HTML file to convert.
    pdf_file : Path
        Path to the PDF file to save.
        
    Returns
    -------
    Path or None
        Path to generated PDF, or None if WeasyPrint not installed.
    """
    try:
        from weasyprint import HTML
        from weasyprint.text.fonts import FontConfiguration
    except ImportError:
        print(
            "  ⚠ PDF skipped: weasyprint not installed.\n"
            "    Install with:  pip install 'csttool[reports]'\n"
            "    Or via conda:  conda install -c conda-forge weasyprint\n"
            f"    HTML report is available at: {html_file}"
        )
        return None

    try:
        font_config = FontConfiguration()
        html = HTML(filename=str(html_file))
        html.write_pdf(
            str(pdf_file),
            font_config=font_config,
            presentational_hints=True,
        )
    except OSError as exc:
        print(
            "  ⚠ PDF skipped: weasyprint's native libraries (Cairo/Pango/GDK-PixBuf) "
            "could not be loaded.\n"
            f"    Error: {exc}\n"
            "    On Windows, install via conda-forge to get the native libraries:\n"
            "      conda install -c conda-forge weasyprint pango cairo gdk-pixbuf\n"
            f"    HTML report is available at: {html_file}"
        )
        return None

    print(f"  ✓ PDF generated: {pdf_file}")
    return pdf_file

def save_pdf_report(
    comparison,
    visualization_paths,
    output_dir,
    subject_id,
    version=None,
    space="Native Space",
    html_path=None
):
    """
    Generate PDF report by converting HTML report.
    """
    output_dir = Path(output_dir)
    pdf_path = output_dir / f"{subject_id}_report.pdf"
    
    if html_path is None:
        # Fallback: try to find the standard HTML report
        possible_html = output_dir / f"{subject_id}_report.html"
        if possible_html.exists():
            html_path = possible_html
        else:
            print("⚠️ HTML report not found for PDF conversion. Generating temporary HTML...")
            # We would need to call save_html_report here, but let's rely on caller
            return None

    return html_to_pdf(html_path, pdf_path)

def generate_complete_report(
    comparison,
    streamlines_left,
    streamlines_right,
    fa_map,
    affine,
    output_dir,
    subject_id,
    background_image=None,
    version=None,
    space="Native Space",
    metadata=None,
    fa_path=None,
    v1_path=None,
    density_path=None,
    roi_dseg_path=None,
    cst_left_path=None,
    cst_right_path=None,
):
    """
    Generate all report formats: JSON, CSV, and PDF with visualizations.
    
    Parameters
    ----------
    comparison : dict
        Bilateral comparison metrics
    streamlines_left : Streamlines
        Left CST streamlines
    streamlines_right : Streamlines
        Right CST streamlines
    fa_map : ndarray
        3D FA map
    affine : ndarray
        4x4 affine transformation
    output_dir : str or Path
        Output directory
    subject_id : str
        Subject identifier
    background_image : ndarray, optional
        3D T1 or FA image for tractogram QC background (defaults to fa_map)
    version : str
        csttool version string
    fa_path, v1_path, density_path, roi_dseg_path, cst_left_path, cst_right_path :
        path, optional
        Persisted products for the 1x4 report QC strip, which reads from disk so
        the figure is reproducible from the products alone. ``fa_path`` and the
        two tractogram paths are the minimum; the rest degrade their own panel.
        With no ``fa_path`` the strip is skipped entirely and the report renders
        the legacy 1x3 triptych, which is what a caller that has not migrated
        gets.

    Returns
    -------
    report_paths : dict
        Dictionary of all generated report file paths
    """

    from .visualizations import (
        plot_tract_profiles,
        plot_bilateral_comparison,
        create_summary_figure,
        plot_profile_matrix,
        plot_report_qc_strip,
        plot_tractogram_qc_triptych,
    )
    from csttool.viz import style as _style
    _style.apply_house_style()  # shared typography/spines/DPI for metrics figures

    # Use package version if not specified
    if version is None:
        version = __version__
    
    output_dir = Path(output_dir)
    viz_dir = output_dir / "visualizations"
    viz_dir.mkdir(parents=True, exist_ok=True)
    
    print("\nGenerating complete report package...")
    
    # Use FA map as background if no background image provided
    if background_image is None:
        background_image = fa_map
    
    # Generate the two composite figures used by the one-page PDF report.
    # The FA map is the QC background by default, so the grayscale FA colorbar is
    # scientifically valid (background_kind="fa").
    pdf_viz_paths = {
        'profile_matrix': plot_profile_matrix(
            comparison['left'], comparison['right'], viz_dir, subject_id
        ),
    }

    # The 1x4 strip replaces the triptych in the report. It reads persisted
    # products, so it needs paths rather than the in-memory arrays; a caller that
    # supplies none falls back to the triptych rather than losing the QC figure.
    if fa_path is not None:
        pdf_viz_paths['qc_strip'] = plot_report_qc_strip(
            fa_path=fa_path,
            v1_path=v1_path,
            density_path=density_path,
            roi_dseg_path=roi_dseg_path,
            cst_left_path=cst_left_path,
            cst_right_path=cst_right_path,
            output_dir=viz_dir,
            subject_id=subject_id,
        )
    else:
        pdf_viz_paths['tractogram_qc_triptych'] = plot_tractogram_qc_triptych(
            streamlines_left,
            streamlines_right,
            background_image,
            affine,
            viz_dir,
            subject_id,
            background_kind=("fa" if background_image is fa_map else "other"),
        )
    
    # Also generate individual plots for detailed analysis
    viz_paths = {}
    
    if 'fa' in comparison['left']:
        viz_paths['tract_profiles_fa'] = plot_tract_profiles(
            comparison['left'],
            comparison['right'],
            viz_dir,
            subject_id,
            scalar='fa'
        )
    
    if 'md' in comparison['left']:
        viz_paths['tract_profiles_md'] = plot_tract_profiles(
            comparison['left'],
            comparison['right'],
            viz_dir,
            subject_id,
            scalar='md'
        )
    
    viz_paths['bilateral_comparison'] = plot_bilateral_comparison(
        comparison,
        viz_dir,
        subject_id
    )
    
    viz_paths['summary'] = create_summary_figure(
        comparison,
        streamlines_left,
        streamlines_right,
        fa_map,
        affine,
        viz_dir,
        subject_id
    )
    
    # Merge all visualization paths
    viz_paths.update(pdf_viz_paths)
    
    # Generate reports
    # 1. HTML Report (critical as PDF is derived from it); pass the FA affine so
    #    the orientation code is computed dynamically.
    html_path = save_html_report(
        comparison, pdf_viz_paths, output_dir, subject_id, version, space,
        metadata, fa_affine=affine,
    )
    
    # 2. PDF Report (from HTML)
    pdf_path = save_pdf_report(comparison, pdf_viz_paths, output_dir, subject_id, version, space, html_path)
    
    report_paths = {
        'json': save_json_report(comparison, output_dir, subject_id, metadata=metadata),
        'csv': save_csv_summary(comparison, output_dir, subject_id),
        'html': html_path,
        'pdf': pdf_path,
        'visualizations': viz_paths
    }
    
    print("\n  ✓ Complete report package generated")
    return report_paths