import warnings

import numpy as np
from dipy.core.gradients import gradient_table

from csttool.defaults import (
    DEFAULT_MIN_SEPARATION_ANGLE,
    DEFAULT_NPEAKS,
    DEFAULT_RELATIVE_PEAK_THRESHOLD,
)


def get_max_sh_order(n_directions: int) -> int:
    """
    Return maximum safe SH order for given number of gradient directions.
    
    The number of SH coefficients required for order l is (l+1)(l+2)/2.
    We require n_directions >= required coefficients for stable fitting.
    
    Args:
        n_directions: Number of unique non-b0 gradient directions.
        
    Returns:
        Maximum SH order that can be safely used.
    """
    # Minimum directions required: (l+1)(l+2)/2
    # l=2: 6, l=4: 15, l=6: 28, l=8: 45
    if n_directions >= 45:
        return 8
    elif n_directions >= 28:
        return 6
    elif n_directions >= 15:
        return 4
    else:
        return 2  # DTI only


def _count_gradient_directions(gtab) -> int:
    """Count unique non-b0 gradient directions from a GradientTable."""
    import numpy as np
    from csttool.defaults import DEFAULT_B0_THRESHOLD
    # Exclude b0 volumes at the configured threshold
    b0_threshold = DEFAULT_B0_THRESHOLD
    dwi_mask = gtab.bvals > b0_threshold
    
    if not dwi_mask.any():
        return 0
    
    # Get unique directions (normalize and count unique)
    bvecs = gtab.bvecs[dwi_mask]
    # Round to avoid floating point precision issues
    unique_dirs = np.unique(np.round(bvecs, decimals=4), axis=0)
    return len(unique_dirs)


def _cluster_dwi_shells(gtab, tolerance=50.0):
    """Group non-b0 volumes into acquisition shells by b-value proximity."""
    dwi_indices = np.flatnonzero(~np.asarray(gtab.b0s_mask, dtype=bool))
    ordered = dwi_indices[np.argsort(np.asarray(gtab.bvals)[dwi_indices])]
    shells = []
    for index in ordered:
        b_value = float(gtab.bvals[index])
        if not shells or abs(b_value - shells[-1]["center"]) > tolerance:
            shells.append({"indices": [int(index)], "center": b_value})
        else:
            shells[-1]["indices"].append(int(index))
            values = np.asarray(gtab.bvals)[shells[-1]["indices"]]
            shells[-1]["center"] = float(np.median(values))
    return shells


def select_csa_shell(data, gtab, *, tolerance=50.0):
    """Select one shell plus all b0 volumes for CSA-ODF fitting.

    Prefer a shell with at least six measurements in the conventional
    700–3000 s/mm² range, breaking ties by proximity to b=2000. Extremely
    undersampled inputs retain their best-sampled shell with an explicit
    warning so legacy low-direction diagnostics still run; SH validation
    independently reduces the requested order.
    """
    data = np.asanyarray(data)
    if data.shape[-1] != len(gtab.bvals):
        raise ValueError(
            f"DWI has {data.shape[-1]} volumes but gradient table has "
            f"{len(gtab.bvals)}."
        )
    b0_mask = np.asarray(gtab.b0s_mask, dtype=bool)
    if not b0_mask.any():
        raise ValueError("CSA-ODF fitting requires at least one b0 volume.")

    shells = _cluster_dwi_shells(gtab, tolerance=tolerance)
    if not shells:
        raise ValueError("CSA-ODF fitting requires at least one DWI volume.")
    eligible = [shell for shell in shells if len(shell["indices"]) >= 6]
    fallback = None
    if eligible:
        preferred = [
            shell for shell in eligible if 700.0 <= shell["center"] <= 3000.0
        ]
        candidates = preferred or eligible
    else:
        candidates = shells
        fallback = "no shell has six directions; fit is underdetermined"
        warnings.warn(
            "No DWI shell has six directions; using the best-sampled shell. "
            "CSA-ODF estimates are underdetermined.",
            UserWarning,
            stacklevel=2,
        )
    selected = max(
        candidates,
        key=lambda shell: (
            len(shell["indices"]),
            -abs(shell["center"] - 2000.0),
            -shell["center"],
        ),
    )

    selected_mask = b0_mask.copy()
    selected_mask[selected["indices"]] = True
    selected_bvals = np.asarray(gtab.bvals)[selected_mask]
    selected_bvecs = np.asarray(gtab.bvecs)[selected_mask]
    b0_threshold = float(getattr(gtab, "b0_threshold", 50.0))
    selected_gtab = gradient_table(
        bvals=selected_bvals,
        bvecs=selected_bvecs,
        b0_threshold=b0_threshold,
    )
    selected_data = data if selected_mask.all() else data[..., selected_mask]
    shell_values = np.asarray(gtab.bvals)[selected["indices"]]
    metadata = {
        "model": "CSA-ODF",
        "volume_selection": "single-shell",
        "selection_rule": (
            "all b0 volumes plus the best-sampled shell in 700-3000 s/mm2; "
            "ties nearest b=2000 s/mm2"
        ),
        "shell_tolerance": float(tolerance),
        "selected_shell_b": float(np.median(shell_values)),
        "selected_shell_min_b": float(shell_values.min()),
        "selected_shell_max_b": float(shell_values.max()),
        "b0_count": int(b0_mask.sum()),
        "direction_count": len(selected["indices"]),
        "fit_volume_count": int(selected_mask.sum()),
        "fallback": fallback,
        "input_volume_count": int(len(gtab.bvals)),
        "available_shells": [
            {
                "center_b": float(shell["center"]),
                "volume_count": len(shell["indices"]),
            }
            for shell in shells
        ],
    }
    return selected_data, selected_gtab, metadata


def validate_sh_order(gtab, sh_order: int, verbose: bool = False) -> int:
    """
    Validate and potentially reduce SH order based on available gradient directions.
    
    Args:
        gtab: DIPY GradientTable.
        sh_order: Requested spherical harmonic order.
        verbose: Print validation details.
        
    Returns:
        Validated (possibly reduced) SH order.
    """
    n_directions = _count_gradient_directions(gtab)
    max_safe_order = get_max_sh_order(n_directions)
    
    if verbose:
        print(f"    • Gradient directions: {n_directions}")
        print(f"    • Maximum safe SH order: {max_safe_order}")
    
    if sh_order > max_safe_order:
        warnings.warn(
            f"Requested SH order {sh_order} requires ≥{(sh_order+1)*(sh_order+2)//2} "
            f"gradient directions, but only {n_directions} found. "
            f"Reducing to SH order {max_safe_order} for stable fitting.",
            UserWarning
        )
        return max_safe_order
    
    return sh_order


def estimate_directions(data, gtab, white_matter, sh_order=6,
                        sphere_name="symmetric362", npeaks=DEFAULT_NPEAKS,
                        verbose=False, return_metadata=False):
    """Estimate principal diffusion directions using CSA ODF model. 
       Code adapted from https://docs.dipy.org/dev/examples_built/streamline_analysis/streamline_tools.html
    
    Args:
        data (ndarray): 4D DWI data array (X, Y, Z, N).
        gtab (GradientTable): DIPY gradient table.
        white_matter (ndarray): Binary white matter mask for direction estimation.
        sh_order (int): Maximum spherical harmonic order (default 6).
            Will be automatically reduced if insufficient gradient directions.
        sphere_name (str): Name of sphere to build model around. symmetric 362 for speed, default_sphere for accuracy.
        npeaks (int): Maximum number of ODF peaks retained per voxel
            (default DEFAULT_NPEAKS). 1 keeps only the dominant direction. >1
            retains secondary peaks in crossing regions, which LocalTracking
            uses both as additional initial directions at the seed and as
            candidates for the deterministic direction getter mid-track.
            Selection stays deterministic: PeaksAndMetrics subclasses DIPY's
            EuDXDirectionGetter, documented as a "Deterministic Direction
            Getter based on peak directions".
        verbose (bool): Print processing details.
        
    Returns:
        PeaksAndMetrics: Direction field object containing:
            - peak_dirs: Principal directions per voxel
            - peak_values: ODF values at peaks
            - peak_indices: Indices into the sphere
            - gfa: Generalized fractional anisotropy
    """
    from dipy.reconst.shm import CsaOdfModel
    from dipy.direction import peaks_from_model
    from dipy.data import get_sphere

    sphere = get_sphere(name=sphere_name)
    
    # CSA is a single-shell model. Never silently fit all shells from a
    # multi-shell acquisition.
    model_data, model_gtab, model_metadata = select_csa_shell(data, gtab)
    validated_sh_order = validate_sh_order(
        model_gtab, sh_order, verbose=verbose
    )
    model_metadata["requested_sh_order"] = int(sh_order)
    model_metadata["validated_sh_order"] = int(validated_sh_order)
    model_metadata["relative_peak_threshold"] = float(
        DEFAULT_RELATIVE_PEAK_THRESHOLD
    )
    model_metadata["min_separation_angle"] = float(
        DEFAULT_MIN_SEPARATION_ANGLE
    )
    model_metadata["npeaks"] = int(npeaks)
    model_metadata["sphere"] = sphere_name

    if verbose:
        print(f"  → Estimating directions (CSA ODF, SH order={validated_sh_order})...")
        print(
            f"    • Selected b={model_metadata['selected_shell_b']:.0f} s/mm² "
            f"({model_metadata['direction_count']} directions + "
            f"{model_metadata['b0_count']} b0)"
        )
        print(f"    • White matter voxels: {white_matter.sum():,}")

    csamodel = CsaOdfModel(model_gtab, sh_order_max=validated_sh_order)
    
    csapeaks = peaks_from_model(
        model=csamodel,
        data=model_data,
        sphere=sphere,
        relative_peak_threshold=DEFAULT_RELATIVE_PEAK_THRESHOLD,
        min_separation_angle=DEFAULT_MIN_SEPARATION_ANGLE,
        mask=white_matter,
        # Retains this many peaks per voxel; no influence on determinism.
        # See DEFAULT_NPEAKS in defaults.py for what changing it does.
        npeaks=npeaks,
    )
    
    if verbose:
        # Count voxels with valid peaks. Index 0 is the dominant peak -- DIPY
        # sorts peaks by descending ODF value -- so this counts voxels with at
        # least one peak regardless of how many are retained.
        valid_peaks = (csapeaks.peak_values[..., 0] > 0).sum()
        coverage = valid_peaks / white_matter.sum() * 100
        print(f"    • Voxels with valid peaks: {valid_peaks:,}")
        print(f"    • Coverage: {coverage:.1f}%")
    
    if return_metadata:
        return csapeaks, model_metadata
    return csapeaks
