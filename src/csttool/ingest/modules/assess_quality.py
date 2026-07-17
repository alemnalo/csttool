from typing import List, Tuple, Dict, Any, Optional, Union
import numpy as np
from csttool.tracking.modules.estimate_directions import get_max_sh_order
from csttool.defaults import DEFAULT_B0_THRESHOLD


def count_unique_directions(bvecs: np.ndarray, bvals: np.ndarray, b0_threshold: int = None) -> int:
    """Count unique gradient directions (excluding b=0 volumes).
    
    Args:
        bvecs: (N, 3) or (3, N) array of gradient directions
        bvals: (N,) array of b-values
        b0_threshold: b-value threshold for b=0 volumes
            (default: DEFAULT_B0_THRESHOLD)
        
    Returns:
        Number of unique gradient directions
    """
    if b0_threshold is None:
        b0_threshold = DEFAULT_B0_THRESHOLD
    if bvals.ndim == 2:
        bvals = bvals.flatten()
    if bvecs.shape[0] == 3 and bvecs.shape[1] != 3:
        bvecs = bvecs.T
        
    dwi_mask = bvals > b0_threshold
    if not np.any(dwi_mask):
        return 0
        
    dwi_bvecs = bvecs[dwi_mask]
    norms = np.linalg.norm(dwi_bvecs, axis=1, keepdims=True)
    norms[norms == 0] = 1
    normalized_bvecs = dwi_bvecs / norms
    
    unique_dirs = np.unique(np.round(normalized_bvecs, decimals=4), axis=0)
    return len(unique_dirs)


def extract_acquisition_metadata(
    bvecs: np.ndarray,
    bvals: np.ndarray,
    voxel_size: Tuple[float, float, float],
    bids_json: Optional[Dict[str, Any]] = None,
    overrides: Optional[Dict[str, Any]] = None
) -> Dict[str, Any]:
    """Extract acquisition parameters for reporting.
    
    Args:
        bvecs: (N, 3) or (3, N) array of gradient directions
        bvals: (N,) array of b-values
        voxel_size: (x, y, z) voxel dimensions in mm
        bids_json: Optional BIDS JSON metadata dict
        overrides: Optional dict of CLI-provided values (take precedence)
        
    Returns:
        Dict with acquisition metadata for reports
    """
    overrides = overrides or {}
    bids_json = bids_json or {}
    
    if bvals.ndim == 2:
        bvals = bvals.flatten()
    
    # Extract from BIDS JSON (convert units where needed)
    field_strength = None
    if 'MagneticFieldStrength' in bids_json:
        try:
            field_strength = float(bids_json['MagneticFieldStrength'])
        except (ValueError, TypeError):
            pass
    
    echo_time_ms = None
    if 'EchoTime' in bids_json:
        try:
            # BIDS stores in seconds, convert to ms
            echo_time_ms = float(bids_json['EchoTime']) * 1000
        except (ValueError, TypeError):
            pass
    
    acq = {
        'field_strength_T': field_strength,
        'echo_time_ms': echo_time_ms,
        'b_values': sorted(set(bvals.astype(int).tolist())),
        'n_directions': count_unique_directions(bvecs, bvals),
        'n_volumes': len(bvals),
        'resolution_mm': list(voxel_size),
    }
    
    # CLI overrides take precedence
    acq.update(overrides)
    
    return acq

def detect_shells(bvals: np.ndarray, tolerance: float = 0.05) -> List[Tuple[float, np.ndarray]]:
    """
    Detect b-value shells with adaptive tolerance.

    Args:
        bvals: Array of b-values
        tolerance: Relative tolerance for clustering (default 5%)

    Returns:
        List of (shell_bval, shell_indices) tuples
    """
    shells = []
    sorted_indices = np.argsort(bvals)
    sorted_bvals = bvals[sorted_indices]

    current_shell_bval = sorted_bvals[0]
    current_shell_indices = [sorted_indices[0]]

    for i in range(1, len(sorted_bvals)):
        bval = sorted_bvals[i]
        # Check if within tolerance of current shell
        if current_shell_bval == 0 or bval <= current_shell_bval * (1 + tolerance):
            current_shell_indices.append(sorted_indices[i])
        else:
            # Save current shell
            shells.append((current_shell_bval, np.array(current_shell_indices)))
            # Start new shell
            current_shell_bval = bval
            current_shell_indices = [sorted_indices[i]]

    # Add final shell
    shells.append((current_shell_bval, np.array(current_shell_indices)))

    return shells


def assess_acquisition_quality(
    bvecs: np.ndarray,
    bvals: np.ndarray,
    voxel_size: Tuple[float, float, float],
    bids_json: Optional[Dict[str, Any]] = None,
    b0_threshold: float = None,
    return_metadata: bool = False
) -> Union[List[Tuple[str, str]], Tuple[List[Tuple[str, str]], Dict[str, Any]]]:
    """
    Enhanced DWI acquisition quality assessment for CST tractography.

    Returns list of (severity, message) tuples.
    Severity levels: "CRITICAL", "WARNING", "INFO"

    Args:
        bvecs: (N, 3) or (3, N) array of gradient directions
        bvals: (N,) or (1, N) array of b-values
        voxel_size: (x, y, z) voxel dimensions in mm
        bids_json: Optional dictionary of BIDS metadata
        b0_threshold: Threshold for b=0 volumes (default: DEFAULT_B0_THRESHOLD)
        return_metadata: If True, return (warnings, metadata) tuple

    Returns:
        List of (severity, message) tuples, or tuple with metadata dict if return_metadata=True
    """
    warnings = []
    metadata = {}  # Collect statistics for reporting

    # 1. Input validation
    if b0_threshold is None:
        b0_threshold = DEFAULT_B0_THRESHOLD
    if bvals.ndim == 2:
        bvals = bvals.flatten()

    # Validate bvecs shape
    if bvecs.ndim != 2 or bvecs.shape[1] != 3:
        if bvecs.shape[0] == 3 and bvecs.shape[1] != 3:
            bvecs = bvecs.T
        else:
            warnings.append(("CRITICAL",
                f"bvecs must have shape (N, 3), got {bvecs.shape}"))
            if return_metadata:
                return warnings, metadata
            return warnings

    if bvals.shape[0] != bvecs.shape[0]:
        warnings.append(("CRITICAL",
            f"bvals ({bvals.shape[0]}) and bvecs ({bvecs.shape[0]}) count mismatch"))
        if return_metadata:
            return warnings, metadata
        return warnings

    # Check for invalid b-values
    if np.any(bvals < 0):
        warnings.append(("CRITICAL", "Negative b-values detected"))
    if np.any(bvals > 10000):
        warnings.append(("WARNING",
            f"Extremely high b-value ({np.max(bvals):.0f} s/mm²)"))

    # 2. Analyze b=0 volumes
    b0_mask = bvals <= b0_threshold
    n_b0 = np.sum(b0_mask)

    if n_b0 == 0:
        warnings.append(("CRITICAL", "No b=0 volumes found"))
    elif n_b0 < 3:
        warnings.append(("WARNING",
            f"Only {n_b0} b=0 volume(s). Recommend ≥3 for robust registration"))

    # Check b=0 distribution for motion correction
    b0_indices = np.array([])
    max_gap = 0
    if n_b0 > 1:
        b0_indices = np.where(b0_mask)[0]
        gaps = np.diff(b0_indices)
        max_gap = np.max(gaps)
        if max_gap > 30:
            warnings.append(("INFO",
                f"Largest gap between b=0 volumes is {max_gap} volumes. "
                "Consider more evenly distributed b=0s for motion correction."))
    elif n_b0 == 1:
        b0_indices = np.where(b0_mask)[0]

    # 3. Analyze DWI volumes
    dwi_mask = bvals > b0_threshold
    n_dwi = np.sum(dwi_mask)

    if n_dwi == 0:
        warnings.append(("CRITICAL", "No diffusion-weighted volumes found"))
        if return_metadata:
            metadata.update({
                'n_b0': int(n_b0),
                'n_dwi': 0,
                'n_directions': 0,
                'shells': [],
                'b0_distribution': {
                    'n_volumes': int(n_b0),
                    'max_gap': int(max_gap),
                    'indices': b0_indices.tolist()
                },
                'bids_fields_present': {}
            })
            return warnings, metadata
        return warnings

    # 4. Check gradient quality for DWI volumes
    dwi_bvecs = bvecs[dwi_mask]
    dwi_bvals = bvals[dwi_mask]

    # Check for near-zero gradients in DWI volumes
    norms = np.linalg.norm(dwi_bvecs, axis=1)
    if np.any(norms < 0.1):
        n_bad = np.sum(norms < 0.1)
        warnings.append(("CRITICAL",
            f"{n_bad} DWI volume(s) have near-zero gradient vectors"))

    # Normalize (existing logic)
    norms[norms == 0] = 1
    normalized_bvecs = dwi_bvecs / norms[:, np.newaxis]

    # 5. Shell detection
    shells = detect_shells(dwi_bvals, tolerance=0.05)

    if len(shells) > 1:
        shell_bvals = [int(s[0]) for s in shells]
        warnings.append(("INFO",
            f"Detected {len(shells)} b-value shell(s): {shell_bvals}"))

    # Per-shell quality checks
    shell_info = []
    for shell_bval, shell_indices in shells:
        if shell_bval <= b0_threshold:
            continue  # Skip b=0 shell

        # Count unique directions in this shell
        shell_vecs = normalized_bvecs[shell_indices]
        unique_shell_dirs = np.unique(np.round(shell_vecs, decimals=4), axis=0)
        n_shell_dirs = len(unique_shell_dirs)

        shell_info.append({
            'bval': float(shell_bval),
            'n_volumes': len(shell_indices),
            'n_directions': n_shell_dirs
        })

        if n_shell_dirs < 6 and shell_bval > 500:
            warnings.append(("WARNING",
                f"Shell b≈{shell_bval:.0f} has only {n_shell_dirs} unique direction(s). "
                "Consider ≥6 per shell for reliable modeling."))

    # 6. Total unique direction count (across all shells)
    unique_dirs = np.unique(np.round(normalized_bvecs, decimals=4), axis=0)
    n_directions = len(unique_dirs)

    if n_directions < 15:
        warnings.append((
            "CRITICAL",
            f"Only {n_directions} gradient directions detected. "
            f"Minimum 15 required for basic tractography, 28+ recommended."
        ))
    elif n_directions < 28:
        warnings.append((
            "WARNING",
            f"{n_directions} gradient directions limits SH order to 4. "
            f"Consider acquiring >=28 directions for SH order 6."
        ))

    # 7. Check b-value ranges
    max_bval = np.max(bvals)
    if max_bval < 800:
        warnings.append((
            "WARNING",
            f"Maximum b-value ({max_bval:.0f} s/mm²) may underestimate FA "
            f"and reduce diffusion contrast."
        ))
    elif max_bval > 3000:
        warnings.append((
            "WARNING",
            f"High b-value ({max_bval:.0f} s/mm²) may have SNR limitations. "
            f"Ensure adequate denoising."
        ))

    # 8. Voxel size checks
    # Check voxel volume
    voxel_volume_mm3 = np.prod(voxel_size)
    if voxel_volume_mm3 > 15:
        warnings.append(("WARNING",
            f"Large voxel volume ({voxel_volume_mm3:.1f} mm³) may reduce SNR"))

    # Existing max voxel check
    max_voxel = max(voxel_size)
    if max_voxel > 2.5:
        warnings.append((
            "WARNING",
            f"Large voxel size ({voxel_size[0]:.1f}x{voxel_size[1]:.1f}x"
            f"{voxel_size[2]:.1f} mm) may cause partial volume effects "
            f"in internal capsule and brainstem."
        ))

    # Enhanced anisotropy check
    voxel_ratio = max(voxel_size) / min(voxel_size)
    if voxel_ratio > 2.0:
        warnings.append(("CRITICAL",
            f"Highly anisotropic voxels ({voxel_ratio:.1f}:1) "
            "may cause severe directional bias in tractography"))
    elif voxel_ratio > 1.5:
        warnings.append(("WARNING",
            f"Anisotropic voxels (ratio {voxel_ratio:.1f}) may cause "
            "directional bias in tractography"))

    # 9. Enhanced BIDS validation
    bids_fields_present = {}
    if bids_json:
        # Check for required fields for distortion correction
        required_for_correction = ['PhaseEncodingDirection', 'TotalReadoutTime']
        missing_fields = [f for f in required_for_correction if f not in bids_json]

        if missing_fields:
            warnings.append(("WARNING",
                f"Missing BIDS fields for distortion correction: {', '.join(missing_fields)}"))

        # Validate phase encoding direction
        pe_dir = bids_json.get('PhaseEncodingDirection')
        if pe_dir and pe_dir not in ['i', 'j', 'k', 'i-', 'j-', 'k-']:
            warnings.append(("INFO",
                f"Unusual phase encoding direction: {pe_dir}"))

        # Track BIDS field presence
        bids_fields_present = {
            'PhaseEncodingDirection': 'PhaseEncodingDirection' in bids_json,
            'TotalReadoutTime': 'TotalReadoutTime' in bids_json,
            'EchoTime': 'EchoTime' in bids_json,
            'MultibandAccelerationFactor': 'MultibandAccelerationFactor' in bids_json,
        }

        # Existing echo time check
        echo_time = bids_json.get("EchoTime")
        if echo_time:
            try:
                echo_time_ms = float(echo_time) * 1000
                if echo_time_ms > 100:
                    warnings.append((
                        "WARNING",
                        f"Long echo time ({echo_time_ms:.0f} ms) reduces SNR due to T2 decay."
                    ))
            except (ValueError, TypeError):
                pass

        # Existing multiband factor check
        mb_factor = bids_json.get("MultibandAccelerationFactor")
        if mb_factor:
            try:
                if float(mb_factor) > 4:
                    warnings.append((
                        "INFO",
                        f"High multiband factor ({mb_factor}) may reduce SNR. "
                        f"Verify image quality."
                    ))
            except (ValueError, TypeError):
                pass

        # Existing parallel imaging factor check
        parallel_factor = bids_json.get("ParallelReductionFactorInPlane")
        if parallel_factor:
            try:
                if float(parallel_factor) > 3:
                    warnings.append((
                        "INFO",
                        f"High parallel imaging factor ({parallel_factor}) may reduce SNR."
                    ))
            except (ValueError, TypeError):
                pass

    # Populate metadata for return
    if return_metadata:
        metadata = {
            'n_b0': int(n_b0),
            'n_dwi': int(n_dwi),
            'n_directions': n_directions,
            'shells': shell_info,
            'b0_distribution': {
                'n_volumes': int(n_b0),
                'max_gap': int(max_gap),
                'indices': b0_indices.tolist()
            },
            'bids_fields_present': bids_fields_present
        }
        return warnings, metadata

    return warnings
