"""
endpoint_filtering.py

Filter streamlines based on endpoint locations in ROI masks.
Extracts bilateral CST by requiring streamlines to connect brainstem ↔ motor cortex.

Pipeline position: Registration → Warp Atlas → Create ROI Masks → **Endpoint Filtering**
"""

import numpy as np
import nibabel as nib
from pathlib import Path
from dipy.tracking.streamline import Streamlines
from dipy.io.streamline import load_tractogram, save_tractogram
from dipy.io.stateful_tractogram import StatefulTractogram, Space


def world_to_voxel(points, affine):
    """
    Convert world coordinates (mm) to voxel indices.
    
    Parameters
    ----------
    points : ndarray, shape (N, 3)
        Points in world coordinates (mm).
    affine : ndarray, shape (4, 4)
        Affine matrix mapping voxel to world coordinates.
        
    Returns
    -------
    voxels : ndarray, shape (N, 3)
        Voxel indices (integers).
    """
    # Invert affine: world -> voxel
    inv_affine = np.linalg.inv(affine)
    
    # Add homogeneous coordinate
    ones = np.ones((points.shape[0], 1))
    points_h = np.hstack([points, ones])
    
    # Transform
    voxels_h = points_h @ inv_affine.T
    voxels = np.round(voxels_h[:, :3]).astype(int)
    
    return voxels


def get_streamline_endpoints(streamline):
    """
    Get the two endpoints of a streamline.
    
    Parameters
    ----------
    streamline : ndarray, shape (N, 3)
        Single streamline as array of points.
        
    Returns
    -------
    endpoints : tuple
        (start_point, end_point) each shape (3,)
    """
    return streamline[0], streamline[-1]


def point_in_mask(point, mask, affine):
    """
    Check if a world-coordinate point falls within a binary mask.
    
    Parameters
    ----------
    point : ndarray, shape (3,)
        Point in world coordinates (mm).
    mask : ndarray
        Binary mask in voxel space.
    affine : ndarray, shape (4, 4)
        Affine matrix for the mask.
        
    Returns
    -------
    inside : bool
        True if point is inside the mask.
    """
    # Convert to voxel coordinates
    voxel = world_to_voxel(point.reshape(1, 3), affine)[0]
    
    # Check bounds
    if (voxel[0] < 0 or voxel[0] >= mask.shape[0] or
        voxel[1] < 0 or voxel[1] >= mask.shape[1] or
        voxel[2] < 0 or voxel[2] >= mask.shape[2]):
        return False
    
    return mask[voxel[0], voxel[1], voxel[2]] > 0


def endpoints_in_rois(streamline, roi_a, roi_b, affine):
    """
    Check if streamline endpoints are in two different ROIs.
    
    Checks both directions (start↔end) since streamline orientation
    is arbitrary.
    
    Parameters
    ----------
    streamline : ndarray, shape (N, 3)
        Single streamline.
    roi_a : ndarray
        First ROI binary mask.
    roi_b : ndarray
        Second ROI binary mask.
    affine : ndarray, shape (4, 4)
        Affine matrix for the ROI masks.
        
    Returns
    -------
    valid : bool
        True if one endpoint is in roi_a and the other in roi_b.
    """
    start, end = get_streamline_endpoints(streamline)
    
    # Check both orientations
    # Option 1: start in roi_a, end in roi_b
    if point_in_mask(start, roi_a, affine) and point_in_mask(end, roi_b, affine):
        return True
    
    # Option 2: start in roi_b, end in roi_a
    if point_in_mask(start, roi_b, affine) and point_in_mask(end, roi_a, affine):
        return True
    
    return False


def filter_streamlines_by_endpoints(
    streamlines,
    roi_a,
    roi_b,
    affine,
    verbose=True
):
    """
    Filter streamlines that connect two ROIs.

    Parameters
    ----------
    streamlines : Streamlines or list
        Input streamlines to filter.
    roi_a : ndarray
        First ROI binary mask (e.g., brainstem).
    roi_b : ndarray
        Second ROI binary mask (e.g., motor cortex).
    affine : ndarray, shape (4, 4)
        Affine matrix for the ROI masks.
    verbose : bool, optional
        Print progress information.

    Returns
    -------
    filtered : Streamlines
        Streamlines with endpoints in both ROIs.
    indices : ndarray
        Indices of kept streamlines in original array.
    """
    kept_indices = []

    for i, sl in enumerate(streamlines):
        if endpoints_in_rois(sl, roi_a, roi_b, affine):
            kept_indices.append(i)

        # Progress update
        if verbose and (i + 1) % 10000 == 0:
            print(f"    • Processed {i + 1:,} / {len(streamlines):,} streamlines...")

    kept_indices = np.array(kept_indices, dtype=int)

    if len(kept_indices) > 0:
        filtered = Streamlines([streamlines[i] for i in kept_indices])
    else:
        filtered = Streamlines()

    if verbose:
        print(f"  → Filtering: {len(streamlines):,} → {len(filtered):,} streamlines")
    if verbose:
        print(f"    • Reduction: {100 * (1 - len(filtered) / len(streamlines)):.1f}%")

    return filtered, kept_indices


def extract_bilateral_cst(
    streamlines,
    masks,
    affine,
    min_length=20.0,
    max_length=200.0,
    verbose=True
):
    """
    Extract bilateral CST from whole-brain tractogram.
    
    Filters streamlines that connect brainstem to motor cortex,
    separated by hemisphere.
    
    Parameters
    ----------
    streamlines : Streamlines
        Whole-brain tractogram.
    masks : dict
        ROI masks from create_cst_roi_masks():
        - 'brainstem': Brainstem mask
        - 'motor_left': Left motor cortex mask
        - 'motor_right': Right motor cortex mask
    affine : ndarray, shape (4, 4)
        Affine matrix for the masks.
    min_length : float, optional
        Minimum streamline length in mm. Default is 20.
    max_length : float, optional
        Maximum streamline length in mm. Default is 200.
    verbose : bool, optional
        Print progress information.
        
    Returns
    -------
    result : dict
        Dictionary containing:
        - 'cst_left': Left CST streamlines
        - 'cst_right': Right CST streamlines
        - 'cst_combined': All CST streamlines
        - 'left_indices': Indices of left CST in original
        - 'right_indices': Indices of right CST in original
        - 'stats': Extraction statistics
    """
    from dipy.tracking.streamline import length
    
    if verbose:
        print("=" * 60)
        print("EXTRACT BILATERAL CST")
        print("=" * 60)
        print(f"\nInput: {len(streamlines):,} streamlines")

    # -------------------------------------------------------------------------
    # Step 1: Length filtering
    # -------------------------------------------------------------------------
    if verbose:
        print(f"\n[Step 1/3] Length filtering ({min_length}-{max_length} mm)...")

    lengths = np.array([length(sl) for sl in streamlines])
    length_mask = (lengths >= min_length) & (lengths <= max_length)
    length_valid_indices = np.where(length_mask)[0]

    streamlines_length_filtered = Streamlines([streamlines[i] for i in length_valid_indices])

    if verbose:
        print(f"  → {len(streamlines):,} → {len(streamlines_length_filtered):,} streamlines")
    if verbose:
        print(f"    • Removed: {len(streamlines) - len(streamlines_length_filtered):,} (too short/long)")

    # -------------------------------------------------------------------------
    # Step 2: Extract Left CST (brainstem ↔ motor_left)
    # -------------------------------------------------------------------------
    if verbose:
        print("\n[Step 2/3] Extracting Left CST (brainstem ↔ motor_left)...")

    cst_left, left_indices_filtered = filter_streamlines_by_endpoints(
        streamlines_length_filtered,
        masks['brainstem'],
        masks['motor_left'],
        affine,
        verbose=verbose
    )

    # Map back to original indices
    left_indices_original = length_valid_indices[left_indices_filtered]

    # -------------------------------------------------------------------------
    # Step 3: Extract Right CST (brainstem ↔ motor_right)
    # -------------------------------------------------------------------------
    if verbose:
        print("\n[Step 3/3] Extracting Right CST (brainstem ↔ motor_right)...")

    cst_right, right_indices_filtered = filter_streamlines_by_endpoints(
        streamlines_length_filtered,
        masks['brainstem'],
        masks['motor_right'],
        affine,
        verbose=verbose
    )

    # Map back to original indices
    right_indices_original = length_valid_indices[right_indices_filtered]
    
    # -------------------------------------------------------------------------
    # Combine results
    # -------------------------------------------------------------------------
    cst_combined = Streamlines(list(cst_left) + list(cst_right))
    
    # Statistics
    stats = {
        'total_input': len(streamlines),
        'after_length_filter': len(streamlines_length_filtered),
        'cst_left_count': len(cst_left),
        'cst_right_count': len(cst_right),
        'cst_total_count': len(cst_combined),
        'extraction_rate': len(cst_combined) / len(streamlines) * 100 if len(streamlines) > 0 else 0,
        'left_right_ratio': len(cst_left) / len(cst_right) if len(cst_right) > 0 else float('inf'),
        'min_length': min_length,
        'max_length': max_length
    }
    
    if verbose:
        print("\n" + "=" * 60)
        print("  ✓ CST extraction complete")
        print("=" * 60)
        print("\nResults:")
        print(f"  Left CST:  {stats['cst_left_count']:,} streamlines")
        print(f"  Right CST: {stats['cst_right_count']:,} streamlines")
        print(f"  Total:     {stats['cst_total_count']:,} streamlines")
    if verbose:
        print("\nDiagnostics:")
        print(f"    • Extraction rate: {stats['extraction_rate']:.2f}%")
        print(f"    • L/R ratio: {stats['left_right_ratio']:.2f}")
    
    return {
        'cst_left': cst_left,
        'cst_right': cst_right,
        'cst_combined': cst_combined,
        'left_indices': left_indices_original,
        'right_indices': right_indices_original,
        'stats': stats
    }


def save_cst_tractograms(
    cst_result,
    reference_img,
    output_dir,
    subject_id=None,
    verbose=True
):
    """
    Save extracted CST tractograms as .trk files.

    Parameters
    ----------
    cst_result : dict
        Output from extract_bilateral_cst().
    reference_img : Nifti1Image
        Reference image for tractogram space.
    output_dir : str or Path
        Output directory.
    subject_id : str, optional
        Subject identifier for filenames.
    verbose : bool, optional
        Print progress information.

    Returns
    -------
    paths : dict
        Paths to saved tractogram files.
    """
    output_dir = Path(output_dir)
    trk_dir = output_dir / "trk"
    trk_dir.mkdir(parents=True, exist_ok=True)

    prefix = f"{subject_id}_" if subject_id else ""
    paths = {}

    # Save Left CST
    if len(cst_result['cst_left']) > 0:
        left_path = trk_dir / f"{prefix}cst_left.trk"
        sft_left = StatefulTractogram(cst_result['cst_left'], reference_img, Space.RASMM)
        save_tractogram(sft_left, str(left_path))
        paths['cst_left'] = left_path
        if verbose:
            print(f"  ✓ Left CST: {left_path}")
    else:
        if verbose:
            print("  ⚠️ No Left CST streamlines to save")
        paths['cst_left'] = None

    # Save Right CST
    if len(cst_result['cst_right']) > 0:
        right_path = trk_dir / f"{prefix}cst_right.trk"
        sft_right = StatefulTractogram(cst_result['cst_right'], reference_img, Space.RASMM)
        save_tractogram(sft_right, str(right_path))
        paths['cst_right'] = right_path
        if verbose:
            print(f"  ✓ Right CST: {right_path}")
    else:
        if verbose:
            print("  ⚠️ No Right CST streamlines to save")
        paths['cst_right'] = None

    # Save Combined CST
    if len(cst_result['cst_combined']) > 0:
        combined_path = trk_dir / f"{prefix}cst_bilateral.trk"
        sft_combined = StatefulTractogram(cst_result['cst_combined'], reference_img, Space.RASMM)
        save_tractogram(sft_combined, str(combined_path))
        paths['cst_combined'] = combined_path
        if verbose:
            print(f"  ✓ Bilateral CST: {combined_path}")
    else:
        if verbose:
            print("  ⚠️ No CST streamlines to save")
        paths['cst_combined'] = None

    return paths


def save_extraction_report(
    cst_result,
    output_paths,
    output_dir,
    subject_id=None
):
    """
    Save JSON report of CST extraction results.
    """
    import json
    from datetime import datetime
    from csttool.reproducibility import get_provenance_dict
    
    output_dir = Path(output_dir)
    log_dir = output_dir / "logs"
    log_dir.mkdir(parents=True, exist_ok=True)
    
    prefix = f"{subject_id}_" if subject_id else ""
    
    report = {
        'processing_date': datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
        'subject_id': subject_id,
        'provenance': get_provenance_dict(),
        'statistics': cst_result['stats'],
        'output_files': {
            'cst_left': str(output_paths.get('cst_left')) if output_paths.get('cst_left') else None,
            'cst_right': str(output_paths.get('cst_right')) if output_paths.get('cst_right') else None,
            'cst_combined': str(output_paths.get('cst_combined')) if output_paths.get('cst_combined') else None
        }
    }
    
    report_path = log_dir / f"{prefix}cst_extraction_report.json"
    with open(report_path, 'w') as f:
        json.dump(report, f, indent=2)
    
    print(f"  ✓ Report: {report_path}")
    
    return report_path