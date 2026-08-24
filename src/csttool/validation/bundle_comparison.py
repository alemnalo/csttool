"""
Bundle Comparison Metrics for CST Validation

Provides metrics for comparing candidate CST tractograms against
reference bundles (e.g., TractoInferno PYT).
"""

import json
from pathlib import Path
from datetime import datetime
import warnings

import numpy as np
import nibabel as nib
from nibabel.streamlines import load as load_trk
from dipy.tracking.streamline import set_number_of_points, length
from dipy.tracking.utils import density_map

from ..reproducibility.context import DEFAULT_SEED, derive_seed


class SpatialMismatchError(ValueError):
    """Raised when tractogram and reference space do not match."""
    pass


def _load_streamlines(trk_path: str | Path):
    """Load streamlines from a .trk file."""
    trk_path = Path(trk_path)
    if not trk_path.exists():
        raise FileNotFoundError(f"Tractogram not found: {trk_path}")
    
    tractogram = load_trk(str(trk_path))
    return tractogram.streamlines, tractogram.affine, tractogram.header


def _load_ref_anatomy(ref_path: str | Path):
    """Load reference anatomy (affine and shape)."""
    ref_path = Path(ref_path)
    if not ref_path.exists():
        raise FileNotFoundError(f"Reference anatomy not found: {ref_path}")
    
    img = nib.load(ref_path)
    return img.affine, img.shape, img.header


def _streamlines_to_density(streamlines, affine, shape):
    """Convert streamlines to a binary density map."""
    dm = density_map(streamlines, affine, shape)
    return (dm > 0).astype(np.float32)


def check_spatial_compatibility(
    candidate_path: str | Path,
    reference_path: str | Path,
    ref_anatomy_path: str | Path,
    tol_trans: float = 1.0,
    tol_rot: float = 1e-3,
):
    """
    Verify strictly that candidate and reference tractograms match the
    geometry defined by the reference anatomy image.
    
    Args:
        candidate_path: Path to candidate .trk
        reference_path: Path to reference .trk
        ref_anatomy_path: Path to reference NIfTI (defines grid)
        tol_trans: Translation tolerance in mm
        tol_rot: Rotation/Scaling tolerance
    
    Raises:
        SpatialMismatchError: If affines do not match.
    """
    _, cand_affine, _ = _load_streamlines(candidate_path)
    _, ref_affine, _ = _load_streamlines(reference_path)
    target_affine, _, _ = _load_ref_anatomy(ref_anatomy_path)

    for name, aff in [("Candidate", cand_affine), ("Reference", ref_affine)]:
        # Check translation (last column)
        trans_diff = np.linalg.norm(aff[:3, 3] - target_affine[:3, 3])
        if trans_diff > tol_trans:
            raise SpatialMismatchError(
                f"{name} tractogram translation mismatch: {trans_diff:.3f}mm > {tol_trans}mm\n"
                f"Expected: {target_affine[:3, 3]}\n"
                f"Got:      {aff[:3, 3]}"
            )

        # Check rotation/scale (3x3 submatrix)
        rot_diff = np.max(np.abs(aff[:3, :3] - target_affine[:3, :3]))
        if rot_diff > tol_rot:
            raise SpatialMismatchError(
                f"{name} tractogram rotation/scale mismatch: max diff {rot_diff:.2e} > {tol_rot}\n"
                f"Expected:\n{target_affine[:3, :3]}\n"
                f"Got:\n{aff[:3, :3]}"
            )


def check_hemisphere_alignment(
    candidate_path: str | Path,
    reference_path: str | Path,
    warn_threshold: float = 30.0,
):
    """
    Check if candidate and reference centroids are reasonably close.
    Warn if distance is large (suggesting hemisphere swap or bad tracking).
    
    Args:
        candidate_path: Path to candidate .trk
        reference_path: Path to reference .trk
        warn_threshold: Distance threshold (mm) for warning
        
    Returns:
        warning_msg (str) or None
    """
    cand_streamlines, _, _ = _load_streamlines(candidate_path)
    ref_streamlines, _, _ = _load_streamlines(reference_path)
    
    if len(cand_streamlines) == 0 or len(ref_streamlines) == 0:
        return None
        
    # Compute centroids
    def get_centroid(streamlines):
        # Quick approx: mean of all points
        # To handle massive bundles, subsample or iterate
        # For validation, subsample first 1000 is enough for rough centroid
        sample = streamlines[:1000]
        points = np.vstack([np.array(sl) for sl in sample]) # Use raw points
        return np.mean(points, axis=0)

    c_cent = get_centroid(cand_streamlines)
    r_cent = get_centroid(ref_streamlines)
    
    dist = np.linalg.norm(c_cent - r_cent)
    
    if dist > warn_threshold:
        return (
            f"Candidate centroid is {dist:.1f}mm away from Reference. "
            "Possible hemisphere swap?"
        )
    return None


def compute_bundle_overlap(
    candidate_path: str | Path,
    reference_path: str | Path,
    ref_anatomy_path: str | Path,
) -> dict:
    """
    Compute Dice overlap coefficient between two bundles on a fixed grid.
    
    Args:
        candidate_path: Path to candidate .trk file
        reference_path: Path to reference .trk file
        ref_anatomy_path: Path to reference NIfTI defining the grid
    
    Returns:
        dict with 'dice', 'intersection_volume', 'candidate_volume', 
        'reference_volume' keys
    """
    cand_streamlines, _, _ = _load_streamlines(candidate_path)
    ref_streamlines, _, _ = _load_streamlines(reference_path)
    grid_affine, grid_shape, _ = _load_ref_anatomy(ref_anatomy_path)
    
    # Create density maps on the REFERENCE grid
    cand_density = _streamlines_to_density(cand_streamlines, grid_affine, grid_shape)
    ref_density = _streamlines_to_density(ref_streamlines, grid_affine, grid_shape)
    
    # Compute volumes
    intersection = np.sum(cand_density * ref_density)
    cand_volume = np.sum(cand_density)
    ref_volume = np.sum(ref_density)
    
    # Dice coefficient
    if len(ref_streamlines) == 0:
        dice = float('nan')
    elif len(cand_streamlines) == 0:
        dice = 0.0
    elif cand_volume + ref_volume == 0:
        dice = 0.0
    else:
        dice = 2.0 * intersection / (cand_volume + ref_volume)
    
    return {
        "dice": float(dice),
        "intersection_volume": int(intersection),
        "candidate_volume": int(cand_volume),
        "reference_volume": int(ref_volume),
    }


def compute_overreach(
    candidate_path: str | Path,
    reference_path: str | Path,
    ref_anatomy_path: str | Path,
) -> dict:
    """
    Compute overreach: fraction of candidate bundle outside reference envelope.
    Computed on non-dilated density maps.
    
    Overreach = |candidate - reference| / |candidate|
    """
    cand_streamlines, _, _ = _load_streamlines(candidate_path)
    ref_streamlines, _, _ = _load_streamlines(reference_path)
    grid_affine, grid_shape, _ = _load_ref_anatomy(ref_anatomy_path)
    
    cand_density = _streamlines_to_density(cand_streamlines, grid_affine, grid_shape)
    ref_density = _streamlines_to_density(ref_streamlines, grid_affine, grid_shape)
    
    # Overreach: candidate voxels not in reference
    overreach_mask = cand_density * (1 - ref_density)
    overreach_volume = np.sum(overreach_mask)
    cand_volume = np.sum(cand_density)
    
    if cand_volume == 0:
        overreach = 0.0
    else:
        overreach = float(overreach_volume) / float(cand_volume)
    
    return {
        "overreach": float(overreach),
        "overreach_volume": int(overreach_volume),
        "candidate_volume": int(cand_volume),
    }


def compute_coverage(
    candidate_path: str | Path,
    reference_path: str | Path,
    ref_anatomy_path: str | Path,
) -> dict:
    """
    Compute coverage: fraction of reference bundle covered by candidate.
    Computed on non-dilated density maps.
    
    Coverage = |candidate ∩ reference| / |reference|
    """
    cand_streamlines, _, _ = _load_streamlines(candidate_path)
    ref_streamlines, _, _ = _load_streamlines(reference_path)
    grid_affine, grid_shape, _ = _load_ref_anatomy(ref_anatomy_path)
    
    cand_density = _streamlines_to_density(cand_streamlines, grid_affine, grid_shape)
    ref_density = _streamlines_to_density(ref_streamlines, grid_affine, grid_shape)
    
    intersection = np.sum(cand_density * ref_density)
    ref_volume = np.sum(ref_density)
    
    if ref_volume == 0:
        coverage = float('nan')
    else:
        coverage = float(intersection) / float(ref_volume)
        
    return {
        "coverage": float(coverage),
        "intersection_volume": int(intersection),
        "reference_volume": int(ref_volume),
    }


def streamline_count_ratio(
    candidate_path: str | Path,
    reference_path: str | Path,
) -> dict:
    """
    Ratio of streamline counts: Count(Candidate) / Count(Reference).
    """
    cand_streamlines, _, _ = _load_streamlines(candidate_path)
    ref_streamlines, _, _ = _load_streamlines(reference_path)
    
    n_cand = len(cand_streamlines)
    n_ref = len(ref_streamlines)
    
    if n_ref == 0:
        ratio = float('nan')
    else:
        ratio = n_cand / n_ref
        
    return {
        "streamline_count_ratio": float(ratio),
        "num_candidate": n_cand,
        "num_reference": n_ref,
    }


def mean_closest_distance(
    candidate_path: str | Path,
    reference_path: str | Path,
    step_size_mm: float = 2.0,
    num_samples: int = 1000,
    seed: int = DEFAULT_SEED,
) -> dict:
    """
    Compute Symmetric Mean of Closest Distances (MDF) between two bundles.
    
    MDF = (mean(min_dist(A->B)) + mean(min_dist(B->A))) / 2
    
    Streamlines are resampled to a fixed step size (mm) for robustness.

    A bundle larger than ``num_samples`` is subsampled, so the returned
    distances depend on which streamlines are drawn. The draw is seeded from
    ``seed`` through :func:`~csttool.reproducibility.context.derive_seed`;
    it previously came from the unseeded global NumPy RNG, which made these
    numbers differ between runs on the same inputs.
    """
    rng = np.random.default_rng(derive_seed(seed, "bundle_comparison"))
    cand_streamlines, _, _ = _load_streamlines(candidate_path)
    ref_streamlines, _, _ = _load_streamlines(reference_path)
    
    cand_list = list(cand_streamlines)
    ref_list = list(ref_streamlines)
    
    if len(cand_list) == 0 or len(ref_list) == 0:
        return {
            "mdf_symmetric": float("nan"),
            "mdf_canc_to_ref": float("nan"),
            "mdf_ref_to_cand": float("nan"),
            "mdf_std": float("nan"),
        }

    def _resample_and_sample(streamlines, step_size, max_samples):
        # 1. Resample based on length
        resampled = []
        for sl in streamlines:
            # dipy length returns an iterator/array, get first element
            sl_len = list(length([sl]))[0]
            if sl_len < step_size:
                n_points = 2
            else:
                n_points = max(2, int(sl_len / step_size))
            
            resampled.append(set_number_of_points(sl, n_points))
        
        # 2. Subsample bundle using list manipulation
        if len(resampled) > max_samples:
             indices = np.sort(
                 rng.choice(len(resampled), max_samples, replace=False)
             )
             resampled = [resampled[i] for i in indices]
             
        # Flatten into point cloud
        if not resampled:
             return np.empty((0, 3), dtype=np.float32)
             
        all_points = np.vstack(resampled)
        return all_points.astype(np.float32)

    # Get point clouds
    cand_points = _resample_and_sample(cand_list, step_size_mm, num_samples)
    ref_points = _resample_and_sample(ref_list, step_size_mm, num_samples)

    if len(cand_points) == 0 or len(ref_points) == 0:
         return {
            "mdf_symmetric": float("nan"),
        }

    # Helper for directed distance
    def _directed_dist(source_pts, target_pts):
        # Use KDTree or simple broadcasting if small enough
        # If thousands of points, broadcasting (N, M) is heavy.
        # Simple loop with chunks or KDTree is verifyable. 
        # For simplicity without scipy if possible? numpy is available.
        # simple broadcasting for < 10k points is fine.
        
        # Let's use simple chunked broadcasting
        dists = []
        chunk_size = 100
        for i in range(0, len(source_pts), chunk_size):
            chunk = source_pts[i:i+chunk_size]
            # (Chunk, 1, 3) - (1, Target, 3)
            diff = chunk[:, np.newaxis, :] - target_pts[np.newaxis, :, :]
            # (Chunk, Target)
            d = np.min(np.linalg.norm(diff, axis=2), axis=1)
            dists.extend(d)
        return np.array(dists)

    # Note: Traditional MDF is streamline-wise `min(avg(dist(s1, s2)))`. 
    # Point-cloud based is "Average Hausdorff" or similar.
    # Given the previous implementation used "streamline-wise" logic, 
    # but resampled to fixed N points (making them vectors), 
    # I should try to preserve "Streamline-wise" distance if possible for "Bundle" similarity.
    # BUT, point-cloud is often better for "shape fidelity".
    # I will stick to the previous streamline-based logic but adapted for variable length?
    # No, variable length makes vectorization hard.
    # Plan said: "Resample streamlines to a fixed step size... Fixed point count is deprecated."
    # If I use fixed step size, I get variable point counts.
    # So I cannot easily do streamline-to-streamline distance without DTW or resampling again.
    # Let's use the Point-Cloud approach (Mean Average Minimum Distance) which is robust and standard for "Chamfer-like" metrics.
    # It answers "how far is the average point on A from B".

    cand_to_ref = _directed_dist(cand_points, ref_points)
    ref_to_cand = _directed_dist(ref_points, cand_points)
    
    mean_c2r = np.mean(cand_to_ref)
    mean_r2c = np.mean(ref_to_cand)
    symmetric = (mean_c2r + mean_r2c) / 2.0
    
    return {
        "mdf_symmetric": float(symmetric),
        "mdf_cand_to_ref": float(mean_c2r),
        "mdf_ref_to_cand": float(mean_r2c),
        "mdf_std": float(np.std(np.concatenate([cand_to_ref, ref_to_cand]))),
    }


def generate_validation_report(
    metrics: dict,
    output_path: str | Path,
    candidate_paths: list = None,
    reference_paths: list = None,
    ref_anatomy_path: str | Path = None,
) -> Path:
    """Generate a JSON validation report."""
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    
    report = {
        "timestamp": datetime.now().isoformat(),
        "candidate_tractograms": [str(p) for p in (candidate_paths or [])],
        "reference_tractograms": [str(p) for p in (reference_paths or [])],
        "reference_anatomy": str(ref_anatomy_path) if ref_anatomy_path else None,
        "metrics": metrics,
    }
    
    with open(output_path, "w") as f:
        json.dump(report, f, indent=2)
    
    return output_path
