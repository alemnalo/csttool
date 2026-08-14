"""
preprocess.py

High-level orchestrator for the DWI preprocessing pipeline.
Consolidates all preprocessing steps into a single function.
"""

from pathlib import Path

import numpy as np

from ..defaults import DEFAULT_B0_THRESHOLD, DEFAULT_DENOISE_METHOD
from .modules.external_declaration import (
    DEFAULT_EXTERNAL_CORRECTION,
    declaration_record,
)
from .modules.load_dataset import load_dataset
from .modules.denoise import denoise
from .modules.gibbs_unringing import gibbs_unringing
from .modules.background_segmentation import background_segmentation
from .modules.perform_motion_correction import perform_motion_correction
from .modules.reorient_gradients import (
    max_rotation_angle_deg,
    rotate_bvecs_for_motion,
)
from .modules.reslice_voxels import reslice_voxels
from .modules.save_preprocessed import save_preprocessed


def run_preprocessing(
    input_dir: str | Path,
    output_dir: str | Path,
    filename: str,
    *,
    # Gradient handling
    b0_threshold: float = DEFAULT_B0_THRESHOLD,
    # Provenance
    external_correction: str = DEFAULT_EXTERNAL_CORRECTION,
    # Denoising options
    denoise_method: str = DEFAULT_DENOISE_METHOD,
    coil_count: int = 4,
    # Optional steps
    apply_gibbs_correction: bool = False,
    apply_motion_correction: bool = False,
    target_voxel_size: tuple[float, float, float] | None = None,
    # Visualization options
    save_visualizations: bool = False,
    title_id: str | None = None,
    verbose: bool = False,
) -> dict:
    """
    Run the complete DWI preprocessing pipeline.

    Steps:
        1. Load dataset (NIfTI/DICOM + gradient table)
        2. Reslice to target voxel size (optional)
        3. Denoise (Patch2Self or NLMeans)
        4. Brain masking (median Otsu on b0 volumes)
        5. Gibbs unringing (optional)
        6. Motion correction (optional)
        7. Save outputs

    Parameters
    ----------
    input_dir : str or Path
        Directory containing input NIfTI/DICOM and gradient files.
    output_dir : str or Path
        Output directory for preprocessed files.
    filename : str
        Base filename without extension (e.g., "sub01_dwi").
    b0_threshold : float, default=50
        b-value at or below which a volume counts as a b0, in s/mm². The one
        authoritative threshold for this execution: it builds the gradient
        table, and the resulting ``gtab.b0s_mask`` is what brain masking and
        Patch2Self read.
    external_correction : str, default="unknown"
        User declaration of what correction was applied to the input *before*
        csttool received it. Recorded in the report as a declaration; never
        verified, and it changes no processing decision. See
        ``modules/external_declaration.py``.
    denoise_method : str, default="mppca"
        Denoising method: "nlmeans", "patch2self", or "mppca".
        - "nlmeans" uses PIESNO for sigma estimation; requires coil_count.
        - "patch2self" requires bvals and >= 7 directions.
        - "mppca" (Marchenko-Pastur PCA) is self-adaptive; does not
          require coil_count or bvals — it exploits the 4D DWI redundancy
          directly. Best choice for modern multi-channel acquisitions
          where the effective coil count is unknown.
    coil_count : int, default=4
        Number of scanner coils (for NLMeans noise estimation).
        Ignored when denoise_method is "patch2self" or "mppca".
    apply_gibbs_correction : bool, default=False
        Apply Gibbs ringing correction.
    apply_motion_correction : bool, default=False
        Apply between-volume motion correction.
    target_voxel_size : tuple[float, float, float] or None, default=None
        Target voxel size in mm (x, y, z). If provided, data will be resliced
        to this voxel size. If None, no reslicing is performed.
    save_visualizations : bool, default=False
        Save QC visualizations.
    verbose : bool, default=False
        Print detailed processing information.

    Returns
    -------
    dict
        Dictionary containing:
        - 'output_paths': Paths to saved files
        - 'brain_mask': The computed brain mask array
        - 'motion_correction_applied': Whether motion correction was applied
        - 'gtab': The gradient table
    """
    input_dir = Path(input_dir)
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    # -------------------------------------------------------------------------
    # Step 1: Load dataset
    # -------------------------------------------------------------------------
    if verbose:
        print(f"Loading dataset from {input_dir}")
    
    nii, gtab, nifti_dir, metadata = load_dataset(
        str(input_dir), filename, b0_threshold=b0_threshold
    )
    data = np.asarray(nii.dataobj) if hasattr(nii, 'dataobj') else nii
    affine = nii.affine if hasattr(nii, 'affine') else np.eye(4)
    
    # Get current voxel size from the NIfTI header
    current_voxel_size = nii.header.get_zooms()[:3]
    print(f"PREPROCESSING: Loaded data with shape {data.shape}")
    print(f"PREPROCESSING: Current voxel size: {current_voxel_size} mm")

    # -------------------------------------------------------------------------
    # Step 2: Reslice to target voxel size (optional)
    # -------------------------------------------------------------------------
    if target_voxel_size is not None:
        print(f"PREPROCESSING: Reslicing to target voxel size: {target_voxel_size} mm")
        data, affine = reslice_voxels(
            data,
            affine,
            voxel_size=current_voxel_size,
            new_voxel_size=target_voxel_size
        )
        print(f"PREPROCESSING: Reslicing complete. New shape: {data.shape}")
    elif verbose:
        print("PREPROCESSING: Reslicing skipped")

    # -------------------------------------------------------------------------
    # Step 3: Denoise
    # -------------------------------------------------------------------------
    denoised = denoise(
        data,
        bvals=gtab.bvals,
        brain_mask=None,
        denoise_method=denoise_method,
        N=coil_count,
        b0_threshold=b0_threshold,
    )
    print(f"PREPROCESSING: Denoising complete ({denoise_method})")

    # -------------------------------------------------------------------------
    # Step 4: Brain masking
    # -------------------------------------------------------------------------
    masked_data, brain_mask = background_segmentation(denoised, gtab)
    print("PREPROCESSING: Brain masking complete")

    # -------------------------------------------------------------------------
    # Step 5: Gibbs unringing (optional)
    # -------------------------------------------------------------------------
    if apply_gibbs_correction:
        unringed = gibbs_unringing(masked_data)
        data_for_motion = unringed
        print("PREPROCESSING: Gibbs ringing correction complete")
    else:
        data_for_motion = masked_data
        if verbose:
            print("PREPROCESSING: Gibbs ringing correction skipped")

    # -------------------------------------------------------------------------
    # Step 6: Motion correction (optional)
    # -------------------------------------------------------------------------
    motion_correction_applied = False
    reg_affines = None
    rotated_bvecs = None
    max_rotation_deg = None
    warnings: list[str] = []

    if apply_motion_correction:
        try:
            preprocessed, reg_affines = perform_motion_correction(
                data_for_motion,
                gtab,
                affine,
                brain_mask=brain_mask
            )
            motion_correction_applied = True
            print("PREPROCESSING: Motion correction complete")

            # The volumes have been rotated onto the reference pose, so the
            # b-vectors must follow: a resampled volume paired with its
            # nominal gradient direction biases FA/MD and tilts V1 (Leemans &
            # Jones 2009). This sits inside the same try block on purpose — if
            # the rotation cannot be computed, the handler below discards the
            # motion-corrected data too, so what ships is always a mutually
            # consistent data/gradient pair rather than corrected volumes with
            # uncorrected gradients (the defect this milestone fixes).
            rotated_bvecs = rotate_bvecs_for_motion(
                gtab.bvals,
                gtab.bvecs,
                reg_affines,
                gtab.b0s_mask,
                affine,
                b0_threshold=b0_threshold,
            )
            max_rotation_deg = max_rotation_angle_deg(reg_affines)
            print(
                f"PREPROCESSING: b-vectors rotated "
                f"(max estimated head rotation {max_rotation_deg:.2f}°)"
            )
        except Exception as e:
            print(f"PREPROCESSING: Motion correction failed: {e}")
            print("   Continuing without motion correction")
            warnings.append(
                f"Motion correction was requested but failed ({type(e).__name__}: {e}); "
                "continued with uncorrected data and the original gradients."
            )
            motion_correction_applied = False
            reg_affines = None
            rotated_bvecs = None
            preprocessed = data_for_motion
    else:
        preprocessed = data_for_motion
        if verbose:
            print("PREPROCESSING: Motion correction skipped")

    # -------------------------------------------------------------------------
    # Step 7: Save outputs
    # -------------------------------------------------------------------------
    suffix = "_mc" if motion_correction_applied else "_nomc"
    output_stem = f"{filename}_dwi_preproc{suffix}"
    
    # Build gradient file paths
    bval_path = input_dir / f"{filename}.bval"
    if not bval_path.exists():
        bval_path = input_dir / f"{filename}.bvals"
    
    bvec_path = input_dir / f"{filename}.bvec"
    if not bvec_path.exists():
        bvec_path = input_dir / f"{filename}.bvecs"
    
    gradient_files = {}
    if bval_path.exists():
        gradient_files['bval'] = bval_path
    if bvec_path.exists():
        gradient_files['bvec'] = bvec_path

    output_paths = save_preprocessed(
        data=preprocessed,
        affine=affine,
        output_dir=output_dir,
        filename_stem=output_stem,
        gradient_files=gradient_files if gradient_files else None,
        bvecs=rotated_bvecs,
        brain_mask=brain_mask,
        processing_params={
            'denoise_method': denoise_method,
            'b0_threshold': b0_threshold,
            'external_correction': declaration_record(external_correction),
            'gibbs_correction': apply_gibbs_correction,
            # requested vs applied: a failed motion correction used to be
            # distinguishable only by the output filename suffix.
            'motion_correction_requested': apply_motion_correction,
            'motion_correction': motion_correction_applied,
            'bvecs_rotated': rotated_bvecs is not None,
            'max_rotation_deg': max_rotation_deg,
            'resliced': target_voxel_size is not None,
            'target_voxel_size': target_voxel_size if target_voxel_size else None,
            'warnings': warnings,
        }
    )
    print(f"PREPROCESSING: Saved outputs to {output_dir}")

    # -------------------------------------------------------------------------
    # Step 8: Visualizations (optional)
    # -------------------------------------------------------------------------
    if save_visualizations:
        try:
            from .modules.visualizations import save_all_preprocessing_visualizations
            # Pass the stage directory; save_all_preprocessing_visualizations appends
            # the single "visualizations/" subdir itself. Passing an already-"visualizations"
            # path here doubled it (…/visualizations/visualizations/) so the BIDS reorg,
            # which globs …/preprocessing/visualizations/*.png, never found the figures.
            save_all_preprocessing_visualizations(
                data_original=data,  # Raw data before any processing
                data_denoised=denoised,  # Denoised (same shape as original)
                data_masked=masked_data,  # After brain masking (cropped)
                data_unringed=unringed if apply_gibbs_correction else None,
                data_preprocessed=preprocessed,
                brain_mask=brain_mask,
                gtab=gtab,
                output_dir=output_dir,
                stem=filename,
                denoise_method=denoise_method,
                reg_affines=reg_affines,
                motion_correction_applied=motion_correction_applied,
                affine=affine,
                title_id=title_id,
            )
            print("PREPROCESSING: QC visualizations saved")
        except Exception as e:
            print(f"PREPROCESSING: Visualization saving failed: {e}")

    print(f"\nPREPROCESSING COMPLETED")

    return {
        'output_paths': output_paths,
        'brain_mask': brain_mask,
        'motion_correction_applied': motion_correction_applied,
        'gtab': gtab,
    }
