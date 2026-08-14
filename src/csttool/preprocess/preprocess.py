"""
preprocess.py

High-level orchestrator for the DWI preprocessing pipeline.
Consolidates all preprocessing steps into a single function.
"""

from pathlib import Path

import numpy as np

from ..defaults import DEFAULT_B0_THRESHOLD, DEFAULT_DENOISE_METHOD
from ..reproducibility.provenance import get_provenance_dict
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


#: Closed status vocabulary for ledger entries. Anything outside this set is a
#: bug: a reader must be able to switch on it exhaustively.
_LEDGER_STATUSES = (
    "executed",         # csttool ran this stage
    "not_requested",    # optional stage the caller did not ask for
    "failed_continued", # requested, raised, run continued without it
    "declared_external",# user-declared work that preceded csttool
)


def _geometry(data, affine) -> dict:
    """Shape, voxel sizes and axis orientation of an array + affine pair."""
    import nibabel as nib

    affine = np.asarray(affine, dtype=float)
    zooms = np.sqrt(np.sum(affine[:3, :3] ** 2, axis=0))
    return {
        "shape": [int(s) for s in np.shape(data)],
        "zooms": [float(z) for z in zooms],
        "axis_codes": "".join(nib.aff2axcodes(affine)),
    }


def _stage(
    name,
    *,
    status,
    requested=True,
    performed_by="csttool",
    method=None,
    backend=None,
    parameters=None,
    input_geometry=None,
    output_geometry=None,
    gradient_transform_status="none",
    n_volumes_rotated=0,
    max_rotation_deg=None,
    warnings=None,
    skip_reason=None,
) -> dict:
    """One ledger entry.

    Every stage carries the same key set, present even when empty, so a
    consumer never has to test for a key's existence — only for its value.
    Versions deliberately do not appear here: they live once, in the report's
    shared ``provenance`` block.
    """
    if status not in _LEDGER_STATUSES:
        raise ValueError(f"unknown ledger status {status!r}")
    return {
        "stage": name,
        "performed_by": performed_by,
        "requested": requested,
        "status": status,
        "method": method,
        "backend": backend,
        "parameters": parameters or {},
        "input_geometry": input_geometry,
        "output_geometry": output_geometry,
        "gradient_transform": {
            "status": gradient_transform_status,
            "n_volumes_rotated": int(n_volumes_rotated),
            "max_rotation_deg": max_rotation_deg,
        },
        "warnings": list(warnings or []),
        "skip_reason": skip_reason,
    }


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

    # The ledger is built as the run proceeds, so its order *is* the order
    # things happened. Externally declared work is prepended below, because it
    # happened before csttool ever saw the data.
    declaration = declaration_record(external_correction)
    stages: list[dict] = []
    if declaration["declared"] != DEFAULT_EXTERNAL_CORRECTION:
        stages.append(_stage(
            "external_correction",
            performed_by="external",
            status="declared_external",
            requested=False,
            method=declaration["declared"],
            parameters=declaration,
            skip_reason=(
                "Performed outside csttool, before the data was received; "
                "recorded as a user declaration and not verified."
            ),
        ))

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

    loaded_geometry = _geometry(data, affine)
    stages.append(_stage(
        "load",
        status="executed",
        method="nibabel/dicom2nifti + validated gradient table",
        backend="csttool",
        parameters={
            "input_dir": str(input_dir),
            "filename": filename,
            "b0_threshold": b0_threshold,
            "n_volumes": int(len(gtab.bvals)),
            "n_b0": int(np.count_nonzero(gtab.b0s_mask)),
        },
        input_geometry=loaded_geometry,
        output_geometry=loaded_geometry,
    ))

    # -------------------------------------------------------------------------
    # Step 2: Reslice to target voxel size (optional)
    # -------------------------------------------------------------------------
    reslice_input_geometry = _geometry(data, affine)
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

    stages.append(_stage(
        "reslice",
        requested=target_voxel_size is not None,
        status="executed" if target_voxel_size is not None else "not_requested",
        method="trilinear resampling" if target_voxel_size is not None else None,
        backend="dipy" if target_voxel_size is not None else None,
        parameters={"target_voxel_size": list(target_voxel_size)}
        if target_voxel_size is not None else {},
        input_geometry=reslice_input_geometry,
        output_geometry=_geometry(data, affine),
        # Changing voxel size does not move the physical gradient frame.
        gradient_transform_status="not_required",
        skip_reason=None if target_voxel_size is not None
        else "No --target-voxel-size given",
    ))

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

    stages.append(_stage(
        "denoise",
        status="executed",
        method=denoise_method,
        backend="dipy",
        parameters={
            "coil_count": coil_count if denoise_method == "nlmeans" else None,
            "b0_threshold": b0_threshold,
        },
        input_geometry=_geometry(data, affine),
        output_geometry=_geometry(denoised, affine),
    ))

    # -------------------------------------------------------------------------
    # Step 4: Brain masking
    # -------------------------------------------------------------------------
    masked_data, brain_mask = background_segmentation(denoised, gtab)
    print("PREPROCESSING: Brain masking complete")

    stages.append(_stage(
        "mask",
        status="executed",
        method="median_otsu",
        backend="dipy",
        parameters={
            "median_radius": 2,
            "numpass": 1,
            "autocrop": False,
            "b0_volumes_used": int(np.count_nonzero(gtab.b0s_mask)),
        },
        input_geometry=_geometry(denoised, affine),
        output_geometry=_geometry(masked_data, affine),
    ))

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

    stages.append(_stage(
        "gibbs",
        requested=apply_gibbs_correction,
        status="executed" if apply_gibbs_correction else "not_requested",
        method="gibbs_removal (local subvoxel-shift)" if apply_gibbs_correction else None,
        backend="dipy" if apply_gibbs_correction else None,
        input_geometry=_geometry(masked_data, affine),
        output_geometry=_geometry(data_for_motion, affine),
        skip_reason=None if apply_gibbs_correction else "--unring not given",
    ))

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

    n_dwi = int(np.count_nonzero(~gtab.b0s_mask))
    if not apply_motion_correction:
        motion_status = "not_requested"
    elif motion_correction_applied:
        motion_status = "executed"
    else:
        motion_status = "failed_continued"

    stages.append(_stage(
        "motion_correction",
        requested=apply_motion_correction,
        status=motion_status,
        method="register_dwi_series [center_of_mass, translation, rigid, affine]"
        if apply_motion_correction else None,
        backend="dipy" if apply_motion_correction else None,
        parameters={
            "scope": "between-volume affine motion correction only; no "
                     "eddy-current, outlier or susceptibility correction",
            "reference": "mean of registered b0 volumes",
        } if apply_motion_correction else {},
        input_geometry=_geometry(data_for_motion, affine),
        output_geometry=_geometry(preprocessed, affine),
        gradient_transform_status=(
            "bvecs_rotated" if rotated_bvecs is not None
            else "not_required" if not apply_motion_correction
            else "none"
        ),
        n_volumes_rotated=n_dwi if rotated_bvecs is not None else 0,
        max_rotation_deg=max_rotation_deg,
        warnings=warnings,
        skip_reason=None if apply_motion_correction
        else "--perform-motion-correction not given",
    ))

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

    final_geometry = _geometry(preprocessed, affine)
    stages.append(_stage(
        "save",
        status="executed",
        method="NIfTI + FSL-style gradient sidecars",
        backend="nibabel",
        parameters={
            "filename_stem": output_stem,
            "bvec_source": "rotated by csttool" if rotated_bvecs is not None
            else "copied from input",
            "bval_source": "copied from input",
        },
        input_geometry=final_geometry,
        output_geometry=final_geometry,
        gradient_transform_status=(
            "bvecs_rotated" if rotated_bvecs is not None else "none"
        ),
        n_volumes_rotated=n_dwi if rotated_bvecs is not None else 0,
    ))

    output_paths = save_preprocessed(
        data=preprocessed,
        affine=affine,
        output_dir=output_dir,
        filename_stem=output_stem,
        gradient_files=gradient_files if gradient_files else None,
        bvecs=rotated_bvecs,
        brain_mask=brain_mask,
        ledger=stages,
        provenance=get_provenance_dict(),
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
        'bvecs_rotated': rotated_bvecs is not None,
        'max_rotation_deg': max_rotation_deg,
        'warnings': warnings,
        'gtab': gtab,
    }
