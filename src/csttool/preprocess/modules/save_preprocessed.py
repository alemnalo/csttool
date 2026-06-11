"""
save_preprocessed.py

Save preprocessed DWI data to disk.
"""

import shutil
import json
from pathlib import Path
from datetime import datetime

import numpy as np
import nibabel as nib


def save_preprocessed(
    data: np.ndarray,
    affine: np.ndarray,
    output_dir: str | Path,
    filename_stem: str,
    *,
    gradient_files: dict[str, str | Path] | None = None,
    brain_mask: np.ndarray | None = None,
    metadata: dict | None = None,
    processing_params: dict | None = None,
    create_report: bool = True,
) -> dict[str, Path]:
    """
    Save preprocessed DWI data with auxiliary files and metadata.
    
    Parameters
    ----------
    data : np.ndarray
        Preprocessed 4D DWI data array (X, Y, Z, volumes).
    affine : np.ndarray
        4x4 affine transformation matrix from NIfTI header.
    output_dir : str or Path
        Output directory for all files (flat structure).
    filename_stem : str
        Base filename without extension (e.g., "sub01_dwi_preproc").
        
    gradient_files : dict or None, optional
        Dictionary with keys 'bval' and 'bvec' pointing to source files.
        If provided, these will be copied to the output directory.
    brain_mask : np.ndarray or None, optional
        3D binary brain mask to save alongside data.
    metadata : dict or None, optional
        Custom metadata to include in report (e.g., subject ID, session).
    processing_params : dict or None, optional
        Processing parameters used (e.g., denoising method, motion correction).
    create_report : bool, default=True
        Whether to generate a JSON processing report.
    
    Returns
    -------
    output_paths : dict[str, Path]
        Dictionary mapping output types to their absolute paths:
        - 'data': Path to saved preprocessed data
        - 'bval': Path to copied bval file (if provided)
        - 'bvec': Path to copied bvec file (if provided)
        - 'mask': Path to saved brain mask (if provided)
        - 'report': Path to processing report (if created)
    
    Examples
    --------
    >>> paths = save_preprocessed(
    ...     data=preprocessed_data,
    ...     affine=affine,
    ...     output_dir="/data/preprocessed",
    ...     filename_stem="sub01_dwi_preproc",
    ...     gradient_files={"bval": "original.bval", "bvec": "original.bvec"},
    ...     brain_mask=mask,
    ...     processing_params={"denoise_method": "nlmeans", "motion_correction": True}
    ... )
    """
    # Validate inputs
    if not isinstance(data, np.ndarray):
        raise TypeError(f"data must be a numpy array, got {type(data)}")
    if affine.shape != (4, 4):
        raise ValueError(f"affine must be 4x4, got shape {affine.shape}")
    
    # Setup output directory
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    
    output_paths = {}
    
    # Save preprocessed data
    data_path = output_dir / f"{filename_stem}.nii.gz"
    nib.save(nib.Nifti1Image(data, affine), data_path)
    output_paths['data'] = data_path
    print(f"  ✓ Saved preprocessed data: {data_path}")
    
    # Copy gradient files
    if gradient_files is not None:
        for grad_type in ['bval', 'bvec']:
            if grad_type in gradient_files:
                src = Path(gradient_files[grad_type])
                if src.exists():
                    dest = output_dir / f"{filename_stem}.{grad_type}"
                    shutil.copy2(src, dest)
                    output_paths[grad_type] = dest
                    print(f"  ✓ Copied {grad_type}: {dest}")
                else:
                    print(f"  ⚠️ {grad_type} file not found: {src}")
    
    # Save brain mask
    if brain_mask is not None:
        mask_path = output_dir / f"{filename_stem}_mask.nii.gz"
        nib.save(nib.Nifti1Image(brain_mask.astype(np.uint8), affine), mask_path)
        output_paths['mask'] = mask_path
        print(f"  ✓ Saved brain mask: {mask_path}")
    
    # Create processing report
    if create_report:
        report = {
            'timestamp': datetime.now().isoformat(),
            'filename_stem': filename_stem,
            'data_shape': list(data.shape),
            'data_dtype': str(data.dtype),
            'voxel_size': np.sqrt(np.sum(affine[:3, :3]**2, axis=0)).tolist(),
        }
        
        if processing_params is not None:
            report['processing_params'] = processing_params
        
        if metadata is not None:
            report['metadata'] = metadata
        
        report_path = output_dir / f"{filename_stem}_report.json"
        with open(report_path, 'w') as f:
            json.dump(report, f, indent=2)
        output_paths['report'] = report_path
        print(f"  ✓ Saved processing report: {report_path}")
    
    return output_paths