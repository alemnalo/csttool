"""
perform_motion_correction.py

Perform between-volume motion correction.
https://docs.dipy.org/dev/examples_built/preprocessing/motion_correction.html
"""

import numpy as np
from dipy.align import motion_correction

def perform_motion_correction(
    data: np.ndarray,
    gtab,
    affine: np.ndarray,
    brain_mask: np.ndarray | None = None
) -> tuple[np.ndarray, list]:
    """
    Perform between-volume motion correction.
    
    Parameters
    ----------
    data : np.ndarray
        4D DWI data array.
    gtab : GradientTable
        Gradient table containing b-values and b-vectors.
    affine : np.ndarray
        4x4 affine transformation matrix.
    brain_mask : np.ndarray or None, optional
        Binary brain mask to constrain registration. If provided,
        will be converted to uint8 and passed as static_mask.
    
    Returns
    -------
    data_corrected : np.ndarray
        4D DWI data array with motion correction applied.
    reg_affines : list
        List of 4x4 registration affine matrices for each volume.
    """
    # Ensure mask is binary uint8 if provided
    if brain_mask is not None:
        brain_mask = brain_mask.astype(np.float64)
        data_corrected, reg_affines = motion_correction(
            data,
            gtab,
            affine=affine,
            static_mask=brain_mask
        )
    else:
        data_corrected, reg_affines = motion_correction(
            data,
            gtab,
            affine=affine
        )

    # dipy.align.motion_correction returns a Nifti1Image; extract the array
    if hasattr(data_corrected, 'get_fdata'):
        data_corrected = data_corrected.get_fdata(dtype=np.float32)

    return data_corrected, reg_affines