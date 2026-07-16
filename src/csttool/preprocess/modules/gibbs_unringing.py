"""
gibbs_unringing.py

Remove Gibbs' ringing artifacts from DWI data.
https://docs.dipy.org/dev/examples_built/preprocessing/denoise_gibbs.html
"""

import numpy as np
from dipy.denoise.gibbs import gibbs_removal


def gibbs_unringing(
    data: np.ndarray,
    slice_axis: int = 2,
    n_points: int = 3
) -> np.ndarray:
    """
    Remove Gibbs' ringing artifacts from DWI data.
    
    Gibbs ringing artifacts appear as spurious oscillations near sharp edges
    in MR images due to truncation of k-space data.

    Parameters
    ----------
    data : np.ndarray
        3D or 4D DWI data array.
    slice_axis : int, optional
        Axis along which slices were acquired (0, 1, or 2). Default is 2.
    n_points : int, optional
        Number of neighbor points to access local TV. Default is 3.

    Returns
    -------
    data_corrected : np.ndarray
        3D or 4D DWI data array with Gibbs ringing artifacts removed.
    """
    if slice_axis not in [0, 1, 2]:
        raise ValueError(f"slice_axis must be 0, 1, or 2, got {slice_axis}")
    
    data_corrected = gibbs_removal(
        data,
        slice_axis=slice_axis,
        n_points=n_points,
        num_processes=1  # Use a single process to avoid non-determinism from parallel processing
    )
    return data_corrected
    