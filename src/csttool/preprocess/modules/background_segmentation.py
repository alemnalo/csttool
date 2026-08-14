"""
background_segmentation.py

Estimate brain mask with median Otsu.
https://docs.dipy.org/dev/examples_built/segmentation/brain_extraction_dwi_1.html#sphx-glr-examples-built-segmentation-brain-extraction-dwi-1-py
"""

import numpy as np
from dipy.segment.mask import median_otsu


def background_segmentation(
    data: np.ndarray,
    gtab=None,
    median_radius: int = 2,
    numpass: int = 1,
    autocrop: bool = False
) -> tuple[np.ndarray, np.ndarray]:
    """
    Estimate brain mask with median Otsu.

    Parameters
    ----------
    data : np.ndarray
        4D DWI data array.
    gtab : GradientTable, optional
        Gradient table to identify b0 volumes. If provided, only the volumes
        marked by ``gtab.b0s_mask`` are used for mask computation. If None,
        only the first volume is used (for 4D input).
    median_radius : int, optional
        Radius of the median filter. Default is 2.
    numpass : int, optional
        Number of passes for the median filter. Default is 1.
    autocrop : bool, optional
        Whether to autocrop the data. Default is False.

    Returns
    -------
    masked_data : np.ndarray
        4D masked DWI data array.
    mask : np.ndarray
        3D binary brain mask array.
    """
    # Determine which volumes to use for mask computation
    vol_idx = None
    if gtab is not None:
        # Read the gradient table's own b0 partition rather than re-deriving it
        # from a module-level constant: the table was built with the b0
        # threshold this execution was given (--b0-threshold), so re-thresholding
        # here made masking disagree with the tensor fit whenever the threshold
        # was not the default. One table, one answer.
        b0_idx = np.where(gtab.b0s_mask)[0]
        if b0_idx.size > 0:
            vol_idx = b0_idx
    else:
        # If no gtab provided, default to using the first volume
        if data.ndim == 4:
            vol_idx = [0]
    
    masked_data, mask = median_otsu(
        data,
        vol_idx=vol_idx,
        median_radius=median_radius,
        numpass=numpass,
        autocrop=autocrop
    )

    return masked_data, mask