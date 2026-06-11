"""
denoise.py

Denoise DWI data.
"""

import numpy as np
from dipy.denoise.noise_estimate import piesno
from dipy.denoise.nlmeans import nlmeans
from dipy.denoise.patch2self import patch2self

def denoise(
    data: np.ndarray, 
    bvals: np.ndarray = None, 
    brain_mask: np.ndarray = None, 
    denoise_method: str = "nlmeans",
    N: int = 4
):
    """
    Denoise DWI data.

    Parameters
    ----------
    data : np.ndarray
        4D DWI data array.
    bvals : np.ndarray
        1D array of b values.
    brain_mask : np.ndarray
        3D brain mask array.
    denoise_method : str
        Denoising method to use. Can be "nlmeans" or "patch2self".
    N : int
        Number of scanner head coils used for acquisition, needed for NLM.

    Returns
    -------
    denoised_data : np.ndarray
        4D denoised DWI data array.
    """
    
    available_methods = ["nlmeans", "patch2self"]
    if denoise_method not in available_methods:
        raise ValueError(f"Invalid denoise method: {denoise_method}. Available methods: {available_methods}")

    # Denoise with NLM
    # https://docs.dipy.org/dev/examples_built/preprocessing/denoise_nlmeans.html
    if denoise_method == "nlmeans":
        print("Denoising with NLM...")
        noise, noise_mask = piesno(data, N=N, return_mask=True)
        sigma = float(np.mean(noise))  # Calculate the noise standard deviation
        if brain_mask is None:
            print("  ⚠️ Brain mask is None, using noise mask as rudimentary brain mask")
            brain_mask = ~noise_mask  # Invert the noise mask as rudimentary brain mask
        denoised_data = nlmeans(
            data.astype(np.float32),
            sigma=sigma,
            mask=brain_mask,
            patch_radius=1,
            block_radius=2,
            rician=False,
            num_threads=-1
        )

    # Denoise with Patch2Self
    # Requires bvals
    # https://docs.dipy.org/dev/examples_built/preprocessing/denoise_patch2self.html
    elif denoise_method == "patch2self":
        print("Denoising with Patch2Self...")
        denoised_data = patch2self(
            data,
            bvals=bvals,
            model="ols",
            shift_intensity=True,
            clip_negative_vals=False,
            b0_threshold=50,
        )

    return denoised_data