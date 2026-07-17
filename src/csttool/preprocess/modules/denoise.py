"""
denoise.py

Denoise DWI data using one of three methods:
- NLMeans (Non-Local Means) — requires coil count for PIESNO sigma estimation
- Patch2Self — self-supervised, requires bvals
- MPPCA (Marchenko-Pastur PCA) — self-adaptive, requires neither N nor bvals
"""

import numpy as np
from csttool.defaults import DEFAULT_B0_THRESHOLD
from dipy.denoise.noise_estimate import piesno
from dipy.denoise.nlmeans import nlmeans
from dipy.denoise.patch2self import patch2self

from ...defaults import DEFAULT_DENOISE_METHOD


def denoise(
    data: np.ndarray,
    bvals: np.ndarray = None,
    brain_mask: np.ndarray = None,
    denoise_method: str = DEFAULT_DENOISE_METHOD,
    N: int = 4,
    patch_radius: int = 2,
):
    """
    Denoise DWI data.

    Parameters
    ----------
    data : np.ndarray
        4D DWI data array.
    bvals : np.ndarray
        1D array of b values. Required for Patch2Self; ignored for
        NLMeans and MPPCA.
    brain_mask : np.ndarray
        3D brain mask array. Used by NLMeans (restricts denoise region;
        PIESNO fallback if None) and MPPCA (restricts PCA neighbourhood
        computation). Ignored for Patch2Self.
    denoise_method : str
        Denoising method to use. One of:

        - ``"nlmeans"`` — Non-local means (Veraart et al. 2016).
          Requires *N* (coil count). Uses PIESNO for sigma estimation.
        - ``"patch2self"`` — Self-supervised denoising (Fadnavis et al.
          2020). Requires *bvals*.
        - ``"mppca"`` — Marchenko-Pastur PCA (Veraart et al. 2016).
          Self-adaptive; does NOT require *N* or *bvals*.

    N : int
        Number of scanner receiver coils. Used only by NLMeans for PIESNO
        noise estimation. Ignored by Patch2Self and MPPCA.
        Default: 4.

        .. warning:: Must match the actual acquisition hardware. A wrong
           *N* biases sigma → biases NLMeans shrinkage → biases all
           downstream DTI metrics. For modern multi-channel head coils
           (e.g. 32/64-channel) with adaptive combine readout this
           parameter may not be meaningful; prefer MPPCA in that case.

    patch_radius : int
        Patch radius for MPPCA local-PCA neighbourhood. Default 2 gives
        a 7×7×7 voxel patch (343 voxels). Larger patches increase
        smoothing but reduce spatial specificity. Ignored by NLMeans
        and Patch2Self.

    Returns
    -------
    denoised_data : np.ndarray
        4D denoised DWI data array.

    References
    ----------
    .. [1] Veraart et al., "Denoising-based nonlinear averaging improves
       resolution limited diffusion MRI and DTI". NeuroImage 125, 873-886
       (2016). [NLMeans]
    .. [2] Fadnavis et al., "Patch2Self: Denoising Diffusion MRI with
       Self-Supervised Learning". NeuroImage 223, 117389 (2020).
    .. [3] Veraart et al., "Denoising of diffusion MRI using random matrix
       theory". NeuroImage 134, 387-399 (2016). [MPPCA]
    """

    available_methods = ["nlmeans", "patch2self", "mppca"]
    if denoise_method not in available_methods:
        raise ValueError(
            f"Invalid denoise method: {denoise_method}. "
            f"Available methods: {available_methods}"
        )

    # ==================================================================
    # NLMeans — Non-Local Means denoising
    # Requires : N (coil count for PIESNO sigma estimation)
    # Optional : brain_mask (restricts denoise region)
    # ==================================================================
    if denoise_method == "nlmeans":
        print("Denoising with NLM...")
        noise, noise_mask = piesno(data, N=N, return_mask=True)
        sigma = float(np.mean(noise))  # noise std deviation
        if brain_mask is None:
            print("  ⚠️  Brain mask is None, using noise mask as rudimentary brain mask")
            brain_mask = ~noise_mask
        denoised_data = nlmeans(
            data.astype(np.float32),
            sigma=sigma,
            mask=brain_mask,
            patch_radius=1,
            block_radius=2,
            rician=False,  # Assuming Gaussian noise
            num_threads=1,  # Multithreading introduces non-determinism; use single thread for reproducibility
        )

    # ==================================================================
    # Patch2Self — self-supervised denoising
    # Requires : bvals (uses diffusion signal redundancy across volumes)
    # ==================================================================
    elif denoise_method == "patch2self":
        if bvals is None:
            raise ValueError(
                "Patch2Self requires bvals. Pass the bvals array or use "
                "a different denoise method (nlmeans or mppca)."
            )
        print("Denoising with Patch2Self...")
        denoised_data = patch2self(
            data,
            bvals=bvals,
            model="ols",
            shift_intensity=True,
            clip_negative_vals=False,
            b0_threshold=DEFAULT_B0_THRESHOLD,
        )

    # ==================================================================
    # MPPCA — Marchenko-Pastur PCA denoising
    # Self-adaptive: estimates the noise level from the eigenvalue
    # distribution of local-PCA patches, using the Marchenko-Pastur
    # distribution from random matrix theory.  Does NOT require the
    # number of coils (N) or bvals — it exploits the redundancy of the
    # multi-volume DWI directly.
    #
    # The mask parameter (if provided) limits which voxels are included
    # in the local PCA computation, reducing memory and runtime on
    # large volumes with mostly-background voxels.
    #
    # Reference: Veraart et al., "Denoising of diffusion MRI using
    # random matrix theory", NeuroImage 134, 387-399 (2016).
    # ==================================================================
    elif denoise_method == "mppca":
        from dipy.denoise.localpca import mppca as _mppca

        print(f"Denoising with MPPCA (patch_radius={patch_radius})...")

        if data.ndim != 4:
            raise ValueError(
                f"MPPCA requires 4D data, got {data.ndim}D"
            )

        denoised_data = _mppca(
            data,
            mask=brain_mask,
            patch_radius=patch_radius,
            pca_method="eig",  # Eigenvalue decomposition (faster than SVD for small patches)
            return_sigma=False,
        )

    return denoised_data
