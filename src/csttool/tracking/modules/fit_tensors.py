def fit_tensors(data, gtab, brain_mask, fa_thresh=0.2, visualize=False, verbose=False, fit_method='WLS'):
    """Fit diffusion tensor model and compute scalar maps (FA, MD, RD, AD) plus white matter mask.
    
    Args:
        data (ndarray): 4D DWI data array (X, Y, Z, N).
        gtab (GradientTable): DIPY gradient table.
        brain_mask (ndarray): Binary brain mask.
        fa_thresh (float): FA threshold for white matter mask (default 0.2).
        visualize (bool): Show QC plots.
        verbose (bool): Print processing details.
        fit_method (str): Fitting method for the tensor model (default 'WLS').
    Returns:
        tuple: (tenfit, fa, md, rd, ad, white_matter)
            - tenfit: Fitted TensorModel object
            - fa: Fractional anisotropy map (X, Y, Z)
            - md: Mean diffusivity map (X, Y, Z)
            - rd: Radial diffusivity map (X, Y, Z) = (λ₂ + λ₃) / 2
            - ad: Axial diffusivity map (X, Y, Z) = λ₁
            - white_matter: Binary white matter mask (dilated)
    """
    from dipy.reconst.dti import TensorModel, mean_diffusivity, radial_diffusivity, axial_diffusivity
    from scipy.ndimage import binary_dilation
    import numpy as np
    import matplotlib.pyplot as plt
    
    # Fit tensor model
    if verbose:
        print("  → Fitting tensor model...")

    tenmodel = TensorModel(gtab, fit_method=fit_method)
    tenfit = tenmodel.fit(data, mask=brain_mask)

    # Compute scalar maps with NaN handling
    # Current approach: set all implausible values to 0. Revise if results not satisfactory.
    fa = np.nan_to_num(tenfit.fa, nan=0.0, posinf=0.0, neginf=0.0)
    md = np.nan_to_num(mean_diffusivity(tenfit.evals), nan=0.0, posinf=0.0, neginf=0.0)
    rd = np.nan_to_num(radial_diffusivity(tenfit.evals), nan=0.0, posinf=0.0, neginf=0.0)
    ad = np.nan_to_num(axial_diffusivity(tenfit.evals), nan=0.0, posinf=0.0, neginf=0.0)

    if verbose:
        fa_brain = fa[brain_mask > 0]
        md_brain = md[brain_mask > 0]
        rd_brain = rd[brain_mask > 0]
        ad_brain = ad[brain_mask > 0]
        print("    • Tensor metrics:")
        print(f"    ├─ FA mean: {fa_brain.mean():.3f}, std: {fa_brain.std():.3f}")
        print(f"    ├─ MD mean: {md_brain.mean():.2e}, std: {md_brain.std():.2e}")
        print(f"    ├─ RD mean: {rd_brain.mean():.2e}, std: {rd_brain.std():.2e}")
        print(f"    └─ AD mean: {ad_brain.mean():.2e}, std: {ad_brain.std():.2e}")
    
    # Create white matter mask
    white_matter = (fa > fa_thresh) & brain_mask
    wm_before_dilation = white_matter.sum()

    # Dilate to reach cortical grey matter interface
    white_matter = binary_dilation(white_matter, iterations=1)
    wm_after_dilation = white_matter.sum()

    if verbose:
        print(f"    • White matter (FA > {fa_thresh}):")
        print(f"    ├─ Before dilation: {wm_before_dilation:,} voxels")
        print(f"    └─ After dilation:  {wm_after_dilation:,} voxels (+{wm_after_dilation - wm_before_dilation:,})")
    
    # Visualization
    if visualize:
        fig, axes = plt.subplots(2, 3, figsize=(12, 8))
        fig.suptitle("Tensor Fitting Results", fontsize=14)
        
        mid_slice = data.shape[2] // 2
        b0_slice = data[:, :, mid_slice, 0]
        
        # Row 1: Brain mask
        axes[0, 0].imshow(b0_slice.T, cmap='gray', origin='lower')
        axes[0, 0].set_title('Original b0')
        axes[0, 0].axis('off')
        
        axes[0, 1].imshow(brain_mask[:, :, mid_slice].T, cmap='gray', origin='lower')
        axes[0, 1].set_title(f'Brain Mask\n({brain_mask.sum():,} voxels)')
        axes[0, 1].axis('off')
        
        axes[0, 2].imshow(b0_slice.T, cmap='gray', origin='lower')
        axes[0, 2].imshow(brain_mask[:, :, mid_slice].T, cmap='Reds', alpha=0.5, origin='lower')
        axes[0, 2].set_title('Brain Mask Overlay')
        axes[0, 2].axis('off')
        
        # Row 2: FA, MD, White matter
        axes[1, 0].imshow(fa[:, :, mid_slice].T, cmap='gray', vmin=0, vmax=1, origin='lower')
        axes[1, 0].set_title('FA Map')
        axes[1, 0].axis('off')
        
        axes[1, 1].imshow(md[:, :, mid_slice].T, cmap='hot', origin='lower')
        axes[1, 1].set_title('MD Map')
        axes[1, 1].axis('off')
        
        axes[1, 2].imshow(fa[:, :, mid_slice].T, cmap='gray', vmin=0, vmax=1, origin='lower')
        axes[1, 2].imshow(white_matter[:, :, mid_slice].T, cmap='Blues', alpha=0.5, origin='lower')
        axes[1, 2].set_title(f'White Matter Mask\n({wm_after_dilation:,} voxels)')
        axes[1, 2].axis('off')
        
        plt.tight_layout()
        plt.show()
    
    return tenfit, fa, md, rd, ad, white_matter