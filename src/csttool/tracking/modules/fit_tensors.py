def select_dti_volumes(data, gtab, *, max_b=1200.0, shell_tolerance=50.0):
    """Select b0 and low-b observations supported by the DTI signal model."""
    import warnings

    import numpy as np
    from dipy.core.gradients import gradient_table

    from .estimate_directions import _cluster_dwi_shells

    data = np.asanyarray(data)
    if data.shape[-1] != len(gtab.bvals):
        raise ValueError(
            f"DWI has {data.shape[-1]} volumes but gradient table has "
            f"{len(gtab.bvals)}."
        )
    b0_mask = np.asarray(gtab.b0s_mask, dtype=bool)
    dwi_mask = ~b0_mask
    low_b_mask = dwi_mask & (np.asarray(gtab.bvals) <= float(max_b))
    selection = b0_mask | low_b_mask
    fallback = None

    if int(low_b_mask.sum()) < 6:
        shells = _cluster_dwi_shells(gtab, tolerance=shell_tolerance)
        eligible = [shell for shell in shells if len(shell["indices"]) >= 6]
        if eligible:
            selected_shell = min(eligible, key=lambda shell: shell["center"])
            selection = b0_mask.copy()
            selection[selected_shell["indices"]] = True
            fallback = "lowest shell because fewer than six DWI volumes were <= max_b"
            warnings.warn(
                "Fewer than six low-b observations are available for DTI; "
                f"using the lowest shell at b={selected_shell['center']:.0f} "
                "s/mm².",
                UserWarning,
                stacklevel=2,
            )
        else:
            selection = np.ones(len(gtab.bvals), dtype=bool)
            fallback = "all volumes because no shell contains six observations"

    selected_bvals = np.asarray(gtab.bvals)[selection]
    selected_bvecs = np.asarray(gtab.bvecs)[selection]
    selected_gtab = gradient_table(
        bvals=selected_bvals,
        bvecs=selected_bvecs,
        b0_threshold=float(getattr(gtab, "b0_threshold", 50.0)),
    )
    selected_data = data if selection.all() else data[..., selection]
    selected_dwi = selected_bvals[~np.asarray(selected_gtab.b0s_mask)]
    metadata = {
        "model": "DTI",
        "fit_method_scope": "b0 plus low-b observations",
        "selection_rule": f"b0 plus 0 < b <= {float(max_b):g} s/mm2",
        "maximum_supported_b": float(max_b),
        "input_volume_count": int(len(gtab.bvals)),
        "fit_volume_count": int(selection.sum()),
        "b0_count": int(np.asarray(selected_gtab.b0s_mask).sum()),
        "dwi_count": int((~np.asarray(selected_gtab.b0s_mask)).sum()),
        "maximum_fitted_b": (
            float(selected_dwi.max()) if selected_dwi.size else None
        ),
        "fallback": fallback,
    }
    return selected_data, selected_gtab, metadata


def fit_tensors(data, gtab, brain_mask, fa_thresh=0.2, visualize=False,
                verbose=False, fit_method='WLS', return_metadata=False,
                max_b=1200.0, create_white_matter=True):
    """Fit low-b DTI and compute scalar maps.

    ``create_white_matter`` preserves the legacy FA-threshold/dilation product
    for whole-brain tracking. Preferred PLIC mode sets it false because FA does
    not seed, support, stop, or filter that execution path.

    Returns ``(tenfit, fa, md, rd, ad, white_matter)`` plus metadata when
    requested. ``white_matter`` is ``None`` when ``create_white_matter`` is
    false.
    """
    from dipy.reconst.dti import TensorModel, mean_diffusivity, radial_diffusivity, axial_diffusivity
    from scipy.ndimage import binary_dilation
    import numpy as np
    import matplotlib.pyplot as plt
    
    # DTI assumes mono-exponential low-b decay. Fitting high-b shells biases
    # the tensor and every scalar derived from it.
    fit_data, fit_gtab, fit_metadata = select_dti_volumes(
        data, gtab, max_b=max_b
    )
    fit_metadata["fit_method"] = fit_method
    if verbose:
        print("  → Fitting tensor model...")
        print(
            f"    • Volumes: {fit_metadata['fit_volume_count']} / "
            f"{fit_metadata['input_volume_count']} "
            f"(maximum b={fit_metadata['maximum_fitted_b']})"
        )

    tenmodel = TensorModel(fit_gtab, fit_method=fit_method)
    tenfit = tenmodel.fit(fit_data, mask=brain_mask)

    # Persisted scalar images remain finite for broad NIfTI compatibility.
    # The pre-sanitization fit-validity mask is computed from ``tenfit`` by
    # ``tensor_validity_mask`` and saved separately; metrics must use that mask
    # rather than treating replacement zeros as valid measurements.
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
    
    white_matter = None
    if create_white_matter:
        white_matter = (fa > fa_thresh) & brain_mask
        wm_before_dilation = white_matter.sum()
        white_matter = binary_dilation(white_matter, iterations=1)
        wm_after_dilation = white_matter.sum()

        if wm_after_dilation == 0:
            import warnings
            warnings.warn(
                "No white-matter voxels found (FA <= "+str(fa_thresh)+" everywhere). "
                "The DWI data may be all-zero or have no diffusion contrast; "
                "tractography will produce zero streamlines.",
                UserWarning,
                stacklevel=2,
            )

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
        
        axes[1, 0].imshow(fa[:, :, mid_slice].T, cmap='gray', vmin=0, vmax=1, origin='lower')
        axes[1, 0].set_title('FA Map')
        axes[1, 0].axis('off')

        axes[1, 1].imshow(md[:, :, mid_slice].T, cmap='hot', origin='lower')
        axes[1, 1].set_title('MD Map')
        axes[1, 1].axis('off')

        if create_white_matter:
            axes[1, 2].imshow(fa[:, :, mid_slice].T, cmap='gray', vmin=0, vmax=1, origin='lower')
            axes[1, 2].imshow(white_matter[:, :, mid_slice].T, cmap='Blues', alpha=0.5, origin='lower')
            axes[1, 2].set_title(f'White Matter Mask\n({int(white_matter.sum()):,} voxels)')
        else:
            axes[1, 2].imshow(brain_mask[:, :, mid_slice].T, cmap='Blues', origin='lower')
            axes[1, 2].set_title('Tensor-fit domain')
        axes[1, 2].axis('off')

        plt.tight_layout()
        plt.show()

    result = (tenfit, fa, md, rd, ad, white_matter)
    if return_metadata:
        return (*result, fit_metadata)
    return result


def tensor_validity_mask(tenfit, brain_mask):
    """Return voxels with a finite, mathematically valid tensor solution.

    DIPY orders eigenvalues descending. Negative, non-finite, misordered or
    zero-trace (degenerate/no fitted diffusion) tensors and
    non-finite/out-of-range FA are invalid. No empirical diffusivity cutoff is
    applied.
    """
    import numpy as np

    evals = np.asarray(tenfit.evals)
    fa = np.asarray(tenfit.fa)
    valid = np.asarray(brain_mask, dtype=bool).copy()
    valid &= np.isfinite(evals).all(axis=-1)
    valid &= np.isfinite(fa)
    valid &= (fa >= 0.0) & (fa <= 1.0)
    valid &= (evals >= 0.0).all(axis=-1)
    valid &= evals.sum(axis=-1) > 0.0
    valid &= evals[..., 0] >= evals[..., 1]
    valid &= evals[..., 1] >= evals[..., 2]
    return valid