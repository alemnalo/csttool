def load_and_mask(nii_dirname, nii_fname, visualize=False, verbose=False,
                  brain_mask_path=None):
    """This function loads a NIfTI dataset with its gradient table, then applies median Otsu threshold segmentation to generate a brainmask.

    When ``brain_mask_path`` is supplied, that mask replaces the automatic
    segmentation entirely: ``background_segmentation`` is not called.

    Args:
        nii_dirname (str): Directory of your NIfTI file.
        nii_fname (str): Name of your NIfTI file.
        visualize (bool): Set to True for data visualization. Defaults to False.
        verbose (bool): Set to True for verbose output. Defaults to False.
        brain_mask_path (str | Path | None): Optional external DWI-space brain
            mask to use instead of automatic background segmentation.
            Defaults to None (automatic).

    Returns:
        tuple: (data, affine, img, gtab, masked_data, brain_mask)
            - data: 4D DWI array (X, Y, Z, N)
            - affine: 4x4 transformation matrix
            - img: nibabel Nifti1Image object
            - gtab: DIPY GradientTable
            - masked_data: Brain-masked DWI data
            - brain_mask: Binary brain mask array
    """    
    from csttool.preprocess.modules.load_dataset import load_dataset
    from csttool.tracking.modules.brain_mask import resolve_brain_mask

    # Remove extension if present in fname, as modules.load_dataset expects stem or handles it differently
    # Actually modules.load_dataset expects fname without extension for nifti construction in some paths,
    # but let's check how it uses it.
    # It constructs: os.path.join(dir_path, fname + ".nii.gz")
    # So we should pass fname WITHOUT extension if it's not there.
    # The original caller likely passed it without extension based on usage in cli.py or tracking.
    
    # modules.load_dataset returns: nii, gtab, nifti_dir, metadata
    nii, gtab, _, _ = load_dataset(
        dir_path=nii_dirname,
        fname=nii_fname
    )
    
    data = nii.get_fdata()
    affine = nii.affine
    img = nii

    if verbose:
        print(f"  → Loaded dataset: {nii_dirname}/{nii_fname}")
        print(f"    • Data shape: {data.shape}")
        print(f"    • Gradient table: {len(gtab.bvals)} volumes")

    # Automatic path is background_segmentation(data, gtab), exactly as before;
    # an external mask short-circuits it (see brain_mask.resolve_brain_mask).
    # The old inline call passed visualize=visualize, but the current
    # background_segmentation signature has no visualize argument.
    masked_data, brain_mask, _mask_info = resolve_brain_mask(
        data,
        gtab,
        affine,
        brain_mask_path=brain_mask_path,
        verbose=verbose,
    )

    return data, affine, img, gtab, masked_data, brain_mask
