
import argparse
from pathlib import Path
import nibabel as nib
from dipy.io.streamline import load_tractogram
from dipy.io.image import load_nifti
from dipy.io import read_bvals_bvecs
from dipy.core.gradients import gradient_table

from csttool.reproducibility import RunContext
from csttool.viz import geometry as _viz_geometry

def cmd_extract(args: argparse.Namespace) -> dict | None:
    """
    Extract bilateral CST using atlas-based ROI filtering.

    Supports two methods:
    - endpoint: Filter by streamline endpoints (original)
    - passthrough: Filter by streamlines passing through ROIs (more permissive)

    Note: roi-seeded method requires raw DWI data and is only available via cmd_run.
    """

    verbose = getattr(args, 'verbose', True)
    quiet = getattr(args, 'quiet', False)
    skip_coord_validation = getattr(args, 'skip_coordinate_validation', False)

    # Quiet overrides verbose
    if quiet:
        verbose = False
    
    # Check extraction method
    extraction_method = getattr(args, 'extraction_method', 'passthrough')
    
    if extraction_method in ("roi-seeded", "bidirectional"):
        print(f"  ✗ Error: {extraction_method} method requires raw DWI data")
        print(f"    Use 'csttool run --extraction-method {extraction_method}'")
        print("    or use --extraction-method endpoint|passthrough with csttool extract")
        return None

    # Validate inputs
    if not args.tractogram.exists():
        print(f"  ✗ Tractogram not found: {args.tractogram}")
        return None

    if not args.fa.exists():
        print(f"  ✗ FA map not found: {args.fa}")
        return None
    
    args.out.mkdir(parents=True, exist_ok=True)
    
    # Import extraction modules
    try:
        from csttool.extract.modules.registration import register_mni_to_subject
        from csttool.extract.modules.warp_atlas_to_subject import (
            warp_harvard_oxford_to_subject,
            CST_ROI_CONFIG
        )
        from csttool.extract.modules.create_roi_masks import create_cst_roi_masks
        from csttool.extract.modules.endpoint_filtering import (
            extract_bilateral_cst,
            save_cst_tractograms,
            save_extraction_report
        )
        from csttool.extract.modules.passthrough_filtering import extract_cst_passthrough, sample_peduncle_fa
    except ImportError as e:
        print(f"  ✗ Error importing extraction modules: {e}")
        return None

    # Load inputs
    if not quiet:
        print(f"  → Loading tractogram: {args.tractogram}")

    # Coordinate validation (critical check for silent failure prevention)
    if not skip_coord_validation:
        try:
            from csttool.extract.modules.coordinate_validation import validate_tractogram_coordinates
            validation = validate_tractogram_coordinates(
                tractogram_path=str(args.tractogram),
                reference_path=str(args.fa),
                strict=True,
                verbose=verbose
            )
        except ValueError as e:
            print(f"  ✗ {e}")
            return None
        except Exception as e:
            print(f"  ⚠️ Coordinate validation failed unexpectedly: {e}")
            print("    Proceeding with extraction (use --skip-coordinate-validation to suppress)")
    elif verbose:
        print("    • Skipping coordinate validation (--skip-coordinate-validation)")

    try:
        sft = load_tractogram(str(args.tractogram), 'same')
        streamlines = sft.streamlines
        if not quiet:
            print(f"  ✓ Loaded: {len(streamlines):,} streamlines")
    except Exception as e:
        print(f"  ✗ Error loading tractogram: {e}")
        return None

    if not quiet:
        print(f"  → Loading FA map: {args.fa}")
    try:
        fa_data, fa_affine = load_nifti(str(args.fa))
    except Exception as e:
        print(f"  ✗ Error loading FA map: {e}")
        return None
    
    # Section header
    print("=" * 60)
    print("CST EXTRACTION")
    print("=" * 60)

    # Step 1: Registration
    print(f"\n[Step 1/5] Registering MNI template to subject space...")
    
    level_iters_affine = [1000, 100, 10] if args.fast_registration else [10000, 1000, 100]
    level_iters_syn = [5, 5, 3] if args.fast_registration else [10, 10, 5]
    
    try:
        reg_result = register_mni_to_subject(
            subject_fa_path=args.fa,
            output_dir=args.out,
            level_iters_affine=level_iters_affine,
            level_iters_syn=level_iters_syn,
            verbose=verbose
        )
    except Exception as e:
        print(f"  ✗ Registration failed: {e}")
        return None

    # Step 2: Warp atlases
    print(f"\n[Step 2/5] Warping Harvard-Oxford atlases to subject space...")
    
    try:
        warped = warp_harvard_oxford_to_subject(
            registration_result=reg_result,
            output_dir=args.out,
            subject_id=args.subject_id,
            verbose=verbose
        )
    except Exception as e:
        print(f"  ✗ Warping atlases failed: {e}")
        return None

    # Step 3: Create ROI masks
    print(f"\n[Step 3/5] Creating CST ROI masks...")
    
    try:
        masks = create_cst_roi_masks(
            warped_cortical=warped['cortical_warped'],
            warped_subcortical=warped['subcortical_warped'],
            subject_affine=warped['subject_affine'],
            roi_config=CST_ROI_CONFIG,
            dilate_brainstem=args.dilate_brainstem,
            dilate_motor=args.dilate_motor,
            output_dir=args.out,
            subject_id=args.subject_id,
            verbose=verbose,
            original_subject_affine=warped.get('original_subject_affine'),
            reorientation_transform=warped.get('reorientation_transform'),
            hemisphere_mask=warped.get('hemisphere_mask'),
            midline_x=warped.get('midline_x'),
        )
    except Exception as e:
        print(f"  ✗ Creating ROI masks failed: {e}")
        return None

    # Step 4: Extract CST
    print(f"\n[Step 4/5] Extracting bilateral CST (method: {extraction_method})...")
    
    try:
        if extraction_method == "passthrough":
            cst_result = extract_cst_passthrough(
                streamlines=streamlines,
                masks=masks,
                affine=warped['subject_affine'],  # Use affine that matches the ROI masks
                min_length=args.min_length,
                max_length=args.max_length,
                midline_x=warped.get('midline_x', 0.0),
                midline_distance=warped.get('midline_distance'),
                verbose=verbose
            )
            # Peduncle FA diagnostic (to check for tracking quality asymmetry)
            if verbose:
                sample_peduncle_fa(
                    fa_map=fa_data,
                    fa_affine=fa_affine,
                    brainstem_mask=masks['brainstem'],
                    mask_affine=warped['subject_affine'],
                    hemisphere_mask=warped.get('hemisphere_mask'),
                    midline_x=warped.get('midline_x'),
                    verbose=True
                )
        else:  # "endpoint"
            cst_result = extract_bilateral_cst(
                streamlines=streamlines,
                masks=masks,
                affine=warped['subject_affine'],  # Use affine that matches the ROI masks
                min_length=args.min_length,
                max_length=args.max_length,
                verbose=verbose
            )
    except Exception as e:
        print(f"  ✗ CST extraction failed: {e}")
        return None

    # Step 5: Save outputs
    print(f"\n[Step 5/5] Saving extracted tractograms...")
    
    try:
        reference_img = nib.load(str(args.fa))
        output_paths = save_cst_tractograms(
            cst_result=cst_result,
            reference_img=reference_img,
            output_dir=args.out,
            subject_id=args.subject_id,
            verbose=verbose
        )
        
        save_extraction_report(cst_result, output_paths, args.out, args.subject_id)
    except Exception as e:
        print(f"  ✗ Saving outputs failed: {e}")
        return None
    
    if getattr(args, 'save_visualizations', False):
        from csttool.extract.modules import save_all_extraction_visualizations
        # Reorient the FA background into the same (RAS) voxel grid as the ROI
        # masks so overlays and slice selection align; the masks and affine are
        # produced by registration in RAS. Presentation only - does not affect
        # extracted streamlines or metrics.
        _fa_viz, _affine_viz = _viz_geometry.to_ras(fa_data, fa_affine)
        save_all_extraction_visualizations(
            cst_result=cst_result,
            fa=_fa_viz,
            masks=masks,
            affine=_affine_viz,
            output_dir=args.out,
            subject_id=args.subject_id,
            jacobian_det=reg_result.get('jacobian_det'),
            verbose=verbose
        )

    # Summary
    print(f"\n✓ Extraction complete")
    print(f"  Subject:        {args.subject_id}")
    print(f"  Left CST:       {cst_result['stats']['cst_left_count']:,} streamlines")
    print(f"  Right CST:      {cst_result['stats']['cst_right_count']:,} streamlines")
    print(f"  Total:          {cst_result['stats']['cst_total_count']:,} streamlines")
    print(f"  Extraction rate: {cst_result['stats']['extraction_rate']:.2f}%")
    
    return {
        'cst_left_path': output_paths.get('cst_left'),
        'cst_right_path': output_paths.get('cst_right'),
        'cst_combined_path': output_paths.get('cst_combined'),
        'stats': cst_result['stats']
    }


def run_roi_seeded_extraction(
    preproc_path: Path,
    fa_path: Path,
    output_dir: Path,
    subject_id: str,
    args: argparse.Namespace,
    verbose: bool = True
) -> dict | None:
    """
    Run ROI-seeded CST extraction.
    
    This method seeds streamlines directly from motor cortex ROIs
    and filters by brainstem traversal. Requires raw DWI data.
    """
    from csttool.extract.modules.registration import register_mni_to_subject
    from csttool.extract.modules.warp_atlas_to_subject import (
        warp_harvard_oxford_to_subject,
        CST_ROI_CONFIG
    )
    from csttool.extract.modules.create_roi_masks import create_cst_roi_masks
    from csttool.extract.modules.roi_seeded_tracking import extract_cst_roi_seeded
    from csttool.extract.modules.endpoint_filtering import (
        save_cst_tractograms,
        save_extraction_report
    )

    output_dir.mkdir(parents=True, exist_ok=True)

    ctx = RunContext(run_seed=getattr(args, 'rng_seed', 42))

    # Load preprocessed DWI data
    if verbose:
        print(f"    • Input: {preproc_path}")
    
    dwi_img = nib.load(str(preproc_path))
    data = dwi_img.get_fdata()
    affine = dwi_img.affine
    
    # Load gradient information
    bval_path = preproc_path.with_suffix('').with_suffix('.bval')
    bvec_path = preproc_path.with_suffix('').with_suffix('.bvec')
    
    # Handle .nii.gz extension
    if not bval_path.exists():
        stem = preproc_path.name.replace('.nii.gz', '').replace('.nii', '')
        bval_path = preproc_path.parent / f"{stem}.bval"
        bvec_path = preproc_path.parent / f"{stem}.bvec"
    
    bvals, bvecs = read_bvals_bvecs(str(bval_path), str(bvec_path))
    gtab = gradient_table(bvals, bvecs)
    
    # Load FA map
    fa_img = nib.load(str(fa_path))
    fa_data = fa_img.get_fdata()
    fa_affine = fa_img.affine
    
    # Create brain mask from FA
    brain_mask = fa_data > 0
    
    # Step 1: Registration
    if verbose:
        print(f"\n[Step 1/5] Registering MNI template to subject space...")
    
    level_iters_affine = [1000, 100, 10] if getattr(args, 'fast_registration', False) else [10000, 1000, 100]
    level_iters_syn = [5, 5, 3] if getattr(args, 'fast_registration', False) else [10, 10, 5]
    
    reg_result = register_mni_to_subject(
        subject_fa_path=fa_path,
        output_dir=output_dir,
        level_iters_affine=level_iters_affine,
        level_iters_syn=level_iters_syn,
        verbose=verbose
    )
    
    # Step 2: Warp atlases
    if verbose:
        print(f"\n[Step 2/5] Warping Harvard-Oxford atlases to subject space...")
    
    warped = warp_harvard_oxford_to_subject(
        registration_result=reg_result,
        output_dir=output_dir,
        subject_id=subject_id,
        verbose=verbose
    )
    
    # Step 3: Create ROI masks
    if verbose:
        print(f"\n[Step 3/5] Creating CST ROI masks...")
    
    masks = create_cst_roi_masks(
        warped_cortical=warped['cortical_warped'],
        warped_subcortical=warped['subcortical_warped'],
        subject_affine=fa_affine,
        roi_config=CST_ROI_CONFIG,
        dilate_brainstem=getattr(args, 'dilate_brainstem', 2),
        dilate_motor=getattr(args, 'dilate_motor', 1),
        output_dir=output_dir,
        subject_id=subject_id,
        verbose=verbose,
        original_subject_affine=warped.get('original_subject_affine'),
        reorientation_transform=warped.get('reorientation_transform'),
        hemisphere_mask=warped.get('hemisphere_mask'),
        midline_x=warped.get('midline_x'),
    )
    
    # Step 4: ROI-seeded extraction
    if verbose:
        print(f"\n[Step 4/5] ROI-seeded CST extraction...")
    
    cst_result = extract_cst_roi_seeded(
        data=data,
        gtab=gtab,
        affine=affine,
        brain_mask=brain_mask,
        motor_left_mask=masks['motor_left'],
        motor_right_mask=masks['motor_right'],
        brainstem_mask=masks['brainstem'],
        fa_map=fa_data,
        fa_threshold=getattr(args, 'seed_fa_threshold', 0.15),
        seed_density=getattr(args, 'seed_density', 2),
        step_size=0.5,
        min_length=getattr(args, 'min_length', 30.0),
        max_length=getattr(args, 'max_length', 200.0),
        random_seed=ctx.rng_tracking_seed(),
        npeaks=getattr(args, 'npeaks', 1),
        fit_method=getattr(args, 'fit_method', 'WLS'),
        verbose=verbose
    )
    
    # Step 5: Save outputs
    if verbose:
        print(f"\n[Step 5/5] Saving extracted tractograms...")
    
    output_paths = save_cst_tractograms(
        cst_result=cst_result,
        reference_img=fa_img,
        output_dir=output_dir,
        subject_id=subject_id,
        verbose=verbose
    )
    
    save_extraction_report(cst_result, output_paths, output_dir, subject_id)
    
    # Visualizations
    if getattr(args, 'save_visualizations', False):
        from csttool.extract.modules import save_all_extraction_visualizations
        # Reorient FA into the masks' (RAS) grid so overlays align (see cmd_extract).
        _fa_viz, _affine_viz = _viz_geometry.to_ras(fa_data, fa_affine)
        save_all_extraction_visualizations(
            cst_result=cst_result,
            fa=_fa_viz,
            masks=masks,
            affine=_affine_viz,
            output_dir=output_dir,
            subject_id=subject_id,
            jacobian_det=reg_result.get('jacobian_det'),
            verbose=verbose
        )

    # Summary
    if verbose:
        print(f"\n✓ ROI-seeded extraction complete")
        print(f"  Subject:   {subject_id}")
        print(f"  Left CST:  {cst_result['stats']['cst_left_count']:,} streamlines")
        print(f"  Right CST: {cst_result['stats']['cst_right_count']:,} streamlines")
        print(f"  Total:     {cst_result['stats']['cst_total_count']:,} streamlines")

    return {
        'cst_left_path': output_paths.get('cst_left'),
        'cst_right_path': output_paths.get('cst_right'),
        'cst_combined_path': output_paths.get('cst_combined'),
        'stats': cst_result['stats']
    }


def run_bidirectional_extraction(
    preproc_path: Path,
    fa_path: Path,
    output_dir: Path,
    subject_id: str,
    args: argparse.Namespace,
    verbose: bool = True
) -> dict | None:
    """
    Run bidirectional CST extraction.

    Seeds from motor cortex and brainstem independently, then retains only
    streamlines confirmed by both directions via spatial intersection.
    Requires raw DWI data.
    """
    from csttool.extract.modules.registration import register_mni_to_subject
    from csttool.extract.modules.warp_atlas_to_subject import (
        warp_harvard_oxford_to_subject,
        CST_ROI_CONFIG,
    )
    from csttool.extract.modules.create_roi_masks import create_cst_roi_masks
    from csttool.extract.modules.bidirectional_filtering import extract_cst_bidirectional
    from csttool.extract.modules.endpoint_filtering import (
        save_cst_tractograms,
        save_extraction_report,
    )

    output_dir.mkdir(parents=True, exist_ok=True)

    ctx = RunContext(run_seed=getattr(args, 'rng_seed', 42))

    if verbose:
        print(f"    • Input: {preproc_path}")

    dwi_img = nib.load(str(preproc_path))
    data = dwi_img.get_fdata()
    affine = dwi_img.affine

    bval_path = preproc_path.with_suffix('').with_suffix('.bval')
    bvec_path = preproc_path.with_suffix('').with_suffix('.bvec')
    if not bval_path.exists():
        stem = preproc_path.name.replace('.nii.gz', '').replace('.nii', '')
        bval_path = preproc_path.parent / f"{stem}.bval"
        bvec_path = preproc_path.parent / f"{stem}.bvec"

    bvals, bvecs = read_bvals_bvecs(str(bval_path), str(bvec_path))
    gtab = gradient_table(bvals, bvecs)

    fa_img = nib.load(str(fa_path))
    fa_data = fa_img.get_fdata()
    fa_affine = fa_img.affine
    brain_mask = fa_data > 0

    if verbose:
        print(f"\n[Step 1/5] Registering MNI template to subject space...")

    level_iters_affine = [1000, 100, 10] if getattr(args, 'fast_registration', False) else [10000, 1000, 100]
    level_iters_syn = [5, 5, 3] if getattr(args, 'fast_registration', False) else [10, 10, 5]

    reg_result = register_mni_to_subject(
        subject_fa_path=fa_path,
        output_dir=output_dir,
        level_iters_affine=level_iters_affine,
        level_iters_syn=level_iters_syn,
        verbose=verbose,
    )

    if verbose:
        print(f"\n[Step 2/5] Warping Harvard-Oxford atlases to subject space...")

    warped = warp_harvard_oxford_to_subject(
        registration_result=reg_result,
        output_dir=output_dir,
        subject_id=subject_id,
        verbose=verbose,
    )

    if verbose:
        print(f"\n[Step 3/5] Creating CST ROI masks...")

    masks = create_cst_roi_masks(
        warped_cortical=warped['cortical_warped'],
        warped_subcortical=warped['subcortical_warped'],
        subject_affine=fa_affine,
        roi_config=CST_ROI_CONFIG,
        dilate_brainstem=getattr(args, 'dilate_brainstem', 2),
        dilate_motor=getattr(args, 'dilate_motor', 1),
        output_dir=output_dir,
        subject_id=subject_id,
        verbose=verbose,
        original_subject_affine=warped.get('original_subject_affine'),
        reorientation_transform=warped.get('reorientation_transform'),
        hemisphere_mask=warped.get('hemisphere_mask'),
        midline_x=warped.get('midline_x'),
    )

    if verbose:
        print(f"\n[Step 4/5] Bidirectional CST extraction (forward + reverse + intersection)...")

    cst_result = extract_cst_bidirectional(
        data=data,
        gtab=gtab,
        affine=affine,
        brain_mask=brain_mask,
        motor_left_mask=masks['motor_left'],
        motor_right_mask=masks['motor_right'],
        brainstem_mask=masks['brainstem'],
        fa_map=fa_data,
        fa_threshold=getattr(args, 'seed_fa_threshold', 0.15),
        seed_density=getattr(args, 'seed_density', 2),
        step_size=0.5,
        min_length=getattr(args, 'min_length', 30.0),
        max_length=getattr(args, 'max_length', 200.0),
        random_seed=ctx.rng_tracking_seed(),
        npeaks=getattr(args, 'npeaks', 1),
        fit_method=getattr(args, 'fit_method', 'WLS'),
        verbose=verbose,
    )

    if verbose:
        print(f"\n[Step 5/5] Saving extracted tractograms...")

    output_paths = save_cst_tractograms(
        cst_result=cst_result,
        reference_img=fa_img,
        output_dir=output_dir,
        subject_id=subject_id,
        verbose=verbose,
    )

    save_extraction_report(cst_result, output_paths, output_dir, subject_id)

    if getattr(args, 'save_visualizations', False):
        from csttool.extract.modules import save_all_extraction_visualizations
        # Reorient FA into the masks' (RAS) grid so overlays align (see cmd_extract).
        _fa_viz, _affine_viz = _viz_geometry.to_ras(fa_data, fa_affine)
        save_all_extraction_visualizations(
            cst_result=cst_result,
            fa=_fa_viz,
            masks=masks,
            affine=_affine_viz,
            output_dir=output_dir,
            subject_id=subject_id,
            jacobian_det=reg_result.get('jacobian_det'),
            verbose=verbose,
        )

    if verbose:
        print(f"\n✓ Bidirectional extraction complete")
        print(f"  Subject:   {subject_id}")
        print(f"  Left CST:  {cst_result['stats']['cst_left_count']:,} streamlines")
        print(f"  Right CST: {cst_result['stats']['cst_right_count']:,} streamlines")
        print(f"  Total:     {cst_result['stats']['cst_total_count']:,} streamlines")

    return {
        'cst_left_path': output_paths.get('cst_left'),
        'cst_right_path': output_paths.get('cst_right'),
        'cst_combined_path': output_paths.get('cst_combined'),
        'stats': cst_result['stats'],
    }
