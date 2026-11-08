
import argparse
from pathlib import Path
import numpy as np

from csttool.defaults import (
    DEFAULT_RNG_SEED,
    DEFAULT_DILATE_BRAINSTEM,
    DEFAULT_DILATE_MOTOR,
    DEFAULT_SEED_FA_THRESHOLD,
    DEFAULT_SEED_DENSITY_EXTRACT,
    DEFAULT_MIN_LENGTH,
    DEFAULT_MAX_LENGTH,
    DEFAULT_NPEAKS,
    DEFAULT_FIT_METHOD,
)
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
        
        save_extraction_report(cst_result, output_paths, args.out, args.subject_id,
                               registration=reg_result)

        # CST density volume (unconditional scientific data product, §2.5).
        # Written whenever the extraction stage runs, like FA/MD in the track
        # stage. The PNG that visualizes it is a separate, gated concern.
        from csttool.extract.modules.save_extract_outputs import write_cst_density_product
        density_path = write_cst_density_product(
            cst_result=cst_result,
            reference_img=reference_img,
            out_dir=args.out,
            subject_id=args.subject_id,
            step_size_mm=getattr(args, 'step_size', None),
            verbose=verbose,
        )
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
            template_label=reg_result.get('template', {}).get(
                'display_name', 'MNI template'),
            verbose=verbose
        )
        # CST density prototype panel (visualization-refactor M5). Reads the
        # stored density + FA products; standalone PNG, not in the PDF report
        # until review passes.
        try:
            from pathlib import Path as _P
            from csttool.metrics.modules.qc_figures import plot_cst_density_panel
            if density_path and _P(density_path).exists():
                plot_cst_density_panel(
                    density_path=density_path, fa_path=args.fa,
                    output_dir=args.out / "extraction" / "visualizations",
                    subject_id=args.subject_id,
                )
            else:
                print("  ⚠️  Skipping CST density panel: density product not available")
        except Exception as exc:
            print(f"  ⚠️  Could not generate CST density panel: {exc}")

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
        'density_path': density_path,  # None when the product could not be written
        # FA-grid ROI label map. Survives the extraction-dir cleanup that the
        # three roi_*.nii.gz masks do not, so the report can show what
        # constrained the extraction.
        'roi_dseg_path': masks.get('roi_dseg_path'),
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

    ctx = RunContext(run_seed=getattr(args, 'rng_seed', DEFAULT_RNG_SEED))

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
        dilate_brainstem=getattr(args, 'dilate_brainstem', DEFAULT_DILATE_BRAINSTEM),
        dilate_motor=getattr(args, 'dilate_motor', DEFAULT_DILATE_MOTOR),
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
        fa_threshold=getattr(args, 'seed_fa_threshold', DEFAULT_SEED_FA_THRESHOLD),
        seed_density=getattr(args, 'seed_density', DEFAULT_SEED_DENSITY_EXTRACT),
        step_size=0.5,
        min_length=getattr(args, 'min_length', DEFAULT_MIN_LENGTH),
        max_length=getattr(args, 'max_length', DEFAULT_MAX_LENGTH),
        random_seed=ctx.rng_tracking_seed(),
        npeaks=getattr(args, 'npeaks', DEFAULT_NPEAKS),
        fit_method=getattr(args, 'fit_method', DEFAULT_FIT_METHOD),
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
    
    save_extraction_report(cst_result, output_paths, output_dir, subject_id,
                           registration=reg_result)

    # CST density volume (unconditional scientific data product, §2.5).
    from csttool.extract.modules.save_extract_outputs import write_cst_density_product
    density_path = write_cst_density_product(
        cst_result=cst_result,
        reference_img=fa_img,
        out_dir=output_dir,
        subject_id=subject_id,
        step_size_mm=getattr(args, 'step_size', None),
        verbose=verbose,
    )

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
            template_label=reg_result.get('template', {}).get(
                'display_name', 'MNI template'),
            verbose=verbose
        )
        # CST density prototype panel (visualization-refactor M5).
        try:
            from pathlib import Path as _P
            from csttool.metrics.modules.qc_figures import plot_cst_density_panel
            if density_path and _P(density_path).exists():
                plot_cst_density_panel(
                    density_path=density_path, fa_path=args.fa,
                    output_dir=Path(output_dir) / "extraction" / "visualizations",
                    subject_id=subject_id,
                )
            else:
                print("  ⚠️  Skipping CST density panel: density product not available")
        except Exception as exc:
            print(f"  ⚠️  Could not generate CST density panel: {exc}")

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
        'density_path': density_path,  # None when the product could not be written
        # FA-grid ROI label map. Survives the extraction-dir cleanup that the
        # three roi_*.nii.gz masks do not, so the report can show what
        # constrained the extraction.
        'roi_dseg_path': masks.get('roi_dseg_path'),
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

    ctx = RunContext(run_seed=getattr(args, 'rng_seed', DEFAULT_RNG_SEED))

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
        dilate_brainstem=getattr(args, 'dilate_brainstem', DEFAULT_DILATE_BRAINSTEM),
        dilate_motor=getattr(args, 'dilate_motor', DEFAULT_DILATE_MOTOR),
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
        fa_threshold=getattr(args, 'seed_fa_threshold', DEFAULT_SEED_FA_THRESHOLD),
        seed_density=getattr(args, 'seed_density', DEFAULT_SEED_DENSITY_EXTRACT),
        step_size=0.5,
        min_length=getattr(args, 'min_length', DEFAULT_MIN_LENGTH),
        max_length=getattr(args, 'max_length', DEFAULT_MAX_LENGTH),
        random_seed=ctx.rng_tracking_seed(),
        npeaks=getattr(args, 'npeaks', DEFAULT_NPEAKS),
        fit_method=getattr(args, 'fit_method', DEFAULT_FIT_METHOD),
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

    save_extraction_report(cst_result, output_paths, output_dir, subject_id,
                           registration=reg_result)

    # CST density volume (unconditional scientific data product, §2.5).
    from csttool.extract.modules.save_extract_outputs import write_cst_density_product
    density_path = write_cst_density_product(
        cst_result=cst_result,
        reference_img=fa_img,
        out_dir=output_dir,
        subject_id=subject_id,
        step_size_mm=getattr(args, 'step_size', None),
        verbose=verbose,
    )

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
            template_label=reg_result.get('template', {}).get(
                'display_name', 'MNI template'),
            verbose=verbose,
        )
        # CST density prototype panel (visualization-refactor M5).
        try:
            from pathlib import Path as _P
            from csttool.metrics.modules.qc_figures import plot_cst_density_panel
            if density_path and _P(density_path).exists():
                plot_cst_density_panel(
                    density_path=density_path, fa_path=args.fa,
                    output_dir=Path(output_dir) / "extraction" / "visualizations",
                    subject_id=subject_id,
                )
            else:
                print("  ⚠️  Skipping CST density panel: density product not available")
        except Exception as exc:
            print(f"  ⚠️  Could not generate CST density panel: {exc}")

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
        'density_path': density_path,  # None when the product could not be written
        # FA-grid ROI label map. Survives the extraction-dir cleanup that the
        # three roi_*.nii.gz masks do not, so the report can show what
        # constrained the extraction.
        'roi_dseg_path': masks.get('roi_dseg_path'),
        'stats': cst_result['stats'],
    }


def _load_pve_if_valid(pve_path, ref_shape, ref_affine, name, verbose=True):
    """Load a continuous PVE map and validate co-registration to the FA grid.

    Returns the float array, or raises ValueError on grid/range violation. A PVE
    map must be on the tracking grid (no silent resampling), 3D, and hold
    continuous fractions in [0, 1] — a hard 0/1 label map is refused, because
    CMC is reserved for genuine partial-volume estimates.
    """
    if pve_path is None:
        return None
    img = nib.load(str(pve_path))
    data = np.asarray(img.dataobj, dtype=np.float64)
    if data.ndim != 3 or data.shape != tuple(ref_shape[:3]):
        raise ValueError(
            f"{name} PVE shape {data.shape} != FA grid {tuple(ref_shape[:3])}; "
            f"csttool does not resample tissue maps.")
    if not np.allclose(np.asarray(img.affine), np.asarray(ref_affine),
                       rtol=1e-5, atol=1e-3):
        raise ValueError(f"{name} PVE affine is not aligned with the FA grid.")
    if not np.isfinite(data).all() or data.min() < -1e-6 or data.max() > 1 + 1e-6:
        raise ValueError(
            f"{name} PVE values are outside [0, 1] (min {data.min():.3g}, "
            f"max {data.max():.3g}); CMC needs continuous partial volumes.")
    uniq = np.unique(data)
    if uniq.size <= 2:
        raise ValueError(
            f"{name} PVE looks like a hard label map (values {uniq[:5]}). CMC is "
            f"reserved for continuous PVE; discrete labels are not partial volumes.")
    if verbose:
        print(f"    • {name} PVE validated on the FA grid "
              f"(range {data.min():.2f}-{data.max():.2f})")
    return data


def _load_binary_on_grid(path, ref_shape, ref_affine, name, verbose=True):
    """Load a binary mask and validate it is on the FA grid (never resampled)."""
    img = nib.load(str(path))
    data = np.asarray(img.dataobj)
    if data.ndim != 3 or data.shape != tuple(ref_shape[:3]):
        raise ValueError(
            f"{name} shape {data.shape} != FA grid {tuple(ref_shape[:3])}; "
            f"csttool does not resample masks.")
    if not np.allclose(np.asarray(img.affine), np.asarray(ref_affine),
                       rtol=1e-5, atol=1e-3):
        raise ValueError(f"{name} affine is not aligned with the FA grid.")
    mask = data != 0
    if not mask.any():
        raise ValueError(f"{name} contains no nonzero voxels.")
    if verbose:
        print(f"    • {name}: {int(mask.sum()):,} voxels (external, FA grid)")
    return mask


def run_preferred_extraction(
    preproc_path: Path,
    fa_path: Path,
    output_dir: Path,
    subject_id: str,
    args: argparse.Namespace,
    verbose: bool = True,
) -> dict | None:
    """Preferred, role-separated CST pipeline (``--tracking-mode preferred``).

    Brain support, model-fit mask, PLIC seeds and stopping model are resolved
    independently. Seeds are the bilateral posterior limb of the internal
    capsule (JHU); the model mask is the permissive brain mask; stopping is CMC
    when continuous WM/GM/CSF PVE are supplied, else a permissive brain-mask
    domain (ROI-constrained DWI-only). A valid CST connects the PLIC seed to an
    ipsilateral motor target and an inferior (brainstem/peduncle) target.
    """
    from csttool.extract.modules.registration import register_mni_to_subject
    from csttool.extract.modules.warp_atlas_to_subject import (
        warp_harvard_oxford_to_subject, CST_ROI_CONFIG,
    )
    from csttool.extract.modules.create_roi_masks import create_cst_roi_masks
    from csttool.extract.modules.plic_seeding import (
        PlicSeedError,
        prepare_tracking_grid_masks,
        save_preferred_seed_masks,
        warp_plic_seeds,
    )
    from csttool.extract.modules.plic_cst import (
        extract_cst_plic, build_stopping_model, save_plic_diagnostics,
    )
    from csttool.extract.modules.endpoint_filtering import (
        save_cst_tractograms, save_extraction_report,
    )

    output_dir.mkdir(parents=True, exist_ok=True)
    ctx = RunContext(run_seed=getattr(args, 'rng_seed', DEFAULT_RNG_SEED))

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
    preferred_brain_path = getattr(args, 'preferred_brain_mask_path', None)
    if preferred_brain_path is not None:
        brain_mask = _load_binary_on_grid(
            preferred_brain_path,
            fa_data.shape,
            fa_affine,
            "preferred brain/model domain",
            verbose,
        )
        brain_mask_source = f"persisted:{preferred_brain_path}"
    else:
        brain_mask = fa_data > 0
        brain_mask_source = "nonzero_FA_compatibility_fallback"
    level_iters_affine = [1000, 100, 10] if getattr(args, 'fast_registration', False) else [10000, 1000, 100]
    level_iters_syn = [5, 5, 3] if getattr(args, 'fast_registration', False) else [10, 10, 5]

    if verbose:
        print("\n[1/6] Registering MNI template to subject space...")
    reg_result = register_mni_to_subject(
        subject_fa_path=fa_path, output_dir=output_dir,
        level_iters_affine=level_iters_affine, level_iters_syn=level_iters_syn,
        verbose=verbose)

    if verbose:
        print("\n[2/6] Warping Harvard-Oxford ROIs (motor + brainstem targets)...")
    warped = warp_harvard_oxford_to_subject(
        registration_result=reg_result, output_dir=output_dir,
        subject_id=subject_id, verbose=verbose)
    masks = create_cst_roi_masks(
        warped_cortical=warped['cortical_warped'],
        warped_subcortical=warped['subcortical_warped'],
        subject_affine=fa_affine, roi_config=CST_ROI_CONFIG,
        dilate_brainstem=getattr(args, 'dilate_brainstem', DEFAULT_DILATE_BRAINSTEM),
        dilate_motor=getattr(args, 'dilate_motor', DEFAULT_DILATE_MOTOR),
        output_dir=output_dir, subject_id=subject_id, verbose=verbose,
        original_subject_affine=warped.get('original_subject_affine'),
        reorientation_transform=warped.get('reorientation_transform'),
        hemisphere_mask=warped.get('hemisphere_mask'),
        midline_x=warped.get('midline_x'))

    if verbose:
        print("\n[3/6] Warping + validating JHU PLIC seed regions...")
    try:
        # Registration products are on an internal RAS grid. Convert all of
        # them together to the original DWI grid before brain intersection,
        # hemisphere splitting, or LocalTracking.
        plic = warp_plic_seeds(reg_result, brain_mask=None, verbose=verbose)
        tracking_masks = prepare_tracking_grid_masks(
            reg_result, masks, plic, brain_mask, affine
        )
    except PlicSeedError as exc:
        print(f"  ✗ PLIC/target grid validation failed: {exc}")
        return None

    # Semantic external overrides (validated on the FA grid, never resampled).
    # An external mask supplied for one role does NOT leak into the others.
    plic_left = tracking_masks['plic_left']
    plic_right = tracking_masks['plic_right']
    seed_source = "JHU ICBM-DTI-81 PLIC labels 19/20"
    if getattr(args, 'seed_mask', None) is not None:
        ext_seed = _load_binary_on_grid(args.seed_mask, fa_data.shape, fa_affine,
                                        "seed mask", verbose)
        hemi = tracking_masks['hemisphere_mask']
        plic_left, plic_right = ext_seed & hemi, ext_seed & (~hemi)
        if not plic_left.any() or not plic_right.any():
            print("  ✗ External seed mask is empty on one physical hemisphere.")
            return None
        seed_source = f"external:{args.seed_mask}"
    model_mask = brain_mask
    model_source = "brain_mask"
    if getattr(args, 'model_mask', None) is not None:
        model_mask = _load_binary_on_grid(args.model_mask, fa_data.shape,
                                           fa_affine, "model mask", verbose)
        model_source = f"external:{args.model_mask}"

    if verbose:
        print("\n[4/6] Resolving stopping model...")
    try:
        pve_wm = _load_pve_if_valid(getattr(args, 'wm_pve', None), fa_data.shape, fa_affine, 'WM', verbose)
        pve_gm = _load_pve_if_valid(getattr(args, 'gm_pve', None), fa_data.shape, fa_affine, 'GM', verbose)
        pve_csf = _load_pve_if_valid(getattr(args, 'csf_pve', None), fa_data.shape, fa_affine, 'CSF', verbose)
    except ValueError as exc:
        print(f"  ✗ PVE validation failed: {exc}")
        return None
    n_pve = sum(x is not None for x in (pve_wm, pve_gm, pve_csf))
    if 0 < n_pve < 3:
        print("  ✗ CMC needs all three of --wm-pve/--gm-pve/--csf-pve; "
              f"got {n_pve}. Supply all three or none (DWI-only).")
        return None
    stopping, stopping_meta = build_stopping_model(
        brain_mask, affine, pve_wm=pve_wm, pve_gm=pve_gm, pve_csf=pve_csf,
        step_size=getattr(args, 'step_size', 0.5), verbose=verbose)

    if verbose:
        print("\n[5/6] PLIC-seeded, dual-target CST extraction...")
    cst_result = extract_cst_plic(
        data=data, gtab=gtab, affine=affine, brain_mask=brain_mask,
        plic_left_mask=plic_left, plic_right_mask=plic_right,
        motor_left_mask=tracking_masks['motor_left'],
        motor_right_mask=tracking_masks['motor_right'],
        inferior_left_mask=tracking_masks['inferior_left'],
        inferior_right_mask=tracking_masks['inferior_right'],
        model_mask=model_mask,
        stopping_criterion=stopping, stopping_meta=stopping_meta,
        coordinate_frame_qc=tracking_masks['coordinate_frame_qc'],
        midline_distance=tracking_masks['midline_distance'],
        midline_tol_mm=getattr(args, 'midline_tolerance_mm', 8.0),
        seed_density=getattr(args, 'seed_density', 4),
        step_size=getattr(args, 'step_size', 0.5),
        min_length=getattr(args, 'min_length', 40.0),
        max_length=getattr(args, 'max_length', 300.0),
        sh_order=getattr(args, 'sh_order', 6),
        npeaks=getattr(args, 'npeaks', DEFAULT_NPEAKS),
        random_seed=ctx.rng_tracking_seed(),
        fit_method=getattr(args, 'fit_method', DEFAULT_FIT_METHOD),
        fa_map=fa_data,
        # Complete branch accounting is a production QC product, not an
        # optional debugging mode. The CLI flag controls only extra raw files.
        collect_diagnostics=True,
        verbose=verbose)
    cst_result['parameters']['plic_provenance'] = (
        tracking_masks['plic_provenance']
    )
    cst_result['parameters']['seed_source'] = seed_source
    cst_result['parameters']['brain_mask_source'] = brain_mask_source
    cst_result['parameters']['model_mask_source'] = model_source
    cst_result['parameters']['tracking_mode'] = 'preferred-plic'
    cst_result['parameters']['tracking_algorithm'] = (
        'deterministic LocalTracking'
    )
    cst_result['stats']['seed_source'] = seed_source
    cst_result['stats']['model_mask_source'] = model_source
    cst_result['stats']['coordinate_frame_qc'] = tracking_masks[
        'coordinate_frame_qc'
    ]

    if verbose:
        print("\n[6/6] Saving extracted tractograms...")
    output_paths = save_cst_tractograms(
        cst_result=cst_result, reference_img=fa_img,
        output_dir=output_dir, subject_id=subject_id, verbose=verbose)
    import hashlib
    for side in ("left", "right"):
        path = output_paths.get(f"cst_{side}")
        full_path = output_paths.get(f"cst_{side}_full")
        identity = {
            "analysis_segment_sha256": (
                hashlib.sha256(Path(path).read_bytes()).hexdigest()
                if path else None
            ),
            "full_tracking_sha256": (
                hashlib.sha256(Path(full_path).read_bytes()).hexdigest()
                if full_path else None
            ),
            "random_seed": int(ctx.run_seed),
        }
        cst_result["stats"]["tractography_qc"][side][
            "deterministic_run_identity"
        ] = identity
    seed_provenance = dict(tracking_masks['plic_provenance'])
    seed_provenance['source'] = seed_source
    seed_paths = save_preferred_seed_masks(
        plic_left,
        plic_right,
        affine,
        output_dir,
        subject_id,
        seed_provenance,
    )
    cst_result['stats']['seed_mask_files'] = {
        key: str(path) for key, path in seed_paths.items()
    }
    diagnostic_paths = {}
    if getattr(args, 'save_tracking_diagnostics', False):
        diagnostic_paths = save_plic_diagnostics(
            cst_result, fa_img, output_dir, subject_id, verbose=verbose
        )
        if diagnostic_paths:
            cst_result['stats']['diagnostic_files'] = {
                key: str(path) for key, path in diagnostic_paths.items()
            }
    save_extraction_report(cst_result, output_paths, output_dir, subject_id,
                           registration=reg_result)
    from csttool.extract.modules.save_extract_outputs import write_cst_density_product
    density_path = write_cst_density_product(
        cst_result=cst_result, reference_img=fa_img, out_dir=output_dir,
        subject_id=subject_id, step_size_mm=getattr(args, 'step_size', None),
        verbose=verbose)

    if verbose:
        s = cst_result['stats']
        print(f"\n✓ Preferred extraction complete — {subject_id}")
        print(f"  Left {s['cst_left_count']:,} / Right {s['cst_right_count']:,} "
              f"| stopping: {s['stopping_label']}")

    return {
        'cst_left_path': output_paths.get('cst_left'),
        'cst_right_path': output_paths.get('cst_right'),
        'cst_combined_path': output_paths.get('cst_combined'),
        'density_path': density_path,
        'cst_left_full_path': output_paths.get('cst_left_full'),
        'cst_right_full_path': output_paths.get('cst_right_full'),
        'cst_combined_full_path': output_paths.get('cst_combined_full'),
        'roi_dseg_path': masks.get('roi_dseg_path'),
        'diagnostic_paths': diagnostic_paths,
        'seed_mask_paths': seed_paths,
        'stats': cst_result['stats'],
        'parameters': cst_result['parameters'],
    }
