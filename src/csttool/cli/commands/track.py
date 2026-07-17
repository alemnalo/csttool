
import argparse
from pathlib import Path

from dipy.io.image import load_nifti
from ..utils import extract_stem_from_filename, get_gtab_for_preproc

from csttool.tracking.modules import (
    fit_tensors,
    estimate_directions,
    validate_sh_order,
    seed_and_stop,
    run_tractography,
    save_tracking_outputs,
)
from csttool.reproducibility import RunContext

def cmd_track(args: argparse.Namespace) -> dict | None:
    """
    Run whole-brain deterministic tractography on preprocessed data.
    """
    preproc_nii = args.nifti
    verbose = getattr(args, 'verbose', False)

    # Create RunContext with seed from CLI
    run_seed = getattr(args, 'rng_seed', None)
    ctx = RunContext(run_seed=run_seed) if run_seed is not None else RunContext(run_seed=42)

    if not preproc_nii.exists():
        print(f"  ✗ Preprocessed NIfTI not found: {preproc_nii}")
        return None

    args.out.mkdir(parents=True, exist_ok=True)

    # Determine subject ID / stem
    if args.subject_id:
        stem = args.subject_id
    else:
        stem = extract_stem_from_filename(str(preproc_nii))

    print("=" * 60)
    print("WHOLE-BRAIN TRACTOGRAPHY")
    print("=" * 60)
    if ctx.run_seed is not None:
        print(f"  → Using random seed: {ctx.run_seed}")
    else:
        print(f"  → Random seeding disabled - results will vary between runs")
    print(f"  → Loading preprocessed data: {preproc_nii}")
    
    try:
        data, affine, img = load_nifti(str(preproc_nii), return_img=True)
    except Exception as e:
        print(f"  ✗ Error loading NIfTI: {e}")
        return None

    print(f"  → Building gradient table...")
    try:
        gtab = get_gtab_for_preproc(preproc_nii)
    except FileNotFoundError as e:
        print(f"  ✗ {e}")
        return None

    # Step 1: Brain masking
    print(f"\n[Step 1/6] Brain masking with median Otsu...")

    from csttool.preprocess.modules.background_segmentation import background_segmentation
    masked_data, brain_mask = background_segmentation(
        data,
        gtab
    )

    # Step 2: Tensor fitting
    print(f"\n[Step 2/6] Tensor fit and scalar measures (FA, MD, RD, AD)...")
    try:
        tenfit, fa, md, rd, ad, white_matter = fit_tensors(
            masked_data,
            gtab,
            brain_mask,
            fa_thresh=args.fa_thr,
            fit_method=getattr(args, 'fit_method', 'WLS'),
            visualize=getattr(args, 'show_plots', False),
            verbose=verbose
        )
    except Exception as e:
        print(f"  ✗ Tensor fitting failed: {e}")
        return None

    # Step 3: Direction field estimation
    print(f"\n[Step 3/6] Direction field estimation (CSA ODF model)...")
    try:
        csapeaks = estimate_directions(
            masked_data,
            gtab,
            white_matter,
            sh_order=args.sh_order,
            verbose=verbose
        )
    except Exception as e:
        print(f"  ✗ Direction estimation failed: {e}")
        return None

    # Step 4: Stopping criterion and seeds
    print(f"\n[Step 4/6] Stopping criterion and seed generation...")
    use_brain_mask_stop = getattr(args, 'use_brain_mask_stop', False)
    try:
        seeds, stopping_criterion = seed_and_stop(
            fa,
            affine,
            white_matter=white_matter,
            brain_mask=brain_mask,
            fa_thresh=args.fa_thr,
            density=args.seed_density,
            use_binary=False,
            use_brain_mask_stop=use_brain_mask_stop,
            verbose=verbose
        )
    except Exception as e:
        print(f"  ✗ Seed generation failed: {e}")
        return None

    # Step 5: Deterministic tracking
    print(f"\n[Step 5/6] Deterministic tracking...")
    try:
        streamlines = run_tractography(
            csapeaks,
            stopping_criterion,
            seeds,
            affine,
            step_size=args.step_size,
            random_seed=ctx.rng_tracking_seed(),
            verbose=verbose,
            visualize=getattr(args, 'show_plots', False)
        )
    except Exception as e:
        print(f"  ✗ Tractography failed: {e}")
        return None

    # Step 6: Save outputs
    print(f"\n[Step 6/6] Saving tractogram, scalar maps, and report...")
    
    # Get the validated SH order (may have been reduced due to insufficient directions)
    validated_sh_order = validate_sh_order(gtab, args.sh_order, verbose=False)
    
    tracking_params = {
        'step_size': args.step_size,
        'fa_thresh': args.fa_thr,
        'seed_density': args.seed_density,
        'sh_order': validated_sh_order,
        'sphere': 'symmetric362',
        'stopping_criterion': 'fa_threshold' + ('+brain_mask' if use_brain_mask_stop else ''),
        'relative_peak_threshold': 0.8,
        'min_separation_angle': 45,  # Peak extraction: minimum angle between detected peaks
        'random_seed': ctx.run_seed,
        'use_brain_mask_stop': use_brain_mask_stop,
        'fit_method': getattr(args, 'fit_method', 'WLS'),
    }

    try:
        outputs = save_tracking_outputs(
            streamlines,
            img,
            fa,
            md,
            affine,
            out_dir=args.out,
            stem=stem,
            rd=rd,
            ad=ad,
            tracking_params=tracking_params,
            provenance=ctx.provenance,
            verbose=verbose
        )
    except Exception as e:
        print(f"  ✗ Saving outputs failed: {e}")
        return None
    
    # After save_tracking_outputs() and before the summary print
    if getattr(args, 'save_visualizations', False):
        from csttool.tracking.modules import save_all_tracking_visualizations
        save_all_tracking_visualizations(
            streamlines=streamlines,
            fa=fa,
            md=md,
            white_matter=white_matter,
            brain_mask=brain_mask,
            seeds=seeds,
            affine=affine,
            output_dir=args.out,
            stem=stem,
            tenfit=tenfit,
            fa_thresh=args.fa_thr,
            tracking_params=tracking_params,
            verbose=verbose
        )

    # Summary
    print(f"\n✓ Tracking complete - {stem}")
    print(f"  Whole-brain streamlines: {len(streamlines):,}")
    for key, path in outputs.items():
        print(f"  {key}: {path}")
    
    return {
        'tractogram_path': outputs['tractogram'],
        'fa_path': outputs['fa_map'],
        'md_path': outputs['md_map'],
        'rd_path': outputs.get('rd_map'),  # May be None if not computed
        'ad_path': outputs.get('ad_map'),  # May be None if not computed
        'n_streamlines': len(streamlines),
        'stem': stem,
        'tracking_params': tracking_params
    }
