
import argparse
from pathlib import Path
import numpy as np

from dipy.io.image import load_nifti
from ..utils import extract_stem_from_filename, get_gtab_for_preproc
from csttool.defaults import (
    DEFAULT_B0_THRESHOLD,
    DEFAULT_FA_THRESHOLD,
    DEFAULT_NPEAKS,
    DEFAULT_MIN_SEPARATION_ANGLE,
    DEFAULT_RNG_SEED,
    DEFAULT_RELATIVE_PEAK_THRESHOLD,
    DEFAULT_FIT_METHOD,
)

from csttool.tracking.modules import (
    fit_tensors,
    tensor_validity_mask,
    estimate_directions,
    resolve_brain_mask,
    resolve_wm_mask,
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
    ctx = RunContext(run_seed=run_seed) if run_seed is not None else RunContext(run_seed=DEFAULT_RNG_SEED)
    scalar_only = bool(getattr(args, 'scalar_only', False))

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
    except (EOFError, OSError) as e:
        print(f"  ✗ Preprocessed NIfTI appears truncated or corrupt and could not be read: {preproc_nii}")
        print(f"    ({type(e).__name__}: {e})")
        return None
    except Exception as e:
        print(f"  ✗ Error loading NIfTI: {e}")
        return None

    print(f"  → Building gradient table...")
    try:
        gtab = get_gtab_for_preproc(
            preproc_nii,
            b0_threshold=getattr(args, 'b0_threshold', DEFAULT_B0_THRESHOLD),
        )
    except FileNotFoundError as e:
        print(f"  ✗ {e}")
        return None

    # Step 1: Brain masking
    brain_mask_path = getattr(args, 'brain_mask', None)
    if brain_mask_path is None:
        print(f"\n[Step 1/6] Brain masking with median Otsu...")
    else:
        print(f"\n[Step 1/6] Brain masking from external mask: {brain_mask_path}")

    try:
        masked_data, brain_mask, mask_info = resolve_brain_mask(
            data,
            gtab,
            affine,
            brain_mask_path=brain_mask_path,
            verbose=verbose,
        )
    except (FileNotFoundError, ValueError) as e:
        print(f"  ✗ Brain masking failed: {e}")
        return None

    # Step 2: Tensor fitting
    print(f"\n[Step 2/6] Tensor fit and scalar measures (FA, MD, RD, AD)...")
    try:
        tensor_result = fit_tensors(
            masked_data,
            gtab,
            brain_mask,
            fa_thresh=args.fa_thr,
            fit_method=getattr(args, 'fit_method', DEFAULT_FIT_METHOD),
            visualize=getattr(args, 'show_plots', False),
            verbose=verbose,
            return_metadata=True,
            create_white_matter=not scalar_only,
        )
        if len(tensor_result) == 7:
            (tenfit, fa, md, rd, ad, white_matter,
             tensor_metadata) = tensor_result
        elif len(tensor_result) == 6:
            # Compatibility with injected/third-party fitters implementing the
            # original public return contract.
            tenfit, fa, md, rd, ad, white_matter = tensor_result
            tensor_metadata = {}
        else:
            raise ValueError(
                "fit_tensors returned an unexpected number of values."
            )
    except Exception as e:
        print(f"  ✗ Tensor fitting failed: {e}")
        return None


    if scalar_only:
        tensor_valid = tensor_validity_mask(tenfit, brain_mask)
        tensor_metadata["fit_domain_source"] = "brain_mask"
        tensor_metadata["fit_domain_voxels"] = int(np.asarray(brain_mask).sum())
        tensor_metadata["valid_tensor_voxels"] = int(tensor_valid.sum())
        print("\n[Step 3/3] Saving preferred scalar/model-support products...")
        tracking_params = {
            'tracking_mode': 'preferred-plic',
            'stage_role': 'scalar-model-support',
            'scalar_model': 'DTI',
            'fit_method': getattr(args, 'fit_method', DEFAULT_FIT_METHOD),
            'tensor_fit': tensor_metadata,
            'brain_mask_source': mask_info['brain_mask_source'],
            'brain_mask_path': mask_info['brain_mask_path'],
            'model_mask_source': 'brain_mask',
        }
        try:
            outputs = save_tracking_outputs(
                [],
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
                verbose=verbose,
                tenfit=tenfit,
                white_matter=brain_mask,
                wm_info={'mask_source': mask_info['brain_mask_source']},
                save_tractogram=False,
                mask_role='model',
                tensor_valid_mask=tensor_valid,
            )
        except Exception as e:
            print(f"  ✗ Saving preferred scalar products failed: {e}")
            return None
        return {
            'tractogram_path': None,
            'fa_path': outputs['fa_map'],
            'md_path': outputs['md_map'],
            'rd_path': outputs.get('rd_map'),
            'ad_path': outputs.get('ad_map'),
            'v1_path': outputs.get('v1_map'),
            'model_mask_path': outputs.get('model_mask'),
            'tensor_valid_mask_path': outputs.get('tensor_valid_mask'),
            'wm_mask_path': None,
            'stem': stem,
            'tracking_params': tracking_params,
        }

    # The tensor fit also produced the legacy FA-threshold WM mask
    # (``white_matter``); the resolver decides whether that, an external
    # ``--wm-mask`` or DIPY's DAM segmentation is what this run actually
    # seeds and estimates peaks in, and records why. This deliberately sits
    # after ``fit_tensors`` so the legacy mask is available as the fallback
    # without a second computation.
    wm_mask_path = getattr(args, 'wm_mask', None)
    wm_method = getattr(args, 'wm_method', 'auto')
    if wm_mask_path is not None:
        print(f"\n  → White matter mask: external ({wm_mask_path})")
    elif wm_method == 'legacy':
        print(f"\n  → White matter mask: legacy FA-threshold (pinned)")
    else:
        print(f"\n  → White matter mask: automatic (DAM, legacy fallback)")
    try:
        wm_mask, wm_info = resolve_wm_mask(
            masked_data,
            gtab,
            affine,
            brain_mask,
            white_matter,
            wm_mask_path=wm_mask_path,
            method=wm_method,
            dam_wm_threshold=getattr(args, 'dam_wm_threshold', None),
            dam_low_signal=getattr(args, 'dam_low_signal', None),
            b0_threshold=getattr(args, 'b0_threshold', None),
            verbose=verbose,
        )
    except (FileNotFoundError, ValueError) as e:
        print(f"  ✗ White matter mask resolution failed: {e}")
        return None
    if verbose:
        frac = wm_info['wm_fraction_of_brain'] * 100.0
        print(f"    • WM mask source: {wm_info['wm_mask_source']} "
              f"({wm_info['wm_voxels']:,} voxels, {frac:.1f}% of brain)")
        if wm_info.get('fallback_reason'):
            print(f"    • Fallback reason: {wm_info['fallback_reason']}")

    # Step 3: Direction field estimation
    print(f"\n[Step 3/6] Direction field estimation (CSA ODF model)...")
    try:
        direction_result = estimate_directions(
            masked_data,
            gtab,
            wm_mask,
            sh_order=args.sh_order,
            npeaks=getattr(args, 'npeaks', DEFAULT_NPEAKS),
            verbose=verbose,
            return_metadata=True,
        )
        if isinstance(direction_result, tuple) and len(direction_result) == 2:
            csapeaks, direction_metadata = direction_result
        else:
            # Compatibility with injected/third-party estimators implementing
            # the original public return contract.
            csapeaks = direction_result
            direction_metadata = {}
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
            # Role separation (2026-09-22): the resolved WM mask feeds CSA peak
            # estimation (the model role, above) ONLY. It never becomes the seed
            # territory — an anatomical/DAM mask as the seed starves the cortical
            # rim the ROI-endpoint CST extraction needs (see the preferred
            # pipeline). The whole-brain track stage keeps its historical
            # FA-threshold seeding; deep-WM seeding lives in --tracking-mode
            # preferred, not here.
            seed_mask=None,
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
    
    if 'validated_sh_order' in direction_metadata:
        validated_sh_order = direction_metadata['validated_sh_order']
    else:
        validated_sh_order = validate_sh_order(
            gtab, args.sh_order, verbose=verbose
        )
    
    tracking_params = {
        'tracking_mode': 'legacy-fa',
        'tracking_algorithm': 'deterministic LocalTracking',
        'step_size': args.step_size,
        'fa_thresh': args.fa_thr,
        'seed_density': args.seed_density,
        'seed_source': 'FA-threshold white-matter mask',
        'model_mask_source': wm_info['wm_mask_source'],
        'sh_order': validated_sh_order,
        'sphere': 'symmetric362',
        'stopping_criterion': 'fa_threshold' + ('+brain_mask' if use_brain_mask_stop else ''),
        'relative_peak_threshold': direction_metadata.get(
            'relative_peak_threshold', DEFAULT_RELATIVE_PEAK_THRESHOLD
        ),
        'min_separation_angle': direction_metadata.get(
            'min_separation_angle', DEFAULT_MIN_SEPARATION_ANGLE
        ),
        'npeaks': getattr(args, 'npeaks', DEFAULT_NPEAKS),
        'random_seed': ctx.run_seed,
        'use_brain_mask_stop': use_brain_mask_stop,
        'fit_method': getattr(args, 'fit_method', DEFAULT_FIT_METHOD),
        'scalar_model': 'DTI',
        'tensor_fit': tensor_metadata,
        'direction_model': direction_metadata,
        'brain_mask_source': mask_info['brain_mask_source'],
        'brain_mask_path': mask_info['brain_mask_path'],
        'wm_mask_source': wm_info['wm_mask_source'],
        'wm_mask_path': wm_info['wm_mask_path'],
        'wm_method': wm_info['wm_method'],
        'wm_voxels': wm_info['wm_voxels'],
        'wm_fraction_of_brain': wm_info['wm_fraction_of_brain'],
        'wm_dam_params': wm_info['dam_params'],
        'wm_fallback_reason': wm_info['fallback_reason'],
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
            verbose=verbose,
            tenfit=tenfit,
            white_matter=wm_mask,
            wm_info=wm_info,
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
            white_matter=wm_mask,
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
        # DEC-FA prototype panel (visualization-refactor M5). Reads the stored
        # world-frame V1 product written above; standalone PNG, not embedded in
        # the PDF report until review passes.
        try:
            from csttool.metrics.modules.qc_figures import plot_dec_fa_panel
            v1_path = outputs.get('v1_map')
            fa_path = outputs.get('fa_map')
            if v1_path and Path(v1_path).exists() and fa_path and Path(fa_path).exists():
                plot_dec_fa_panel(
                    v1_path=v1_path, fa_path=fa_path,
                    output_dir=args.out / "tracking" / "visualizations",
                    subject_id=stem, brain_mask_path=None,
                )
            else:
                print(f"  ⚠️ Skipping DEC-FA panel: V1 or FA product not available")
        except Exception as exc:
            print(f"  ⚠️ Could not generate DEC-FA panel: {exc}")

    # Summary
    print(f"\n✓ Tracking complete - {stem}")
    print(f"  Whole-brain streamlines: {len(streamlines):,}")
    for key, path in outputs.items():
        print(f"  {key}: {path}")
    
    return {
        'tractogram_path': outputs['tractogram'],
        'fa_path': outputs['fa_map'],
        'md_path': outputs['md_map'],
        'rd_path': outputs.get('rd_map'),
        'ad_path': outputs.get('ad_map'),
        'v1_path': outputs.get('v1_map'),
        'wm_mask_path': outputs.get('wm_mask'),
        'model_mask_path': outputs.get('model_mask'),
        'tensor_valid_mask_path': outputs.get('tensor_valid_mask'),
        'stem': stem,
        'tracking_params': tracking_params
    }
