
import argparse
import json

from csttool.defaults import (
    DEFAULT_FA_THRESHOLD,
    DEFAULT_SEED_DENSITY,
    DEFAULT_STEP_SIZE,
    DEFAULT_SH_ORDER,
    DEFAULT_FIT_METHOD,
    DEFAULT_RNG_SEED,
    DEFAULT_DILATE_BRAINSTEM,
    DEFAULT_DILATE_MOTOR,
    DEFAULT_MIN_LENGTH_PASSTHROUGH,
    DEFAULT_MAX_LENGTH_PASSTHROUGH,
)
import shutil
import sys
from pathlib import Path
from datetime import datetime
from time import time

from csttool import __version__
from csttool.defaults import DEFAULT_DENOISE_METHOD
from ..utils import extract_stem_from_filename, save_pipeline_report
from csttool.bids.output import (
    write_dataset_description,
    update_participants_tsv,
    bids_filename,
    write_derivative_sidecar,
)

from .doctor import cmd_doctor as cmd_check
from .import_cmd import cmd_import
from .preprocess import cmd_preprocess
from .track import cmd_track
from .extract import cmd_extract, run_roi_seeded_extraction, run_bidirectional_extraction
from .metrics import cmd_metrics

def cmd_run(args: argparse.Namespace) -> None:
    """
    Run complete CST analysis pipeline.
    
    Steps:
        1. check     - Verify environment and dependencies
        2. import    - Convert DICOM to NIfTI (or validate existing NIfTI)
        3. preprocess - Denoise, skull strip, (optional) motion correct
        4. track     - Whole-brain deterministic tractography
        5. extract   - Atlas-based bilateral CST extraction
        6. metrics   - Compute metrics and generate reports
    """
    
    verbose = getattr(args, 'verbose', False)
    quiet = getattr(args, 'quiet', False)
    continue_on_error = getattr(args, 'continue_on_error', False)

    # Quiet overrides verbose
    if quiet:
        verbose = False

    # Create output directory structure
    args.out.mkdir(parents=True, exist_ok=True)
    
    # Determine subject ID
    subject_id = args.subject_id
    if not subject_id:
        if args.dicom:
            subject_id = args.dicom.name
        elif args.nifti:
            subject_id = extract_stem_from_filename(str(args.nifti))
        else:
            subject_id = "subject"
    
    # Initialize pipeline tracking
    pipeline_start = time()
    step_times = {}
    step_results = {}
    failed_steps = []
    pipeline_metadata = {}  # Initialize here to ensure it's available throughout
    
    if not quiet:
        print("\n" + "=" * 60)
        print("CST ANALYSIS PIPELINE")
        print("=" * 60)
        print(f"  Subject ID:  {subject_id}")
        print(f"  Output:      {args.out}")
        print(f"  Started:     {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    
    # =========================================================================
    # STEP 1: CHECK
    # =========================================================================
    if not getattr(args, 'skip_check', False):
        if not quiet:
            print(f"\n[Step 1/6] Environment check...")
        t0 = time()
        
        try:
            # Create a mock args object for cmd_check
            check_args = argparse.Namespace()
            check_ok = cmd_check(check_args)
            step_results['check'] = {'success': check_ok}
            
            if not check_ok and not continue_on_error:
                print(f"\n  ✗ Environment check failed. Fix dependencies and retry.")
                return
                
        except Exception as e:
            print(f"  ✗ Check failed: {e}")
            failed_steps.append('check')
            step_results['check'] = {'success': False, 'error': str(e)}
            if not continue_on_error:
                return
        
        step_times['check'] = time() - t0
    else:
        if not quiet:
            print(f"\n[Step 1/6] Environment check... ⚠️ skipped")
        step_results['check'] = {'success': True, 'skipped': True}
    
    # =========================================================================
    # STEP 2: IMPORT
    # =========================================================================
    if not quiet:
        print(f"\n[Step 2/6] Importing data...")
    t0 = time()
    
    nifti_path = None
    
    try:
        if args.nifti:
            # Check if file exists (handles broken symlinks too)
            if not args.nifti.exists():
                # Check if it's a broken symlink (e.g., git-annex)
                if args.nifti.is_symlink():
                    raise FileNotFoundError(
                        f"NIfTI file is a broken symlink: {args.nifti}\n"
                        f"If using git-annex, run: git annex get {args.nifti}"
                    )
                else:
                    raise FileNotFoundError(f"NIfTI file does not exist: {args.nifti}")
            
            # Use provided NIfTI directly
            nifti_path = args.nifti
            print(f"  → Using existing NIfTI: {nifti_path}")
            step_results['import'] = {'success': True, 'nifti_path': str(nifti_path)}
            
            # Extract acquisition metadata from NIfTI
            try:
                from ..utils import load_with_preproc
                from csttool.ingest import extract_acquisition_metadata
                
                _, _, hdr, gtab, bids_json = load_with_preproc(nifti_path)
                voxel_size = tuple(float(v) for v in hdr.get_zooms()[:3])
                
                # Build CLI overrides
                overrides = {}
                if getattr(args, 'field_strength', None):
                    overrides['field_strength_T'] = args.field_strength
                if getattr(args, 'echo_time', None):
                    overrides['echo_time_ms'] = args.echo_time
                
                acquisition = extract_acquisition_metadata(
                    bvecs=gtab.bvecs,
                    bvals=gtab.bvals,
                    voxel_size=voxel_size,
                    bids_json=bids_json,
                    overrides=overrides
                )
                
                pipeline_metadata['acquisition'] = acquisition
                
            except Exception as e:
                if verbose:
                    print(f"    • Could not extract acquisition metadata: {e}")
                    
        elif args.dicom:
            # Run import from DICOM
            import_args = argparse.Namespace(
                dicom=args.dicom,
                nifti=None,
                out=args.out,
                subject_id=subject_id,
                series=getattr(args, 'series', None),
                scan_only=False,
                verbose=verbose,
                field_strength=getattr(args, 'field_strength', None),
                echo_time=getattr(args, 'echo_time', None)
            )
            
            import_result = cmd_import(import_args)
            
            if import_result and import_result.get('nifti_path'):
                nifti_path = Path(import_result['nifti_path'])
                step_results['import'] = {'success': True, 'result': import_result}
                
                # capture metadata from import
                if 'metadata' in import_result:
                     pipeline_metadata['acquisition'] = import_result['metadata']
            else:
                raise RuntimeError("Import failed to produce NIfTI file")
        else:
            raise ValueError("Must provide --dicom or --nifti")
            
    except Exception as e:
        print(f"  ✗ Import failed: {e}")
        failed_steps.append('import')
        step_results['import'] = {'success': False, 'error': str(e)}
        if not continue_on_error:
            save_pipeline_report(args.out, subject_id, step_results, step_times, failed_steps, pipeline_start)
            return
    
    step_times['import'] = time() - t0
    
    # =========================================================================
    # STEP 3: PREPROCESS
    # =========================================================================
    
    if getattr(args, 'preprocess', False):
        if not quiet:
            print(f"\n[Step 3/6] Preprocessing...")
        t0 = time()
        
        preproc_path = None
        
        try:
            if nifti_path is None:
                raise RuntimeError("No NIfTI available from import step")
            
            preproc_out = args.out / "preprocessing"
            preproc_args = argparse.Namespace(
                dicom=None,
                nifti=nifti_path,
                out=preproc_out,
                subject_id=subject_id,
                coil_count=getattr(args, 'coil_count', 4),
                denoise_method=getattr(args, 'denoise_method', DEFAULT_DENOISE_METHOD),
                show_plots=getattr(args, 'show_plots', False),
                save_visualizations=getattr(args, 'save_visualizations', False),
                unring=getattr(args, 'unring', False),
                perform_motion_correction=getattr(args, 'perform_motion_correction', False),
                target_voxel_size=getattr(args, 'target_voxel_size', None),
                verbose=verbose
            )
            
            preproc_result = cmd_preprocess(preproc_args)
            
            if preproc_result and preproc_result.get('preprocessed_path'):
                preproc_path = Path(preproc_result['preprocessed_path'])
                step_results['preprocess'] = {'success': True, 'result': preproc_result}
            else:
                raise RuntimeError("Preprocessing failed")
                
        except Exception as e:
            print(f"  ✗ Preprocessing failed: {e}")
            failed_steps.append('preprocess')
            step_results['preprocess'] = {'success': False, 'error': str(e)}
            if not continue_on_error:
                save_pipeline_report(args.out, subject_id, step_results, step_times, failed_steps, pipeline_start)
                return
        
        step_times['preprocess'] = time() - t0
        
        # Record metadata for report
        pipeline_metadata['preprocessing'] = {
            'status': 'Executed',
            'method': getattr(args, 'denoise_method', DEFAULT_DENOISE_METHOD),
            'unring': getattr(args, 'unring', False),
            'motion_correction': getattr(args, 'perform_motion_correction', False)
        }
        
    else:
        if not quiet:
            print(f"\n[Step 3/6] Preprocessing... ⚠️ skipped (pass-through)")
        t0 = time()
        # Pass through the input NIfTI as the "preprocessed" data
        preproc_path = nifti_path
        step_results['preprocess'] = {'success': True, 'skipped': True, 'passthrough_path': str(preproc_path)}
        step_times['preprocess'] = time() - t0
        
        # Record metadata for report
        pipeline_metadata['preprocessing'] = {
            'status': 'Skipped (External Preprocessing Used)'
        }
    
    # =========================================================================
    # STEP 4: TRACK
    # =========================================================================
    if not quiet:
        print(f"\n[Step 4/6] Tractography...")
    t0 = time()
    
    tractogram_path = None
    fa_path = None
    md_path = None
    rd_path = None
    ad_path = None
    
    try:
        if preproc_path is None:
            raise RuntimeError("No preprocessed data available")
        
        track_out = args.out / "tracking"
        track_args = argparse.Namespace(
            nifti=preproc_path,
            subject_id=subject_id,
            fa_thr=getattr(args, 'fa_thr', DEFAULT_FA_THRESHOLD),
            seed_density=getattr(args, 'seed_density', DEFAULT_SEED_DENSITY),
            step_size=getattr(args, 'step_size', DEFAULT_STEP_SIZE),
            sh_order=getattr(args, 'sh_order', DEFAULT_SH_ORDER),
            fit_method=getattr(args, 'fit_method', DEFAULT_FIT_METHOD),
            rng_seed=getattr(args, 'rng_seed', DEFAULT_RNG_SEED),
            save_visualizations=getattr(args, 'save_visualizations', False),
            show_plots=getattr(args, 'show_plots', False),
            verbose=verbose,
            out=track_out
        )
        
        track_result = cmd_track(track_args)
        
        if track_result:
            tractogram_path = Path(track_result['tractogram_path'])
            fa_path = Path(track_result['fa_path'])
            md_path = Path(track_result['md_path'])
            # RD and AD maps (if available)
            rd_path = Path(track_result['rd_path']) if 'rd_path' in track_result else None
            ad_path = Path(track_result['ad_path']) if 'ad_path' in track_result else None
            step_results['track'] = {'success': True, 'result': track_result}
            
            # Capture tracking parameters
            if 'tracking_params' in track_result:
                pipeline_metadata['tracking'] = track_result['tracking_params']
        else:
            raise RuntimeError("Tracking failed")
            
    except Exception as e:
        print(f"  ✗ Tracking failed: {e}")
        failed_steps.append('track')
        step_results['track'] = {'success': False, 'error': str(e)}
        if not continue_on_error:
            save_pipeline_report(args.out, subject_id, step_results, step_times, failed_steps, pipeline_start)
            return
    
    step_times['track'] = time() - t0
    
    # =========================================================================
    # STEP 5: EXTRACT
    # =========================================================================
    if not quiet:
        print(f"\n[Step 5/6] CST extraction...")
    t0 = time()
    
    cst_left_path = None
    cst_right_path = None
    
    extraction_method = getattr(args, 'extraction_method', 'passthrough')
    
    try:
        if tractogram_path is None or fa_path is None:
            raise RuntimeError("No tractogram or FA map available")
        
        extract_out = args.out / "extraction"
        
        if extraction_method == "roi-seeded":
            extract_result = run_roi_seeded_extraction(
                preproc_path=preproc_path,
                fa_path=fa_path,
                output_dir=extract_out,
                subject_id=subject_id,
                args=args,
                verbose=verbose,
            )
        elif extraction_method == "bidirectional":
            extract_result = run_bidirectional_extraction(
                preproc_path=preproc_path,
                fa_path=fa_path,
                output_dir=extract_out,
                subject_id=subject_id,
                args=args,
                verbose=verbose,
            )
        else:
            # endpoint or passthrough - use cmd_extract with tractogram
            extract_args = argparse.Namespace(
                tractogram=tractogram_path,
                fa=fa_path,
                subject_id=subject_id,
                dilate_brainstem=getattr(args, 'dilate_brainstem', DEFAULT_DILATE_BRAINSTEM),
                dilate_motor=getattr(args, 'dilate_motor', DEFAULT_DILATE_MOTOR),
                min_length=getattr(args, 'min_length', DEFAULT_MIN_LENGTH_PASSTHROUGH),
                max_length=getattr(args, 'max_length', DEFAULT_MAX_LENGTH_PASSTHROUGH),
                extraction_method=extraction_method,
                fast_registration=getattr(args, 'fast_registration', False),
                save_visualizations=getattr(args, 'save_visualizations', False),
                verbose=verbose,
                out=extract_out
            )
            
            extract_result = cmd_extract(extract_args)
        
        if extract_result:
            cst_left_path = extract_result.get('cst_left_path')
            cst_right_path = extract_result.get('cst_right_path')
            step_results['extract'] = {'success': True, 'result': extract_result}

            # Carry artifact-index diagnostic into report metadata. Only the
            # bidirectional method populates artifact_index in its stats; for all
            # other methods it stays None (rendered as N/A downstream), because the
            # index only describes the bidirectional bundle.
            stats = extract_result['stats']
            pipeline_metadata['extraction'] = {
                'method': extraction_method,
                'artifact_index': stats.get('artifact_index'),
                'forward_reverse_ratio_left': stats.get('forward_reverse_ratio_left'),
                'forward_reverse_ratio_right': stats.get('forward_reverse_ratio_right'),
            }

            if extract_result['stats']['cst_total_count'] == 0:
                print(f"  ⚠️ No CST streamlines extracted")
        else:
            raise RuntimeError("CST extraction failed or produced no streamlines")
            
    except Exception as e:
        print(f"  ✗ CST extraction failed: {e}")
        failed_steps.append('extract')
        step_results['extract'] = {'success': False, 'error': str(e)}
        if not continue_on_error:
            save_pipeline_report(args.out, subject_id, step_results, step_times, failed_steps, pipeline_start)
            return
    
    step_times['extract'] = time() - t0
    
    # =========================================================================
    # STEP 6: METRICS
    # =========================================================================
    if not quiet:
        print(f"\n[Step 6/6] Computing metrics & reports...")
    t0 = time()
    
    try:
        if cst_left_path is None or cst_right_path is None:
            raise RuntimeError("No CST tractograms available")
        
        metrics_out = args.out / "metrics"
        metrics_args = argparse.Namespace(
            cst_left=cst_left_path,
            cst_right=cst_right_path,
            fa=fa_path,
            md=md_path,
            rd=rd_path,
            ad=ad_path,
            subject_id=subject_id,
            generate_pdf=getattr(args, 'generate_pdf', False),
            save_visualizations=getattr(args, 'save_visualizations', False),
            verbose=verbose,
            out=metrics_out,
            space=getattr(args, 'space', "Native Space"),
            # internal: pass pipeline metadata
            pipeline_metadata=pipeline_metadata if 'pipeline_metadata' in locals() else {}
        )
        
        metrics_result = cmd_metrics(metrics_args)
        
        if metrics_result:
            step_results['metrics'] = {'success': True, 'result': metrics_result}
        else:
            raise RuntimeError("Metrics computation failed")
            
    except Exception as e:
        print(f"  ✗ Metrics failed: {e}")
        failed_steps.append('metrics')
        step_results['metrics'] = {'success': False, 'error': str(e)}
    
    step_times['metrics'] = time() - t0

    # Clean up empty intermediate directory
    intermediate_dir = args.out / "intermediate"
    if intermediate_dir.exists() and not any(intermediate_dir.iterdir()):
        intermediate_dir.rmdir()
        if verbose:
            print(f"    • Cleaned up empty intermediate directory")
    
    # =========================================================================
    # BIDS DERIVATIVES OUTPUT (default: same root as --out)
    # =========================================================================
    bids_out = getattr(args, 'bids_out', None) or args.out
    if not failed_steps:
        _write_bids_derivatives(
            args=args,
            subject_id=subject_id,
            session_id=getattr(args, 'session_id', None),
            bids_out=Path(bids_out),
            step_results=step_results,
            tractogram_path=tractogram_path,
            fa_path=fa_path,
            md_path=md_path,
            rd_path=rd_path,
            ad_path=ad_path,
            cst_left_path=cst_left_path,
            cst_right_path=cst_right_path,
            preproc_path=preproc_path,
            pipeline_metadata=pipeline_metadata,
            verbose=verbose,
        )

    # =========================================================================
    # FINAL SUMMARY
    # =========================================================================
    total_time = time() - pipeline_start

    # Save pipeline report
    report_path = save_pipeline_report(args.out, subject_id, step_results, step_times, failed_steps, pipeline_start)
    
    if not quiet:
        print("\n" + "=" * 60)
        print("PIPELINE COMPLETE")
        print("=" * 60)

        if failed_steps:
            print(f"\n  ✗ Pipeline finished with failures: {', '.join(failed_steps)}")
        else:
            print(f"\n✓ Processing complete")

        print(f"  Subject ID:      {subject_id}")
        print(f"  Total time:      {total_time/60:.1f} minutes ({total_time:.0f} seconds)")
        print(f"  Step timing:")
        for step, elapsed in step_times.items():
            status = "✓" if step not in failed_steps else "✗"
            print(f"    {status} {step:12s}: {elapsed:.1f}s")

        print(f"  Outputs:")
        print(f"    Pipeline report: {report_path}")
        if step_results.get('metrics', {}).get('success'):
            print(f"    Metrics:         {args.out / 'metrics'}")
        if step_results.get('extract', {}).get('success'):
            print(f"    CST tractograms: {args.out / 'extraction'}")
    elif failed_steps:
        # Even in quiet mode, report failures
        print(f"Pipeline completed with failures: {', '.join(failed_steps)}")


# ---------------------------------------------------------------------------
# QC image naming map
# ---------------------------------------------------------------------------

# Maps the distinctive filename suffix of each QC PNG to (stage_label, qc_label).
# stage_label groups by pipeline phase; qc_label names the specific artifact.
_QC_NAMES = {
    "_denoising_qc.png":           ("preproc",    "denoising"),
    "_gibbs_unringing_qc.png":     ("preproc",    "gibbs"),
    "_brain_mask_qc.png":          ("preproc",    "brainmask"),
    "_motion_qc.png":              ("preproc",    "motion"),
    "_preprocessing_summary.png":  ("preproc",    "summary"),
    "_tensor_maps.png":            ("tracking",   "tensormaps"),
    "_wm_mask_qc.png":             ("tracking",   "wmmask"),
    "_streamlines_2d.png":         ("tracking",   "streamlines"),
    "_streamline_stats.png":       ("tracking",   "stats"),
    "_tracking_summary.png":       ("tracking",   "summary"),
    "_registration_qc.png":        ("extraction", "registration"),
    "_jacobian_map.png":           ("extraction", "jacobian"),
    "_roi_masks.png":              ("extraction", "roimasks"),
    "_cst_extraction.png":         ("extraction", "cst"),
    "_hemisphere_separation.png":  ("extraction", "hemispheres"),
    "_extraction_summary.png":     ("extraction", "summary"),
    "_tract_profile_fa.png":       ("metrics",    "tractprofile"),
    "_bilateral_comparison.png":   ("metrics",    "bilateral"),
    "_stacked_profiles.png":       ("metrics",    "profiles"),
    "_tractogram_qc_axial.png":    ("metrics",    "tractogram-axial"),
    "_tractogram_qc_sagittal.png": ("metrics",    "tractogram-sagittal"),
    "_tractogram_qc_coronal.png":  ("metrics",    "tractogram-coronal"),
}


def _resolve_qc_name(filename: str):
    """Return (stage_label, qc_label) for a QC PNG, or ('misc', stem) if unknown."""
    for suffix, labels in _QC_NAMES.items():
        if filename.endswith(suffix):
            return labels
    return ("misc", filename.replace(".png", ""))


# ---------------------------------------------------------------------------
# BIDS derivatives writer
# ---------------------------------------------------------------------------

def _write_bids_derivatives(
    args,
    subject_id: str,
    session_id,
    bids_out: Path,
    step_results: dict,
    tractogram_path,
    fa_path,
    md_path,
    rd_path,
    ad_path,
    cst_left_path,
    cst_right_path,
    preproc_path,
    pipeline_metadata: dict,
    verbose: bool,
) -> None:
    """
    Move all pipeline outputs into a BIDS derivatives layout and remove stage dirs.

    Final layout:
        bids_out/
        ├── dataset_description.json
        ├── participants.tsv / participants.json
        └── sub-{sub}/[ses-{ses}/]
            ├── dwi/
            │   ├── *_desc-preproc_dwi.nii.gz  (+ .bval .bvec .json)
            │   ├── *_model-DTI_param-{FA,MD,RD,AD}_dwimap.nii.gz  (+ .json sidecars)
            │   └── tractography/
            │       ├── *_desc-wholebrain_tractogram.trk
            │       ├── *_desc-CSTleft_tractogram.trk
            │       ├── *_desc-CSTright_tractogram.trk
            │       └── *_desc-CSTbilateral_tractogram.trk
            ├── figures/   ← QC images only (PNG slice overlays, streamline views, etc.)
            │   └── *_stage-{stage}_qc-{label}.png
            └── reports/   ← user-facing reports + tabular outputs + pipeline logs
                ├── *_report.html
                ├── *_report.pdf
                ├── *_metrics.json
                ├── *_metrics.csv
                └── *_log-{step}.json

    Stage directories (tracking/, extraction/, metrics/, preprocessing/) are
    removed unconditionally after all files are moved out.
    """
    bids_out = Path(bids_out)

    sub_label = subject_id[4:] if subject_id.startswith("sub-") else subject_id
    ses_label = None
    if session_id:
        ses_label = session_id[4:] if session_id.startswith("ses-") else session_id

    sub_id = f"sub-{sub_label}"
    ses_id = f"ses-{ses_label}" if ses_label else None

    raw_bids_root = getattr(args, 'bids_in', None) or getattr(args, 'raw_bids', None)
    write_dataset_description(bids_out, raw_bids_root=raw_bids_root)

    # Subject/session output tree
    subses_dir = bids_out / sub_id
    if ses_id:
        subses_dir = subses_dir / ses_id

    dwi_dir   = subses_dir / "dwi"
    tract_dir = dwi_dir / "tractography"
    fig_dir   = subses_dir / "figures"    # QC images
    rep_dir   = subses_dir / "reports"    # HTML, PDF, tabular, logs

    for d in (dwi_dir, tract_dir, fig_dir, rep_dir):
        d.mkdir(parents=True, exist_ok=True)

    def _fname(suffix, ext, **entities):
        return bids_filename(sub_label, suffix, ext, session=ses_label, **entities)

    def _qc_name(stage, label):
        parts = [f"sub-{sub_label}"]
        if ses_label:
            parts.append(f"ses-{ses_label}")
        parts += [f"stage-{stage}", f"qc-{label}"]
        return "_".join(parts) + ".png"

    def _rep_name(stem, ext):
        parts = [f"sub-{sub_label}"]
        if ses_label:
            parts.append(f"ses-{ses_label}")
        parts.append(stem)
        return "_".join(parts) + ext

    cli_cmd = " ".join(sys.argv)

    # Files staged inside args.out may be moved (we own them); files outside
    # args.out are user-provided inputs that must be copied, never moved.
    out_root = Path(args.out).resolve()

    def _relocate(src_path: Path, dst_path: Path) -> None:
        src_path = Path(src_path)
        try:
            inside = src_path.resolve().is_relative_to(out_root)
        except (ValueError, AttributeError):
            # is_relative_to may raise on older Python or for non-overlapping paths
            inside = str(src_path.resolve()).startswith(str(out_root) + "/")
        if inside:
            shutil.move(src_path, dst_path)
        else:
            shutil.copy2(src_path, dst_path)

    # ------------------------------------------------------------------
    # Preprocessed DWI
    # ------------------------------------------------------------------
    if preproc_path and Path(preproc_path).exists():
        src = Path(preproc_path)
        dst_nii = dwi_dir / _fname("dwi", ".nii.gz", space="orig", desc="preproc")
        _relocate(src, dst_nii)

        stem_no_ext = src.name.replace(".nii.gz", "").replace(".nii", "")
        for ext in (".bval", ".bvec"):
            candidate = src.parent / (stem_no_ext + ext)
            if not candidate.exists():
                hits = list(src.parent.glob(f"*{ext}"))
                candidate = hits[0] if hits else None
            if candidate and Path(candidate).exists():
                _relocate(candidate, dwi_dir / _fname("dwi", ext, space="orig", desc="preproc"))

        json_cand = src.parent / (stem_no_ext + ".json")
        if json_cand.exists():
            _relocate(json_cand, dwi_dir / _fname("dwi", ".json", space="orig", desc="preproc"))
        else:
            write_derivative_sidecar(dst_nii, sources=[], description="Preprocessed DWI",
                                     command_line=cli_cmd)

        if verbose:
            print(f"    BIDS: {dst_nii.relative_to(bids_out)}")

    # ------------------------------------------------------------------
    # Scalar maps (FA, MD, RD, AD)
    # ------------------------------------------------------------------
    for src_path, param, desc_str in [
        (fa_path, "FA",  "FA map from DTI fit"),
        (md_path, "MD",  "Mean diffusivity map from DTI fit"),
        (rd_path, "RD",  "Radial diffusivity map from DTI fit"),
        (ad_path, "AD",  "Axial diffusivity map from DTI fit"),
    ]:
        if src_path and Path(src_path).exists():
            dst = dwi_dir / _fname("dwimap", ".nii.gz", space="orig", model="DTI", param=param)
            shutil.move(src_path, dst)
            write_derivative_sidecar(dst, sources=[], description=desc_str, command_line=cli_cmd)
            if verbose:
                print(f"    BIDS: {dst.relative_to(bids_out)}")

    # ------------------------------------------------------------------
    # Tractograms
    # ------------------------------------------------------------------
    for src_path, desc_val in [
        (tractogram_path, "wholebrain"),
        (cst_left_path,   "CSTleft"),
        (cst_right_path,  "CSTright"),
    ]:
        if src_path and Path(src_path).exists():
            dst = tract_dir / _fname("tractogram", ".trk", space="orig", desc=desc_val)
            shutil.move(src_path, dst)
            if verbose:
                print(f"    BIDS: {dst.relative_to(bids_out)}")

    # Combined CST (produced by extract step alongside the separated tractograms)
    for combined in (args.out / "extraction").glob("*_cst_combined.trk"):
        dst = tract_dir / _fname("tractogram", ".trk", space="orig", desc="CSTbilateral")
        shutil.move(combined, dst)
        if verbose:
            print(f"    BIDS: {dst.relative_to(bids_out)}")

    # ------------------------------------------------------------------
    # QC images → figures/
    # ------------------------------------------------------------------
    for viz_dir in [
        args.out / "preprocessing" / "visualizations",
        args.out / "tracking"      / "visualizations",
        args.out / "extraction"    / "visualizations",
        args.out / "metrics"       / "visualizations",
    ]:
        if not viz_dir.is_dir():
            continue
        for png in viz_dir.glob("*.png"):
            stage, label = _resolve_qc_name(png.name)
            dst = fig_dir / _qc_name(stage, label)
            shutil.move(png, dst)
            if verbose:
                print(f"    BIDS: {dst.relative_to(bids_out)}")

    # ------------------------------------------------------------------
    # Reports → reports/  (HTML and PDF)
    # ------------------------------------------------------------------
    metrics_out = args.out / "metrics"
    for html in metrics_out.glob("*.html"):
        shutil.move(html, rep_dir / _rep_name("report", ".html"))
    for pdf in metrics_out.glob("*.pdf"):
        shutil.move(pdf, rep_dir / _rep_name("report", ".pdf"))

    # ------------------------------------------------------------------
    # Tabular outputs → reports/  (metrics JSON + CSV)
    # ------------------------------------------------------------------
    for json_f in metrics_out.glob("*_bilateral_metrics.json"):
        shutil.move(json_f, rep_dir / _rep_name("metrics", ".json"))
    for csv_f in metrics_out.glob("*_metrics_summary.csv"):
        shutil.move(csv_f, rep_dir / _rep_name("metrics", ".csv"))

    # ------------------------------------------------------------------
    # Pipeline logs → reports/  (provenance, not scientific payload)
    # ------------------------------------------------------------------
    # (source_dir, default_tag, filename_pattern → specific_tag overrides)
    # The overrides keep distinct reports (e.g. registration vs CST extraction)
    # from clobbering each other when multiple JSONs share a stage directory.
    log_sources = [
        (args.out / "preprocessing" / "nifti",      "log-import",     {}),
        (args.out / "preprocessing" / "dicom_info", "log-series",     {}),
        (args.out / "preprocessing",                "log-preproc",    {}),
        (args.out / "tracking"      / "logs",       "log-tracking",   {}),
        (args.out / "extraction"    / "logs",       "log-extraction", {
            "registration_report":   "log-registration",
            "cst_extraction_report": "log-extraction",
        }),
    ]
    for log_dir, default_tag, overrides in log_sources:
        if not log_dir.is_dir():
            continue
        for jf in log_dir.glob("*.json"):
            tag = default_tag
            for needle, override_tag in overrides.items():
                if needle in jf.name:
                    tag = override_tag
                    break
            shutil.move(jf, rep_dir / _rep_name(tag, ".json"))

    if verbose:
        print(f"    BIDS: {rep_dir.relative_to(bids_out)}/ (reports + logs)")

    # ------------------------------------------------------------------
    # participants.tsv
    # ------------------------------------------------------------------
    acq_meta = pipeline_metadata.get('acquisition', {})
    update_participants_tsv(bids_out, sub_id, metadata={
        "age": acq_meta.get("patient_age"),
        "sex": acq_meta.get("patient_sex"),
    })

    # ------------------------------------------------------------------
    # Remove stage directories unconditionally
    # Each one should be empty after all moves above; rmtree handles any
    # empty subdirectories (e.g. scalar_maps/, logs/) left behind.
    # ------------------------------------------------------------------
    for step in ("preprocessing", "tracking", "extraction", "metrics"):
        d = args.out / step
        if d.exists():
            shutil.rmtree(d)

    print(f"\n  BIDS derivatives written → {bids_out}")
