
import argparse
import json
from pathlib import Path
from dipy.io.streamline import load_tractogram
from dipy.io.image import load_nifti

from csttool.reproducibility import get_provenance_dict

from csttool.metrics import (
    analyze_cst_hemisphere,
    compare_bilateral_cst,
    print_hemisphere_summary,
    plot_tract_profiles,
    plot_bilateral_comparison,
    plot_profile_matrix,
    plot_report_qc_strip,
    plot_tractogram_qc_preview,
    plot_tractogram_qc_triptych,
)
from csttool.metrics.modules.reports import (
    save_json_report,
    save_csv_summary,
    save_html_report,
    save_pdf_report,
)


def _build_extraction_metadata(raw: dict | None) -> dict | None:
    """Build the self-documenting extraction block for the report metadata.

    The artifact index is only defined for bidirectional extraction; for every
    other method ``artifact_index`` is None. We always carry the method and an
    explicit availability flag + reason so the JSON/PDF never has to infer
    absence from missing structure.
    """
    if not raw or not raw.get('method'):
        return None
    ai = raw.get('artifact_index')
    return {
        'method': raw.get('method'),
        'artifact_index': ai,
        'forward_reverse_ratio_left': raw.get('forward_reverse_ratio_left'),
        'forward_reverse_ratio_right': raw.get('forward_reverse_ratio_right'),
        'artifact_index_available': ai is not None,
        'artifact_index_reason': None if ai is not None
            else 'defined only for bidirectional extraction',
    }


def _find_extraction_statistics(cst_left_path: Path, subject_id: str) -> dict | None:
    """Locate the on-disk extraction report next to a standalone tractogram input.

    Returns its ``statistics`` dict, or None if not found / unreadable.
    """
    name = f"{subject_id}_cst_extraction_report.json"
    candidates = [
        cst_left_path.parent.parent / "logs" / name,   # .../extraction/trk/ -> .../extraction/logs/
        cst_left_path.parent / "logs" / name,           # report alongside the tractogram dir
    ]
    for path in candidates:
        try:
            if path.exists():
                return json.loads(path.read_text()).get('statistics')
        except Exception:
            continue
    return None


def _resolve_v1_path(args) -> Path | None:
    """Locate the world-frame V1 product for the streamline-vs-tensor panel.

    Explicit ``--v1`` wins. Otherwise look beside the FA map, where
    ``save_tracking_outputs`` writes both: ``{stem}_fa.nii.gz`` and
    ``{stem}_v1.nii.gz`` are siblings in ``tracking/scalar_maps/``. Returns None
    when neither is available, which is the normal case for a run whose tensor
    fit was skipped — the caller then skips the panel rather than failing.
    """
    explicit = getattr(args, 'v1', None)
    if explicit and Path(explicit).exists():
        return Path(explicit)

    fa = getattr(args, 'fa', None)
    if not fa:
        return None
    fa = Path(fa)
    if not fa.name.endswith('_fa.nii.gz'):
        return None
    candidate = fa.with_name(fa.name[:-len('_fa.nii.gz')] + '_v1.nii.gz')
    return candidate if candidate.exists() else None


def _resolve_density_path(args) -> Path | None:
    """Locate the CST density product for the tissue-plausibility panel.

    Explicit ``--density`` wins. Otherwise look under the extraction stage's
    ``scalar_maps/`` beside the tractograms, where ``write_cst_density_product``
    puts it (``extraction/trk/*.trk`` and ``extraction/scalar_maps/*.nii.gz``
    are siblings). ``csttool run`` passes the path explicitly, so this fallback
    only serves a hand-run ``csttool metrics``. Returns None when the product is
    absent — the caller then skips the panel rather than failing.
    """
    explicit = getattr(args, 'density', None)
    if explicit and Path(explicit).exists():
        return Path(explicit)

    cst_left = getattr(args, 'cst_left', None)
    if not cst_left:
        return None
    scalar_maps = Path(cst_left).parent.parent / 'scalar_maps'
    if not scalar_maps.is_dir():
        return None
    candidates = sorted(scalar_maps.glob('*_cst_density.nii.gz'))
    return candidates[0] if candidates else None


def _resolve_roi_dseg_path(args) -> Path | None:
    """Locate the FA-grid extraction ROI label map for the QC strip's ROI panel.

    Explicit ``--roi-dseg`` wins. Otherwise look in two places, mirroring
    :func:`_resolve_density_path`: beside the FA map in a BIDS ``dwi/``
    directory, which is where ``csttool run`` moves the product, and in the
    extraction stage's ``nifti/``, which is where a hand-run pipeline leaves it
    before that move. ``csttool run`` passes the path explicitly, so these only
    serve a standalone ``csttool metrics``.

    Returns None when the product is absent — an older derivatives tree has no
    dseg at all, and the panel degrades in place rather than the strip failing.
    """
    explicit = getattr(args, 'roi_dseg', None)
    if explicit and Path(explicit).exists():
        return Path(explicit)

    fa = getattr(args, 'fa', None)
    if fa:
        candidates = sorted(Path(fa).parent.glob('*_desc-CSTroi_dseg.nii.gz'))
        if candidates:
            return candidates[0]

    cst_left = getattr(args, 'cst_left', None)
    if cst_left:
        nifti_dir = Path(cst_left).parent.parent / 'nifti'
        if nifti_dir.is_dir():
            candidates = sorted(nifti_dir.glob('*_desc-CSTroi_dseg.nii.gz'))
            if candidates:
                return candidates[0]
    return None


def _node_homology_summary(streamlines_left, streamlines_right, n_points=20) -> dict:
    """Node-homology scalars for the report, from `qc_stats.compare_node_geometry`.

    The twelve regional laterality indices assume node *i* is the same anatomical
    level on both sides. It is not always: node ``i`` is ``i/(n-1)`` of the way
    along whatever was reconstructed, so two hemispheres of different length put
    node ``i`` at different heights. This reports how far apart they actually
    are, so the reader can weigh the regional table rather than take it on trust.

    ``compare_bilateral_cst`` is deliberately not given this job: it sees only
    the two metric dicts, and widening a stable signature to pass streamlines
    through for one caller would be the wrong trade.

    The nested ``left``/``right`` geometry dicts are dropped: they carry a
    per-streamline length array each, which has no place in a summary block that
    is serialised into every report JSON.
    """
    from csttool.metrics.modules import qc_stats

    geometry = qc_stats.compare_node_geometry(
        streamlines_left, streamlines_right, n_points=n_points
    )
    left, right = geometry['left'], geometry['right']

    def _as_list(values):
        return None if values is None else [float(v) for v in values]

    return {
        'max_abs_z_difference_mm': geometry['max_abs_z_difference_mm'],
        'length_difference_mm': geometry['length_difference_mm'],
        'z_difference_mm': _as_list(geometry['z_difference_mm']),
        'arc_difference_mm': _as_list(geometry['arc_difference_mm']),
        'left_length_mean': left['length_mean'],
        'right_length_mean': right['length_mean'],
        'left_n_streamlines': left['n_streamlines'],
        'right_n_streamlines': right['n_streamlines'],
        'n_points': n_points,
    }


def _generate_trust_chain_panels(args, viz_dir, viz_paths) -> None:
    """Render the six trust-chain QC panels, one failure at a time.

    Each panel is independent, so one that cannot be built (a missing V1
    product, an empty hemisphere) must not suppress the others — hence a
    try/except per panel rather than around the group, matching the way the
    prototype panels are already called.
    """
    from csttool.metrics.modules import qc_figures

    cst_left = getattr(args, 'cst_left', None)
    cst_right = getattr(args, 'cst_right', None)
    fa_path = getattr(args, 'fa', None)
    md_path = getattr(args, 'md', None)

    density_path = _resolve_density_path(args)
    if fa_path and md_path and density_path:
        try:
            viz_paths['tissue_plausibility'] = \
                qc_figures.plot_tissue_plausibility_panel(
                    fa_path=fa_path, md_path=md_path,
                    density_path=density_path, output_dir=viz_dir,
                    subject_id=args.subject_id,
                )
            print(f"  ✓ Saved: {viz_paths['tissue_plausibility']}")
        except Exception as e:
            print(f"  ⚠️ Could not generate tissue plausibility panel: {e}")
    else:
        missing = [name for name, value in (('FA', fa_path), ('MD', md_path),
                                            ('CST density', density_path))
                   if not value]
        print("  ⚠️ Skipping tissue plausibility panel: missing "
              f"{', '.join(missing)}")

    v1_path = _resolve_v1_path(args)
    if v1_path is not None:
        try:
            viz_paths['v1_angle'] = qc_figures.plot_v1_angle_panel(
                cst_left_path=cst_left, cst_right_path=cst_right,
                v1_path=v1_path, output_dir=viz_dir, subject_id=args.subject_id,
            )
            print(f"  ✓ Saved: {viz_paths['v1_angle']}")
        except Exception as e:
            print(f"  ⚠️ Could not generate V1 angle panel: {e}")
    else:
        print("  ⚠️ Skipping V1 angle panel: no world-frame V1 map available")

    if fa_path:
        try:
            viz_paths['profile_dispersion'] = \
                qc_figures.plot_profile_dispersion_panel(
                    cst_left_path=cst_left, cst_right_path=cst_right,
                    scalar_path=fa_path, output_dir=viz_dir,
                    subject_id=args.subject_id, scalar='fa',
                )
            print(f"  ✓ Saved: {viz_paths['profile_dispersion']}")
        except Exception as e:
            print(f"  ⚠️ Could not generate profile dispersion panel: {e}")

        try:
            viz_paths['sampling_saturation'] = \
                qc_figures.plot_sampling_saturation_panel(
                    cst_left_path=cst_left, cst_right_path=cst_right,
                    scalar_path=fa_path, output_dir=viz_dir,
                    subject_id=args.subject_id, scalar='fa',
                )
            print(f"  ✓ Saved: {viz_paths['sampling_saturation']}")
        except Exception as e:
            print(f"  ⚠️ Could not generate sampling saturation panel: {e}")

        try:
            viz_paths['profile_attrition'] = \
                qc_figures.plot_profile_attrition_panel(
                    cst_left_path=cst_left, cst_right_path=cst_right,
                    scalar_path=fa_path, output_dir=viz_dir,
                    subject_id=args.subject_id, scalar='fa',
                )
            print(f"  ✓ Saved: {viz_paths['profile_attrition']}")
        except Exception as e:
            print(f"  ⚠️ Could not generate profile attrition panel: {e}")
    else:
        print("  ⚠️ Skipping dispersion, saturation and attrition panels: "
              "no FA map")

    try:
        viz_paths['node_homology'] = qc_figures.plot_node_homology_panel(
            cst_left_path=cst_left, cst_right_path=cst_right,
            output_dir=viz_dir, subject_id=args.subject_id,
        )
        print(f"  ✓ Saved: {viz_paths['node_homology']}")
    except Exception as e:
        print(f"  ⚠️ Could not generate node homology panel: {e}")


def cmd_metrics(args: argparse.Namespace) -> dict | None:
    """
    Compute bilateral CST metrics and generate reports.
    """
    verbose = getattr(args, 'verbose', True)
    
    # Validate inputs
    if not args.cst_left.exists():
        print(f"  ✗ Left CST tractogram not found: {args.cst_left}")
        return None
    
    if not args.cst_right.exists():
        print(f"  ✗ Right CST tractogram not found: {args.cst_right}")
        return None
    
    args.out.mkdir(parents=True, exist_ok=True)

    print("=" * 60)
    print("BILATERAL CST METRICS")
    print("=" * 60)

    # Load tractograms
    print(f"  → Loading left CST: {args.cst_left}")
    try:
        sft_left = load_tractogram(str(args.cst_left), 'same')
        streamlines_left = sft_left.streamlines
        print(f"  ✓ Loaded {len(streamlines_left):,} streamlines")
    except Exception as e:
        print(f"  ✗ Failed to load left CST: {e}")
        return None

    print(f"  → Loading right CST: {args.cst_right}")
    try:
        sft_right = load_tractogram(str(args.cst_right), 'same')
        streamlines_right = sft_right.streamlines
        print(f"  ✓ Loaded {len(streamlines_right):,} streamlines")
    except Exception as e:
        print(f"  ✗ Failed to load right CST: {e}")
        return None
    
    # Load scalar maps
    fa_map, fa_affine = None, None
    md_map = None
    
    if args.fa:
        if args.fa.exists():
            print(f"  → Loading FA map: {args.fa}")
            fa_map, fa_affine = load_nifti(str(args.fa))
        else:
            print(f"  ⚠️ FA map not found: {args.fa}")
    
    if args.md:
        if args.md.exists():
            print(f"  → Loading MD map: {args.md}")
            md_map, _ = load_nifti(str(args.md))
        else:
            print(f"  ⚠️ MD map not found: {args.md}")
    
    # Load RD and AD maps if provided
    rd_map = None
    ad_map = None
    
    if getattr(args, 'rd', None):
        if args.rd.exists():
            print(f"  → Loading RD map: {args.rd}")
            rd_map, _ = load_nifti(str(args.rd))
        else:
            print(f"  ⚠️ RD map not found: {args.rd}")
    
    if getattr(args, 'ad', None):
        if args.ad.exists():
            print(f"  → Loading AD map: {args.ad}")
            ad_map, _ = load_nifti(str(args.ad))
        else:
            print(f"  ⚠️ AD map not found: {args.ad}")
    
    affine = fa_affine if fa_affine is not None else sft_left.affine
    
    # Analyze hemispheres
    print("\n[Step 1/4] Analyzing left CST...")
    try:
        left_metrics = analyze_cst_hemisphere(
            streamlines=streamlines_left,
            fa_map=fa_map,
            md_map=md_map,
            rd_map=rd_map,
            ad_map=ad_map,
            affine=affine,
            hemisphere='left'
        )
        if verbose:
            print_hemisphere_summary(left_metrics)
    except Exception as e:
        print(f"  ✗ Failed to analyze left CST: {e}")
        return None
    
    print("\n[Step 2/4] Analyzing right CST...")
    try:
        right_metrics = analyze_cst_hemisphere(
            streamlines=streamlines_right,
            fa_map=fa_map,
            md_map=md_map,
            rd_map=rd_map,
            ad_map=ad_map,
            affine=affine,
            hemisphere='right'
        )
        if verbose:
            print_hemisphere_summary(right_metrics)
    except Exception as e:
        print(f"  ✗ Failed to analyze right CST: {e}")
        return None
    
    # Bilateral comparison
    print("\n[Step 3/4] Computing bilateral comparison...")
    try:
        comparison = compare_bilateral_cst(left_metrics, right_metrics)
    except Exception as e:
        print(f"  ✗ Failed during bilateral comparison: {e}")
        return None

    # Node homology: the regional table compares node i on each side, so state
    # how far apart node i actually is. Descriptive only — no threshold, no
    # pass/fail, no suppression of any regional value (plan §8.4).
    try:
        comparison['node_homology'] = _node_homology_summary(
            streamlines_left, streamlines_right
        )
    except Exception as e:
        print(f"  ⚠️ Could not compute node homology: {e}")


    # Save reports
    print("\n[Step 4/4] Generating reports...")
    
    # Get pipeline metadata passed from run.py (or empty if standalone)
    pipeline_metadata = getattr(args, 'pipeline_metadata', {})
    
    # Build metadata for reports from pipeline data
    metadata = {
        'acquisition': pipeline_metadata.get('acquisition', {}),
        'processing': {},
        'qc_thresholds': {
            'fa_threshold': getattr(args, 'fa_threshold', None),
            'min_length': getattr(args, 'min_length', None),
            'max_length': getattr(args, 'max_length', None),
        }
    }
    
    # Build processing metadata from tracking params if available
    tracking_params = pipeline_metadata.get('tracking', {})
    if tracking_params:
        metadata['processing']['tracking_params'] = tracking_params
        # Derive tracking method from params
        sh_order = tracking_params.get('sh_order', 'N/A')
        metadata['processing']['tracking_method'] = f"Deterministic (CSA, SH order {sh_order})"
    
    # Add preprocessing info if available
    if 'preprocessing' in pipeline_metadata:
        metadata['processing']['preprocessing'] = pipeline_metadata['preprocessing']

    # Add full provenance (git, deps, platform, hardware, thread env, command line)
    metadata['provenance'] = get_provenance_dict()
    
    # Add ROI approach (static for now - atlas-based)
    metadata['processing']['roi_approach'] = 'Atlas-to-Subject (Harvard-Oxford)'

    # Add extraction artifact-index diagnostic. Prefer pipeline metadata (run.py);
    # for a standalone `csttool metrics` call, fall back to the on-disk extraction
    # report next to the input tractogram. Populated only when the method is known.
    extraction_raw = pipeline_metadata.get('extraction')
    if not extraction_raw:
        extraction_raw = _find_extraction_statistics(args.cst_left, args.subject_id)
    extraction_meta = _build_extraction_metadata(extraction_raw)
    if extraction_meta is not None:
        metadata['processing']['extraction'] = extraction_meta

    # Remove None values from qc_thresholds
    metadata['qc_thresholds'] = {k: v for k, v in metadata['qc_thresholds'].items() if v is not None}

    
    json_path = None
    csv_path = None
    
    try:
        json_path = save_json_report(comparison, args.out, args.subject_id, metadata=metadata)
        print(f"  ✓ Saved: {json_path}")
    except Exception as e:
        print(f"  ✗ Failed to save JSON report: {e}")
    
    try:
        csv_path = save_csv_summary(comparison, args.out, args.subject_id)
        print(f"  ✓ Saved: {csv_path}")
    except Exception as e:
        print(f"  ✗ Failed to save CSV summary: {e}")
    
    # Generate visualizations
    viz_dir = args.out / "visualizations"
    viz_dir.mkdir(exist_ok=True)
    
    viz_paths = {}
    
    try:
        if fa_map is not None and 'fa' in left_metrics:
            viz_paths['tract_profiles'] = plot_tract_profiles(
                left_metrics, right_metrics, viz_dir, args.subject_id, scalar='fa'
            )
            print(f"  ✓ Saved: {viz_paths['tract_profiles']}")
    except Exception as e:
        print(f"  ⚠️ Could not generate tract profiles: {e}")
    
    try:
        viz_paths['bilateral_comparison'] = plot_bilateral_comparison(
            comparison, viz_dir, args.subject_id
        )
        print(f"  ✓ Saved: {viz_paths['bilateral_comparison']}")
    except Exception as e:
        print(f"  ⚠️ Could not generate bilateral comparison: {e}")

    # Generate report figures and standalone QC visuals when requested.
    # The PDF report consumes the two composite figures (profile_matrix +
    # tractogram_qc_triptych); the standalone per-view QC PNGs are also written
    # for --save-visualizations so sub-*/figures/ still receives the
    # individual views. The triptych shares one grayscale FA scale + colorbar.
    if getattr(args, 'generate_pdf', False) or getattr(args, 'save_visualizations', False):
        try:
            viz_paths['profile_matrix'] = plot_profile_matrix(
                left_metrics, right_metrics, viz_dir, args.subject_id
            )
            print(f"  ✓ Saved: {viz_paths['profile_matrix']}")
        except Exception as e:
            print(f"  ⚠️ Could not generate profile matrix: {e}")

        try:
            if fa_map is not None:
                bg_img = fa_map
                bg_affine = affine
            else:
                 # TODO: Handle case where FA is missing more gracefully if possible
                 # For now, skip if no background
                 bg_img = None
                 bg_affine = None

            if bg_img is not None:
                # Standalone per-view QC PNGs (for figures/ / diagnostics).
                for view in ['axial', 'sagittal', 'coronal']:
                    viz_paths[f'tractogram_qc_{view}'] = plot_tractogram_qc_preview(
                        streamlines_left,
                        streamlines_right,
                        bg_img,
                        bg_affine,
                        viz_dir,
                        args.subject_id,
                        slice_type=view,
                        set_title=False,
                    )
                    print(f"  ✓ Saved: {viz_paths[f'tractogram_qc_{view}']}")
        except Exception as e:
            print(f"  ⚠️ Could not generate tractogram QC: {e}")

        # Composite 1x4 QC strip for the PDF report. Four panels answering four
        # different questions on one shared coronal slice, replacing the 1x3
        # triptych that answered one question from three angles. It reads the
        # persisted products, so it is reproducible from those alone.
        if getattr(args, 'generate_pdf', False) and args.fa:
            try:
                viz_paths['qc_strip'] = plot_report_qc_strip(
                    fa_path=args.fa,
                    v1_path=_resolve_v1_path(args),
                    density_path=_resolve_density_path(args),
                    roi_dseg_path=_resolve_roi_dseg_path(args),
                    cst_left_path=args.cst_left,
                    cst_right_path=args.cst_right,
                    output_dir=viz_dir,
                    subject_id=args.subject_id,
                )
                print(f"  ✓ Saved: {viz_paths['qc_strip']}")
            except Exception as e:
                print(f"  ⚠️ Could not generate report QC strip: {e}")

    # CST-over-FA prototype panel (visualization-refactor M5). Standalone PNG,
    # not embedded in the PDF report until scientific review passes. Uses the
    # same FA + bilateral tractograms the legacy triptych consumes, on a fixed
    # Normalize(0,1) so the two are directly comparable during the migration.
    if getattr(args, 'save_visualizations', False):
        try:
            from csttool.metrics.modules.qc_figures import plot_cst_over_fa_panel
            if fa_map is not None:
                viz_paths['cst_over_fa'] = plot_cst_over_fa_panel(
                    cst_left_path=args.cst_left,
                    cst_right_path=args.cst_right,
                    fa_path=args.fa,
                    output_dir=viz_dir,
                    subject_id=args.subject_id,
                )
                print(f"  ✓ Saved: {viz_paths['cst_over_fa']}")
        except Exception as e:
            print(f"  ⚠️ Could not generate CST-over-FA panel: {e}")

    # Trust-chain QC panels. Each answers one question about whether the numbers
    # in the report are believable, rather than whether the picture looks right:
    # does the tensor support the streamlines (V1 angle), is the mean profile a
    # consensus (dispersion), is it converged in N (saturation), and do the two
    # hemispheres' nodes refer to the same anatomy (homology). Standalone PNGs,
    # not embedded in the PDF report until scientific review passes.
    if getattr(args, 'save_visualizations', False):
        _generate_trust_chain_panels(args, viz_dir, viz_paths)


    # Generate PDF if requested
    pdf_path = None
    html_path = None
    if getattr(args, 'generate_pdf', False):
        try:
            # Use new HTML→PDF pipeline with metadata
            space = getattr(args, 'space', "Native Space")
            
            # First generate HTML report; pass the FA affine so the
            # orientation code is computed dynamically.
            html_path = save_html_report(
                comparison,
                viz_paths,
                args.out,
                args.subject_id,
                space=space,
                metadata=metadata,
                fa_affine=affine,
            )
            print(f"  ✓ Saved: {html_path}")
            
            # Then convert to PDF
            pdf_path = save_pdf_report(
                comparison, 
                viz_paths, 
                args.out, 
                args.subject_id,
                space=space,
                html_path=html_path
            )
            if pdf_path:
                print(f"  ✓ Saved: {pdf_path}")
        except Exception as e:
            print(f"  ⚠️ Could not generate report: {e}")
            import traceback
            traceback.print_exc()
    
    # Summary
    print(f"\n✓ Processing complete")
    print(f"  Subject: {args.subject_id}")
    print(f"  Left CST:  {left_metrics['morphology']['n_streamlines']:,} streamlines, "
          f"volume = {left_metrics['morphology']['tract_volume']/1000:.2f} cm³")
    print(f"  Right CST: {right_metrics['morphology']['n_streamlines']:,} streamlines, "
          f"volume = {right_metrics['morphology']['tract_volume']/1000:.2f} cm³")

    if 'fa' in left_metrics:
        print(f"  FA left:  {left_metrics['fa']['mean']:.3f} +/- {left_metrics['fa']['std']:.3f}")
        print(f"  FA right: {right_metrics['fa']['mean']:.3f} +/- {right_metrics['fa']['std']:.3f}")
        if 'fa' in comparison['asymmetry']:
            print(f"  FA LI:    {comparison['asymmetry']['fa']['laterality_index']:.3f}")
    
    return {
        'json_path': json_path,
        'csv_path': csv_path,
        'pdf_path': pdf_path,
        'comparison': comparison
    }
