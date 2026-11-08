def save_tracking_outputs(streamlines, img, fa, md, affine, out_dir, stem, rd=None, ad=None, tracking_params=None, provenance=None, verbose=False, *, tenfit=None, white_matter=None, wm_info=None, save_tractogram=True, mask_role="white_matter", tensor_valid_mask=None):
    """Save tractography outputs: tractogram, scalar maps, and processing report.

    Args:
        streamlines (Streamlines): Generated streamlines from run_tractography().
        img (Nifti1Image): Reference NIfTI image for tractogram space.
        fa (ndarray): Fractional anisotropy map (X, Y, Z).
        md (ndarray): Mean diffusivity map (X, Y, Z).
        affine (ndarray): 4x4 affine transformation matrix.
        out_dir (str or Path): Output directory.
        stem (str): Subject/scan identifier for filenames.
        rd (ndarray, optional): Radial diffusivity map (X, Y, Z).
        ad (ndarray, optional): Axial diffusivity map (X, Y, Z).
        tracking_params (dict): Parameters used for tracking (for reproducibility).
        provenance (dict, optional): Provenance information (git hash, versions, platform).
        verbose (bool): Print processing details.
        tenfit: optional DIPY ``TensorFit`` whose principal eigenvector field (V1)
            is rotated to the anatomical world (RAS+) frame and persisted as
            ``{stem}_v1.nii.gz`` with a ``NIFTI_INTENT_VECTOR`` header and a
            self-describing sidecar. ``None`` (the default) skips V1 with a
            warning rather than raising, so existing library callers that do not
            pass the tensor fit are unaffected (visualization-refactoring-plan
            §7.1, R-14). The eigenvectors are read as ``tenfit.evecs[..., :, 0]``
            per DIPY's ``decompose_tensor`` convention.

    Returns:
        dict: Paths to all saved outputs:
            - tractogram: Path to .trk file
            - fa_map: Path to FA NIfTI
            - md_map: Path to MD NIfTI
            - rd_map: Path to RD NIfTI (if provided)
            - ad_map: Path to AD NIfTI (if provided)
            - v1_map: Path to V1 world-frame NIfTI (if ``tenfit`` provided)
            - report: Path to JSON report

    Returns:
        dict: Paths to all saved outputs:
            - tractogram: Path to .trk file
            - fa_map: Path to FA NIfTI
            - md_map: Path to MD NIfTI
            - rd_map: Path to RD NIfTI (if provided)
            - ad_map: Path to AD NIfTI (if provided)
            - report: Path to JSON report
    """
    from pathlib import Path
    from datetime import datetime
    import json
    import numpy as np
    import nibabel as nib
    from dipy.io.stateful_tractogram import StatefulTractogram, Space
    from dipy.io.streamline import save_tractogram
    from dipy.tracking.streamline import length
    from csttool import __version__
    
    out_dir = Path(out_dir)
    
    # Create output directory structure
    tractogram_dir = out_dir / "tractograms"
    scalar_dir = out_dir / "scalar_maps"
    log_dir = out_dir / "logs"

    if save_tractogram:
        tractogram_dir.mkdir(parents=True, exist_ok=True)
    scalar_dir.mkdir(parents=True, exist_ok=True)
    log_dir.mkdir(parents=True, exist_ok=True)
    
    outputs = {}
    
    # 1. Save a whole-brain tractogram only for the legacy whole-brain path.
    if save_tractogram:
        tractogram_path = tractogram_dir / f"{stem}_whole_brain.trk"
        sft = StatefulTractogram(streamlines, img, Space.RASMM)
        save_tractogram(sft, str(tractogram_path))
        outputs['tractogram'] = tractogram_path

        if verbose:
            print(f"  ✓ Saved: {tractogram_path}")

    # 2. Save FA map
    fa_path = scalar_dir / f"{stem}_fa.nii.gz"
    nib.save(nib.Nifti1Image(fa.astype(np.float32), affine), fa_path)
    outputs['fa_map'] = fa_path

    if verbose:
        print(f"  ✓ Saved: {fa_path}")

    # 3. Save MD map
    md_path = scalar_dir / f"{stem}_md.nii.gz"
    nib.save(nib.Nifti1Image(md.astype(np.float32), affine), md_path)
    outputs['md_map'] = md_path

    if verbose:
        print(f"  ✓ Saved: {md_path}")

    # 4. Save RD map (if provided)
    if rd is not None:
        rd_path = scalar_dir / f"{stem}_rd.nii.gz"
        nib.save(nib.Nifti1Image(rd.astype(np.float32), affine), rd_path)
        outputs['rd_map'] = rd_path

        if verbose:
            print(f"  ✓ Saved: {rd_path}")

    # 5. Save AD map (if provided)
    if ad is not None:
        ad_path = scalar_dir / f"{stem}_ad.nii.gz"
        nib.save(nib.Nifti1Image(ad.astype(np.float32), affine), ad_path)
        outputs['ad_map'] = ad_path

        if verbose:
            print(f"  ✓ Saved: {ad_path}")

    if tensor_valid_mask is not None:
        validity_path = scalar_dir / f"{stem}_tensor_valid_mask.nii.gz"
        nib.save(
            nib.Nifti1Image(
                np.asarray(tensor_valid_mask, dtype=np.uint8), affine
            ),
            validity_path,
        )
        outputs['tensor_valid_mask'] = validity_path
        from csttool.bids.output import write_derivative_sidecar
        write_derivative_sidecar(
            validity_path,
            sources=[],
            description=(
                "Tensor-fit validity domain for preferred scalar sampling."
            ),
            command_line=None,
            software_versions=None,
            extra={
                "ValidityRules": [
                    "inside brain mask",
                    "finite FA and eigenvalues",
                    "0 <= FA <= 1",
                    "ordered eigenvalues lambda1 >= lambda2 >= lambda3 >= 0",
                    "positive tensor trace (non-degenerate fitted diffusion)",
                ],
                "NumericScalarThresholdClipping": False,
                "VoxelCount": int(
                    np.asarray(tensor_valid_mask, dtype=bool).sum()
                ),
            },
        )
        if verbose:
            print(f"  ✓ Saved: {validity_path}")

    # 5b. Save V1 principal-eigenvector field (world RAS+ frame), unconditional
    # The eigenvectors come out of DIPY in the b-vec (voxel) frame; we rotate them
    # into the anatomical world frame with the orthonormal polar factor of the
    # affine, so the stored product is self-describing and reproducible by a
    # third party from this file + the FA map using two public DIPY/NumPy ops.
    if tenfit is not None:
        try:
            from csttool.spatial import rotate_vector_field_to_world
            from csttool.bids.output import write_derivative_sidecar
            evecs = np.asarray(tenfit.evecs)
            # Principal eigenvector: evecs[..., :, 0] (DIPY columnar convention).
            v1_voxel = evecs[..., :, 0]
            v1_world, frame_diag = rotate_vector_field_to_world(v1_voxel, affine)
            # Store as (X, Y, Z, 1, 3) float32 with NIFTI_INTENT_VECTOR so the
            # header alone declares it a vector field (FSL's 4D form is not
            # self-describing). Consumers wanting (X,Y,Z,3) squeeze axis 3.
            v1_5d = v1_world.reshape(v1_world.shape[0], v1_world.shape[1],
                                     v1_world.shape[2], 1, 3)
            v1_img = nib.Nifti1Image(v1_5d.astype(np.float32), affine)
            v1_img.header.set_intent('vector')
            v1_path = scalar_dir / f"{stem}_v1.nii.gz"
            nib.save(v1_img, v1_path)
            outputs['v1_map'] = v1_path
            # Sidecar: the frame is the load-bearing key.
            write_derivative_sidecar(
                v1_path,
                sources=[],
                description="Principal diffusion eigenvector (V1), rotated to "
                            "anatomical world (RAS+) axes.",
                command_line=None,
                software_versions=None,
                extra={
                    "VectorFrame": "world-RAS",
                    "VectorFrameSource": "polar-decomposition of voxel-to-RASMM affine linear part",
                    "AffineDeterminant": float(frame_diag["det"]),
                    "ObliquityRad": [float(x) for x in frame_diag["obliquity_rad"]],
                    "ShearMagnitude": float(frame_diag["shear_magnitude"]),
                    "ShearWarning": bool(frame_diag["shear_warning"]),
                    "DerivedFrom": f"{stem}_fa.nii.gz",
                    "EigenvectorConvention": "dipy.reconst.dti decompose_tensor, evecs[..., :, 0]",
                },
            )
            if verbose:
                print(f"  ✓ Saved: {v1_path}")
        except Exception as exc:
            # V1 is a new additive product; a failure must not break the rest of
            # the track stage outputs. Report and continue.
            print(f"  ⚠️ Could not save V1 world-frame field: {exc}")
            import warnings
            warnings.warn(f"V1 world-frame product not written: {exc}")
    else:
        import warnings
        warnings.warn(
            "tenfit not passed to save_tracking_outputs; V1 world-frame "
            "product will not be written. Pass tenfit=... to persist V1.",
            stacklevel=2,
        )

    # Save the domain mask used by this execution path. Preferred mode writes a
    # model/tensor-support mask; legacy mode writes its actual WM mask.
    if white_matter is not None:
        try:
            from csttool.bids.output import write_derivative_sidecar
            info = wm_info or {}
            if mask_role == "model":
                mask_path = scalar_dir / f"{stem}_model_mask.nii.gz"
                description = (
                    "Brain-mask domain used for low-b tensor fitting, CSA peak "
                    "estimation, and preferred propagation-boundary support."
                )
                output_key = "model_mask"
            else:
                mask_path = scalar_dir / f"{stem}_wm_mask.nii.gz"
                description = (
                    "White-matter mask used for legacy seeding and CSA peak "
                    "estimation."
                )
                output_key = "wm_mask"
            nib.save(
                nib.Nifti1Image(
                    np.asarray(white_matter, dtype=np.uint8), affine
                ),
                mask_path,
            )
            outputs[output_key] = mask_path
            extra = {
                "MaskRole": mask_role,
                "MaskSource": info.get("mask_source", info.get("wm_mask_source")),
                "VoxelCount": int(np.asarray(white_matter, dtype=bool).sum()),
            }
            if mask_role == "white_matter":
                extra.update({
                    "WmMaskSource": info.get("wm_mask_source"),
                    "WmMethod": info.get("wm_method"),
                    "WmMaskPath": info.get("wm_mask_path"),
                    "DamParams": info.get("dam_params"),
                    "FallbackReason": info.get("fallback_reason"),
                    "FractionOfBrain": info.get("wm_fraction_of_brain"),
                    "Dilated": info.get("wm_mask_source") == "legacy",
                    "FaThreshold": (
                        tracking_params.get("fa_thresh")
                        if tracking_params else None
                    ),
                    "DerivedFrom": f"{stem}_fa.nii.gz",
                })
            write_derivative_sidecar(
                mask_path,
                sources=[],
                description=description,
                command_line=None,
                software_versions=None,
                extra=extra,
            )
            if verbose:
                print(f"  ✓ Saved: {mask_path}")
        except Exception as exc:
            print(f"  ⚠️ Could not save {mask_role} mask: {exc}")
            import warnings
            warnings.warn(f"{mask_role} mask product not written: {exc}")
    
    # 6. Compute statistics for report
    lengths = np.array([length(s) for s in streamlines]) if len(streamlines) > 0 else np.array([])
    
    fa_valid = fa[fa > 0]
    md_valid = md[md > 0]
    
    # Default tracking parameters if not provided
    if tracking_params is None:
        tracking_params = {}
    
    scalar_stats = {
        'fa_mean': float(fa_valid.mean()) if len(fa_valid) > 0 else 0.0,
        'fa_std': float(fa_valid.std()) if len(fa_valid) > 0 else 0.0,
        'fa_median': float(np.median(fa_valid)) if len(fa_valid) > 0 else 0.0,
        'md_mean': float(md_valid.mean()) if len(md_valid) > 0 else 0.0,
        'md_std': float(md_valid.std()) if len(md_valid) > 0 else 0.0,
        'md_median': float(np.median(md_valid)) if len(md_valid) > 0 else 0.0,
    }
    
    # Add RD stats if available
    if rd is not None:
        rd_valid = rd[rd > 0]
        scalar_stats['rd_mean'] = float(rd_valid.mean()) if len(rd_valid) > 0 else 0.0
        scalar_stats['rd_std'] = float(rd_valid.std()) if len(rd_valid) > 0 else 0.0
        scalar_stats['rd_median'] = float(np.median(rd_valid)) if len(rd_valid) > 0 else 0.0
    
    # Add AD stats if available
    if ad is not None:
        ad_valid = ad[ad > 0]
        scalar_stats['ad_mean'] = float(ad_valid.mean()) if len(ad_valid) > 0 else 0.0
        scalar_stats['ad_std'] = float(ad_valid.std()) if len(ad_valid) > 0 else 0.0
        scalar_stats['ad_median'] = float(np.median(ad_valid)) if len(ad_valid) > 0 else 0.0
    
    output_files = {
        'fa_map': str(fa_path),
        'md_map': str(md_path),
    }
    if 'tractogram' in outputs:
        output_files['tractogram'] = str(outputs['tractogram'])
    if rd is not None:
        output_files['rd_map'] = str(outputs['rd_map'])
    if ad is not None:
        output_files['ad_map'] = str(outputs['ad_map'])
    if 'v1_map' in outputs:
        output_files['v1_map'] = str(outputs['v1_map'])
    if 'wm_mask' in outputs:
        output_files['wm_mask'] = str(outputs['wm_mask'])
    if 'model_mask' in outputs:
        output_files['model_mask'] = str(outputs['model_mask'])
    if 'tensor_valid_mask' in outputs:
        output_files['tensor_valid_mask'] = str(outputs['tensor_valid_mask'])
    
    tracking_report = {
        output_key: tracking_params[source_key]
        for source_key, output_key in (
            ('step_size', 'step_size_mm'),
            ('seed_density', 'seed_density'),
            ('sh_order', 'sh_order'),
            ('sphere', 'sphere'),
            ('relative_peak_threshold', 'relative_peak_threshold'),
            ('min_separation_angle', 'min_separation_angle'),
            ('brain_mask_source', 'brain_mask_source'),
            ('brain_mask_path', 'brain_mask_path'),
            ('tracking_mode', 'tracking_mode'),
            ('tracking_algorithm', 'tracking_algorithm'),
            ('seed_source', 'seed_source'),
            ('model_mask_source', 'model_mask_source'),
            ('scalar_model', 'scalar_model'),
            ('fit_method', 'fit_method'),
            ('tensor_fit', 'tensor_fit'),
            ('direction_model', 'direction_model'),
            ('stage_role', 'stage_role'),
        )
        if source_key in tracking_params
    }
    if 'fa_thresh' in tracking_params:
        tracking_report['fa_threshold'] = tracking_params['fa_thresh']
    if 'stopping_criterion' in tracking_params:
        tracking_report['stopping_criterion'] = tracking_params['stopping_criterion']
    if mask_role == "white_matter":
        tracking_report.update({
            'wm_mask_source': tracking_params.get('wm_mask_source'),
            'wm_mask_path': tracking_params.get('wm_mask_path'),
            'wm_method': tracking_params.get('wm_method'),
            'wm_dam_params': tracking_params.get('wm_dam_params'),
            'wm_fallback_reason': tracking_params.get('wm_fallback_reason'),
            'wm_voxels': tracking_params.get('wm_voxels'),
            'wm_fraction_of_brain': tracking_params.get('wm_fraction_of_brain'),
        })

    report = {
        'processing_info': {
            'date': datetime.now().isoformat(),
            'subject_stem': stem,
            'csttool_version': __version__,
        },
        'tracking_parameters': tracking_report,
        'data_info': {
            'volume_shape': [int(x) for x in img.shape[:3]],
            'voxel_size_mm': [float(x) for x in img.header.get_zooms()[:3]],
            'n_gradients': int(img.shape[3]) if len(img.shape) > 3 else None,
        },
        'streamline_stats': {
            'count': len(streamlines),
            'length_mean_mm': float(lengths.mean()) if len(lengths) > 0 else 0.0,
            'length_std_mm': float(lengths.std()) if len(lengths) > 0 else 0.0,
            'length_min_mm': float(lengths.min()) if len(lengths) > 0 else 0.0,
            'length_max_mm': float(lengths.max()) if len(lengths) > 0 else 0.0,
            'length_median_mm': float(np.median(lengths)) if len(lengths) > 0 else 0.0,
        },
        'scalar_stats': scalar_stats,
        'output_files': output_files
    }

    # Add provenance information if provided
    if provenance is not None:
        report['provenance'] = provenance
    
    # 7. Save report
    report_path = log_dir / f"{stem}_tracking_report.json"
    with open(report_path, 'w') as f:
        json.dump(report, f, indent=2)
    outputs['report'] = report_path
    
    if verbose:
        print(f"  ✓ Saved: {report_path}")
    
    return outputs