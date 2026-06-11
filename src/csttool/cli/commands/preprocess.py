
import argparse
from pathlib import Path
from ..utils import resolve_nifti

def cmd_preprocess(args: argparse.Namespace) -> dict | None:
    """Run the preprocessing pipeline on the input data."""
    from csttool.preprocess import run_preprocessing
    
    try:
        nii = resolve_nifti(args)
    except FileNotFoundError as e:
        print(f"  ✗ {e}")
        return None

    args.out.mkdir(parents=True, exist_ok=True)
    
    # Extract stem from filename
    stem = nii.name.replace(".nii.gz", "").replace(".nii", "")
    input_dir = nii.parent

    # Run the preprocessing pipeline via orchestrator
    result = run_preprocessing(
        input_dir=input_dir,
        output_dir=args.out,
        filename=stem,
        coil_count=args.coil_count,
        denoise_method=getattr(args, 'denoise_method', 'nlmeans'),
        apply_gibbs_correction=getattr(args, 'unring', False),
        apply_motion_correction=getattr(args, 'perform_motion_correction', False),
        target_voxel_size=tuple(args.target_voxel_size) if args.target_voxel_size else None,
        save_visualizations=getattr(args, 'save_visualizations', False),
        verbose=getattr(args, 'verbose', False),
    )

    if result is None:
        return None

    return {
        'preprocessed_path': result['output_paths'].get('data'),
        'motion_correction': result['motion_correction_applied'],
        'stem': stem
    }
