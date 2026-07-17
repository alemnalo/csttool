
import argparse
import json
from pathlib import Path

from csttool.defaults import DEFAULT_B0_THRESHOLD

import numpy as np
from dipy.io.image import load_nifti
from dipy.io import read_bvals_bvecs

from csttool.ingest import assess_acquisition_quality
from csttool.tracking.modules.estimate_directions import get_max_sh_order

def cmd_check_dataset(args: argparse.Namespace) -> int:
    """Assess acquisition quality of a DWI dataset."""
    
    # 1. Locate files
    dwi_path = args.dwi
    if not dwi_path.exists():
        print(f"  ✗ DWI file not found: {dwi_path}")
        return 1
        
    # Try to find gradients if not provided
    bval_path = args.bval
    bvec_path = args.bvec
    
    if not bval_path or not bvec_path:
        try:
            if not bval_path:
                candidates = [
                    dwi_path.with_suffix('.bval'),
                    dwi_path.with_name(dwi_path.name.split('.')[0] + '.bval')
                ]
                if dwi_path.name.endswith('.nii.gz'):
                     candidates.append(dwi_path.with_name(dwi_path.name[:-7] + '.bval'))
                
                for c in candidates:
                    if c.exists():
                        bval_path = c
                        break
            
            if not bvec_path:
                 candidates = [
                    dwi_path.with_suffix('.bvec'),
                    dwi_path.with_name(dwi_path.name.split('.')[0] + '.bvec')
                ]
                 if dwi_path.name.endswith('.nii.gz'):
                     candidates.append(dwi_path.with_name(dwi_path.name[:-7] + '.bvec'))
                 
                 for c in candidates:
                    if c.exists():
                        bvec_path = c
                        break
                        
        except Exception:
            pass
            
    if not bval_path or not bval_path.exists():
        print("  ✗ Could not locate .bval file. Please provide --bval.")
        return 1
    if not bvec_path or not bvec_path.exists():
        print("  ✗ Could not locate .bvec file. Please provide --bvec.")
        return 1
        
    # 2. Load Data
    try:
        # Load NIfTI header for voxel size
        img = load_nifti(str(dwi_path), return_img=True)[2]
        header = img.header
        voxel_size = tuple(float(x) for x in header.get_zooms()[:3])
        
        # Load gradients
        bvals, bvecs = read_bvals_bvecs(str(bval_path), str(bvec_path))
    except Exception as e:
        print(f"  ✗ Failed: loading files: {e}")
        return 1
        
    # Load JSON if available
    json_data = {}
    if args.json:
        if args.json.exists():
            try:
                with open(args.json, 'r') as f:
                    json_data = json.load(f)
            except Exception as e:
                print(f"  ⚠️ Could not read JSON: {e}")
        else:
            print(f"  ⚠️ JSON file not found: {args.json}")
            
    # 3. Assess Quality with metadata return
    b0_threshold = getattr(args, 'b0_threshold', DEFAULT_B0_THRESHOLD)
    warnings_list, metadata = assess_acquisition_quality(
        bvecs=bvecs,
        bvals=bvals,
        voxel_size=voxel_size,
        bids_json=json_data,
        b0_threshold=b0_threshold,
        return_metadata=True
    )

    # 4. Generate Enhanced Report
    print("=" * 60)
    print("ACQUISITION QUALITY REPORT")
    print("=" * 60)

    print(f"  Subject/File: {dwi_path.name}")
    print(f"  Scan Date:    {json_data.get('AcquisitionDateTime', 'Unknown')}")

    # B=0 Volume Analysis Section
    print("\n[Step 1/5] B=0 volumes")
    b0_info = metadata['b0_distribution']
    print(f"  Count:       {b0_info['n_volumes']}")
    if b0_info['n_volumes'] > 1:
        print(f"  Maximum gap: {b0_info['max_gap']} volumes")
        if args.verbose:
            print(f"    • Indices: {b0_info['indices']}")

    # Enhanced Acquisition Parameters
    print("\n[Step 2/5] Acquisition parameters")

    # Shell-aware reporting
    if len(metadata['shells']) == 0:
        print("  ⚠️ No DWI shells detected")
    elif len(metadata['shells']) == 1:
        shell = metadata['shells'][0]
        print("  Acquisition type:    Single-shell")
        print(f"  B-value:             {shell['bval']:.0f} s/mm²")
        print(f"  Gradient directions: {shell['n_directions']}")
        print(f"  Total DWI volumes:   {shell['n_volumes']}")
    else:
        print(f"  Acquisition type:    Multi-shell ({len(metadata['shells'])} shells)")
        for i, shell in enumerate(metadata['shells'], 1):
            if args.verbose:
                print(f"    • Shell {i}: b={shell['bval']:.0f} s/mm² "
                      f"({shell['n_directions']} directions, {shell['n_volumes']} volumes)")

    print(f"  Voxel size:          {voxel_size[0]:.2f} x {voxel_size[1]:.2f} x {voxel_size[2]:.2f} mm")

    if 'EchoTime' in json_data:
        try:
            et = float(json_data['EchoTime'])
            et_ms = et * 1000 if et < 1.0 else et
            print(f"  Echo time:           {et_ms:.1f} ms")
        except:
            pass

    if 'MultibandAccelerationFactor' in json_data:
        print(f"  Multiband factor:    {json_data['MultibandAccelerationFactor']}")

    # BIDS Fields Validation
    print("\n[Step 3/5] BIDS metadata")
    bids_fields = metadata.get('bids_fields_present', {})
    critical_fields = ['PhaseEncodingDirection', 'TotalReadoutTime']

    for field in critical_fields:
        if bids_fields.get(field):
            print(f"  ✓ {field} (required for distortion correction)")
        else:
            print(f"  ✗ {field} (required for distortion correction)")

    if args.verbose:
        optional_fields = ['EchoTime', 'MultibandAccelerationFactor']
        for field in optional_fields:
            if bids_fields.get(field):
                print(f"    • {field}: present")
            else:
                print(f"    • {field}: missing")

    # Quality Assessment Section
    print("\n[Step 4/5] Quality assessment")

    if not warnings_list:
        print("  ✓ No quality issues detected")
    else:
        for severity, msg in warnings_list:
            if severity == "CRITICAL":
                print(f"  ✗ [{severity}] {msg}")
            elif severity == "WARNING":
                print(f"  ⚠️ [{severity}] {msg}")
            else:
                print(f"  • [{severity}] {msg}")

    # Enhanced Recommended Settings
    print("\n[Step 5/5] Recommended settings")

    # Use total unique directions for SH order
    n_directions = metadata['n_directions']
    rec_sh = get_max_sh_order(n_directions)
    print(f"  Maximum SH order:    {rec_sh}")

    # Per-shell recommendations in verbose mode
    if args.verbose and len(metadata['shells']) > 1:
        for shell in metadata['shells']:
            shell_sh = get_max_sh_order(shell['n_directions'])
            print(f"    • b={shell['bval']:.0f}: SH order {shell_sh}")

    suggested_step = min(voxel_size) * 0.5
    print(f"  Suggested step size: {suggested_step:.2f} mm")

    # Verbose mode detailed metrics
    if args.verbose:
        print("    • Detailed metrics:")
        print(f"    ├─ Total volumes:    {len(bvals)}")
        print(f"    ├─ B=0 volumes:      {b0_info['n_volumes']}")
        print(f"    ├─ DWI volumes:      {metadata['n_dwi']}")
        print(f"    ├─ Unique directions: {n_directions}")
        print(f"    ├─ B0 threshold:     {b0_threshold} s/mm²")
        voxel_volume = np.prod(voxel_size)
        print(f"    ├─ Voxel volume:     {voxel_volume:.2f} mm³")
        voxel_ratio = max(voxel_size) / min(voxel_size)
        print(f"    └─ Voxel anisotropy: {voxel_ratio:.2f}:1")

    print("\n✓ Quality check complete")

    return 0
