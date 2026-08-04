"""
load_dataset.py

Load dataset from DICOM directory or NIfTI file and build gradient table.

"""

import os
from pathlib import Path

import dicom2nifti
import numpy as np

import nibabel as nib
from dipy.io import read_bvals_bvecs

from csttool.defaults import DEFAULT_B0_THRESHOLD
from csttool.preprocess.modules.gradient_validation import (
    validate_gradient_table,
    reorient_dwi_to_ras,
)

def load_dataset(dir_path: str, fname: str):
    """
    Load dataset from DICOM directory or NIfTI file and build gradient table.

    Parameters
    ----------
    dir_path : str
        Path to the directory containing the dataset.
    fname : str
        Name of the file to load.
    
    Returns
    -------
    nii : Nifti1Image
        NIfTI image.
    bval : str
        Path to the bval file.
    bvec : str
        Path to the bvec file.
    gtab : GradientTable
        Gradient table.
    """
    # Check if the directory exists
    dir_path = Path(dir_path)
    if not dir_path.is_dir():
        raise ValueError(f"Directory {dir_path} does not exist")

    # Check if DICOM directory
    if any(f.suffix == ".dcm" for f in dir_path.iterdir()):
        print(f"DICOM directory detected: {dir_path}")
        # Convert DICOM to NIfTI.
        # AU21: dicom2nifti's ``reorient_nifti=True`` reorients the *image* to
        # LAS but leaves the *bvecs* in the original scanner voxel space, so the
        # two become mutually inconsistent (a silent gradient flip that
        # corrupts the tensor fit). We therefore convert with
        # ``reorient_nifti=False`` (image + bvecs both in scanner voxel space,
        # consistent) and reorient *both* to RAS together below, matching the
        # dcm2niix primary path's convention.
        print(f"Converting DICOM to NIfTI...")
        nifti_dir = dir_path.parent / "nifti"
        nifti_dir.mkdir(parents=True, exist_ok=True)  # Create directory if it doesn't exist
        result = dicom2nifti.dicom_series_to_nifti(
            str(dir_path),
            str(nifti_dir / (fname + ".nii.gz")),
            reorient_nifti=False,
        )
        nii_path = result["NII_FILE"]
        bval_path = result.get("BVAL_FILE")
        bvec_path = result.get("BVEC_FILE")

        # Reorient image + bvecs to RAS together and rewrite the bvec sidecar so
        # every later stage (get_gtab_for_preproc reads these same files) sees a
        # consistent image/bvec pair. b-values are orientation-invariant, so the
        # .bval file is left untouched.
        raw_img = nib.load(nii_path)
        _bvals, raw_bvecs = read_bvals_bvecs(bval_path, bvec_path)
        ras_img, ras_bvecs = reorient_dwi_to_ras(raw_img, raw_bvecs)
        nib.save(ras_img, nii_path)
        # bvecs are stored transposed (3, N) per FSL/dicom2nifti convention.
        np.savetxt(bvec_path, ras_bvecs.T, fmt="%.8f")
        nii = ras_img
    else:
        print(f"NIfTI directory detected: {dir_path}")
        nifti_dir = dir_path
        nii_path = os.path.join(dir_path, fname + ".nii.gz")
        
        # Try .bval first, then .bvals for gradient files
        bval_path = os.path.join(dir_path, fname + ".bval")
        if not os.path.exists(bval_path):
            bval_path = os.path.join(dir_path, fname + ".bvals")
        
        bvec_path = os.path.join(dir_path, fname + ".bvec")
        if not os.path.exists(bvec_path):
            bvec_path = os.path.join(dir_path, fname + ".bvecs")
        
        nii = nib.load(nii_path)

    # Read bvalues and bvectors, build a validated gradient table.
    # AU21: validate at load (unit-norm DWI bvecs, non-negative bvals, b0>=1,
    # shape/count match) so a malformed table fails loudly instead of silently
    # corrupting the tensor fit.
    bvals, bvecs = read_bvals_bvecs(bval_path, bvec_path)
    gtab = validate_gradient_table(
        bvals, bvecs, b0_threshold=DEFAULT_B0_THRESHOLD
    )
    num_of_gradients = len(gtab)

    
    # Try to load JSON sidecar
    metadata = {}
    json_path = nifti_dir / (fname + ".json")
    if not json_path.exists():
        # Try finding json with same name as nifti
        json_path = Path(nii_path).with_suffix('').with_suffix('.json')
        
    if json_path.exists():
        try:
            import json
            with open(json_path, 'r') as f:
                sidecar = json.load(f)
                
            # Extract standard BIDS fields
            metadata = {
                'MagneticFieldStrength': sidecar.get('MagneticFieldStrength'),
                'EchoTime': sidecar.get('EchoTime'),
                'RepetitionTime': sidecar.get('RepetitionTime'),
                'Manufacturer': sidecar.get('Manufacturer'),
                'DeviceSerialNumber': sidecar.get('DeviceSerialNumber'),
                'SoftwareVersions': sidecar.get('SoftwareVersions'),
                'dcm2niix_version': sidecar.get('ConversionSoftwareVersion'),
            }
            # Clean up None values
            metadata = {k: v for k, v in metadata.items() if v is not None}
            print(f"  → Loaded metadata from sidecar: {json_path}")
        except Exception as e:
            print(f"  ⚠️ Could not read JSON sidecar: {e}")

    # Add derived fields
    metadata.update({
        'VoxelSize': [float(x) for x in nii.header.get_zooms()[:3]],
        'Dimensions': [int(x) for x in nii.shape],
        'NumVolumes': nii.shape[3] if len(nii.shape) > 3 else 1,
        'NumDirections': len(gtab.bvals) if hasattr(gtab, 'bvals') else 0,
        'MaxBValue': float(gtab.bvals.max()) if hasattr(gtab, 'bvals') and len(gtab.bvals) > 0 else 0,
    })

    print(gtab.info)
    print(f"Number of gradients: {num_of_gradients}")
    print("\n" + "=" * 60)
  
    return nii, gtab, nifti_dir, metadata