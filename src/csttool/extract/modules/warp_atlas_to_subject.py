"""
warp_atlas_to_subject.py

Warp MNI parcellation atlas to subject native space using registration mapping.
Uses Harvard-Oxford atlas from templateflow for ROI definition: https://nilearn.github.io/dev/modules/description/harvard_oxford.html

"""

import numpy as np
import nibabel as nib
from pathlib import Path
from nilearn import image
from nibabel.orientations import apply_orientation, ornt_transform, inv_ornt_aff
from ...data.loader import get_harvard_oxford_path


# Harvard-Oxford label definitions for CST extraction
# Subcortical atlas labels (HarvardOxford-sub)
HARVARDOXFORD_SUBCORTICAL = {
    'brainstem': 8,
    'left_thalamus': 4,
    'right_thalamus': 15,
    'left_caudate': 5,
    'right_caudate': 16,
}

# Cortical atlas labels (HarvardOxford-cort) - maxprob-thr25-2mm version
# Note: Left/Right are distinguished by hemisphere, not separate labels
HARVARDOXFORD_CORTICAL = {
    'precentral_gyrus': 7,
    'postcentral_gyrus': 17,
    'superior_frontal_gyrus': 3,
}

# CST-specific ROI configuration
CST_ROI_CONFIG = {
    'brainstem': {
        'atlas': 'subcortical',
        'label': 8,
        'description': 'Brainstem - inferior CST endpoint'
    },
    'motor_left': {
        'atlas': 'cortical',
        'label': 7,
        'hemisphere': 'left',
        'description': 'Left Precentral Gyrus - superior CST endpoint'
    },
    'motor_right': {
        'atlas': 'cortical', 
        'label': 107,
        'hemisphere': 'right',
        'description': 'Right Precentral Gyrus - superior CST endpoint'
    }
}


def reorient_to_original(data, reorientation_transform):
    """
    Apply inverse of reorientation transform to convert RAS data back to original orientation.
    
    Parameters
    ----------
    data : ndarray
        Data in RAS orientation.
    reorientation_transform : ndarray
        Transform that was used to convert original -> RAS.
        
    Returns
    -------
    reoriented_data : ndarray
        Data in original orientation.
    """
    # Compute inverse transform: RAS -> original
    # The reorientation_transform maps original axes to RAS axes
    # We need to invert this to map RAS -> original
    inverse_transform = ornt_transform(
        nib.io_orientation(np.eye(4)),  # RAS orientation
        reorientation_transform          # Original orientation in transform coords
    )
    # Actually, we need to invert: swap start_ornt and end_ornt
    # reorientation_transform was: ornt_transform(original -> RAS)
    # inverse is: ornt_transform(RAS -> original)
    n_axes = len(reorientation_transform)
    inverse_transform = np.zeros_like(reorientation_transform)
    for i, (axis, flip) in enumerate(reorientation_transform):
        axis = int(axis)
        inverse_transform[axis, 0] = i
        inverse_transform[axis, 1] = flip
    
    return apply_orientation(data, inverse_transform)


def fetch_harvard_oxford(verbose=True):
    """
    Load Harvard-Oxford atlases from user data directory.

    Loads both cortical and subcortical parcellations in MNI152
    space at 1mm resolution.

    The atlases must be fetched using 'csttool fetch-data --accept-fsl-license'.

    Parameters
    ----------
    verbose : bool, optional
        Print progress information. Default is True.

    Returns
    -------
    atlases : dict
        Dictionary containing:
        - 'cortical_path': Path to cortical atlas NIfTI
        - 'subcortical_path': Path to subcortical atlas NIfTI
        - 'cortical_img': Loaded cortical NIfTI image
        - 'subcortical_img': Loaded subcortical NIfTI image

    Raises
    ------
    DataNotInstalledError
        If atlases are not found in user data directory

    Notes
    -----
    Run 'csttool fetch-data --accept-fsl-license' to download FSL-licensed
    atlas data.

    Atlas variants:
    - 'cort-maxprob-thr25-1mm': Cortical, max probability, 25% threshold
    - 'sub-maxprob-thr25-1mm': Subcortical, max probability, 25% threshold
    """
    if verbose:
        print("  → Loading Harvard-Oxford atlases...")

    # Get atlas paths from user data directory
    cort_path = get_harvard_oxford_path("cortical", "1mm")
    subcort_path = get_harvard_oxford_path("subcortical", "1mm")

    # Load the atlas images
    if verbose:
        print("    • Loading cortical atlas...")
    cort_img = nib.load(cort_path)

    if verbose:
        print("    • Loading subcortical atlas...")
    subcort_img = nib.load(subcort_path)

    if verbose:
        print(f"    ✓ Cortical atlas: {cort_path}")
        print(f"    ✓ Subcortical atlas: {subcort_path}")
        print(f"    • Cortical shape: {cort_img.shape}")
        print(f"    • Subcortical shape: {subcort_img.shape}")

    return {
        'cortical_path': str(cort_path),
        'subcortical_path': str(subcort_path),
        'cortical_img': cort_img,
        'subcortical_img': subcort_img,
    }


def split_atlas_hemispheres_mni(atlas_img, verbose=True):
    """
    Split atlas labels into Left and Right hemispheres in MNI space.

    Modifies label values in the Right Hemisphere (X >= 0) by adding an offset of 100.
    Left Hemisphere labels remain unchanged.

    This must be done in MNI space *before* warping to subject space, because
    non-linear registration warps the anatomical midline. Splitting at X=0
    in subject space is inaccurate due to these warps and potential subject offsets.

    Parameters
    ----------
    atlas_img : Nifti1Image
        Atlas image in MNI space (will be reoriented to RAS if needed).
    verbose : bool, optional
        Print progress information.

    Returns
    -------
    modified_img : Nifti1Image
        Atlas image with separate L/R labels in original orientation.
    """
    if verbose:
        print("  → Splitting atlas hemispheres in MNI space...")

    # Check current orientation
    axcodes = nib.orientations.aff2axcodes(atlas_img.affine)
    if verbose:
        print(f"    • Atlas orientation: {axcodes}")

    # Reorient to RAS if needed
    if axcodes != ('R', 'A', 'S'):
        if verbose:
            print(f"    • Reorienting from {axcodes} to RAS for processing...")
        atlas_img = nib.as_closest_canonical(atlas_img)
        axcodes_new = nib.orientations.aff2axcodes(atlas_img.affine)
        if axcodes_new != ('R', 'A', 'S'):
            raise ValueError(
                f"Failed to reorient atlas to RAS. Got {axcodes_new} instead. "
                "This atlas may have an unusual orientation."
            )
    
    data = atlas_img.get_fdata().astype(np.int32)
    affine = atlas_img.affine
    shape = data.shape
    
    # Create coordinate grid in world space
    # Optimization: Only compute X coordinates since that's all we need
    # i index corresponds to x dimension in RAS image space
    i_indices = np.arange(shape[0])
    
    # Compute World X for each i index
    # world_x = affine[0,0]*i + affine[0,1]*j + ... + offset
    # In pure RAS, affine[0,:] controls X. vectorizing:
    # We need to know if the image is rotated. MNI templates usually are aligned.
    
    # safer full meshgrid approach, but memory intensive? 1mm brain is small enough (~8MB floats)
    # actually 182*218*182 is ~7M voxels.
    
    i, j, k = np.meshgrid(
        np.arange(shape[0]),
        np.arange(shape[1]),
        np.arange(shape[2]),
        indexing='ij'
    )
    
    # Calculate World X coordinate
    # x_world = M[0,0]*i + M[0,1]*j + M[0,2]*k + M[0,3]
    x_world = (affine[0, 0] * i + 
               affine[0, 1] * j + 
               affine[0, 2] * k + 
               affine[0, 3])
    
    # In RAS space: X > 0 is Right, X < 0 is Left
    # (Using >= 0 for Right to include midline in Right, or split? 
    # MNI midline is 0. Let's say X >= 0 is Right)
    right_hemisphere_mask = x_world >= 0
    
    # Modify labels in right hemisphere
    # Only modify non-zero labels
    mask_to_modify = right_hemisphere_mask & (data > 0)
    
    # Add offset (e.g., 100) to Right hemisphere labels
    # Label 7 (Precentral) -> 107 (Right Precentral)
    offset = 100
    
    modified_data = data.copy()
    modified_data[mask_to_modify] += offset
    
    if verbose:
        n_modified = np.sum(mask_to_modify)
        print(f"    ✓ Modified {n_modified:,} voxels in Right Hemisphere (Offset +{offset})")

    if verbose:
        # Check label 7 specifically
        orig_7 = np.sum(data == 7)
        new_7 = np.sum(modified_data == 7)
        new_107 = np.sum(modified_data == 107)
        print(f"    • Label 7 split: Total={orig_7}, Left={new_7}, Right={new_107}")
        
    return nib.Nifti1Image(modified_data, affine, atlas_img.header)


def resample_atlas_to_mni_grid(atlas_img, mni_shape, mni_affine, verbose=True):
    """
    Resample a Harvard-Oxford atlas to match the DIPY MNI template grid.
    
    The Harvard-Oxford atlas (182x218x182) has a different grid than the
    DIPY MNI template (197x233x189). Since the registration mapping was
    computed with the DIPY template, we must resample the atlas to the
    same grid before applying the warp.
    
    Parameters
    ----------
    atlas_img : Nifti1Image
        Atlas in Harvard-Oxford grid.
    mni_shape : tuple
        Shape of DIPY MNI template (197, 233, 189).
    mni_affine : ndarray
        4x4 affine matrix of DIPY MNI template.
    verbose : bool, optional
        Print progress information.
        
    Returns
    -------
    resampled_data : ndarray
        Atlas data resampled to MNI template grid.
    """
    atlas_data = atlas_img.get_fdata()
    atlas_affine = atlas_img.affine
    
    
    if verbose:
        print(f"    • Resampling atlas from {atlas_data.shape} to {mni_shape}...")

    # Create a proxy image for the MNI template (target geometry)
    # We only need the grid definition (shape + affine), not the data
    # (Creating a dummy image is cheap)
    mni_proxy = nib.Nifti1Image(np.zeros(mni_shape), mni_affine)

    # Resample atlas to match MNI template grid
    # nilearn.image.resample_to_img robustly handles inconsistent affines/grids
    # by resampling the source image to match the target image's geometry
    resampled_img = image.resample_to_img(
        source_img=atlas_img,
        target_img=mni_proxy,
        interpolation='nearest',
        copy_header=True,
        force_resample=True
    )

    resampled_data = resampled_img.get_fdata()

    if verbose:
        orig_labels = np.unique(atlas_data[atlas_data > 0])
        new_labels = np.unique(resampled_data[resampled_data > 0])
        print(f"    ✓ Resampled: {len(orig_labels)} → {len(new_labels)} labels preserved")
    
    return resampled_data


# Motor-ROI labels for the centroid QC (Harvard-Oxford cortical precentral
# split: 7 = left, 107 = right). Hard-coded here to match CST_ROI_CONFIG and
# the split applied by ``split_atlas_hemispheres_mni``.
_MOTOR_LEFT_LABEL = 7
_MOTOR_RIGHT_LABEL = 107


def compute_atlas_warp_qc(
    warped_atlas,
    orig_labels,
    subject_affine,
    midline_x=0.0,
    motor_left_label=_MOTOR_LEFT_LABEL,
    motor_right_label=_MOTOR_RIGHT_LABEL,
):
    """Build the assertable atlas-warp QC dict (AU33).

    The previous ``warp_atlas_to_subject`` only ``print()``ed its QC checks
    (label-count change, motor-centroid side, motor Z-difference), so they were
    assert-free: a registration that silently dropped a label or swapped a
    hemisphere would warn to stdout and pass. This pure function lifts those
    checks into a dict so a test (or a caller) can assert on them.

    Parameters
    ----------
    warped_atlas : ndarray of int
        Atlas labels after warping to subject space.
    orig_labels : array-like of int
        Unique nonzero labels of the atlas *before* warping (on the grid that
        was actually warped, i.e. after any resample to the MNI grid).
    subject_affine : ndarray, shape (4, 4)
        Affine of the subject grid the atlas was warped onto (world coords).
    midline_x : float, optional
        Subject-space anatomical midline world X (from
        ``compute_warped_midline``). Motor centroids are compared against it.
    motor_left_label, motor_right_label : int, optional
        Label values for the L/R motor cortex. Defaults match the
        Harvard-Oxford cortical split (7 / 107).

    Returns
    -------
    qc : dict
        Always contains ``labels_original``, ``labels_warped``,
        ``labels_preserved`` and ``label_counts``. When both motor labels
        survive warping it additionally carries the motor-centroid fields
        (see ``warp_atlas_to_subject``'s return docstring).
    """
    warped_unique = np.unique(warped_atlas[warped_atlas > 0])
    orig = np.asarray(orig_labels)
    orig_unique = np.unique(orig[orig > 0]) if orig.size else np.array([], dtype=int)

    label_counts = {int(l): int(np.sum(warped_atlas == l)) for l in warped_unique}
    labels_preserved = set(orig_unique.tolist()) == set(warped_unique.tolist())
    qc = {
        'labels_original': sorted(int(l) for l in orig_unique),
        'labels_warped': sorted(int(l) for l in warped_unique),
        'labels_preserved': bool(labels_preserved),
        'label_counts': label_counts,
    }

    if motor_left_label in warped_unique and motor_right_label in warped_unique:
        left_coords = np.array(np.where(warped_atlas == motor_left_label)).T
        right_coords = np.array(np.where(warped_atlas == motor_right_label)).T
        left_centroid = left_coords.mean(axis=0)
        right_centroid = right_coords.mean(axis=0)
        left_world = subject_affine @ np.append(left_centroid, 1)
        right_world = subject_affine @ np.append(right_centroid, 1)
        left_x = float(left_world[0])
        right_x = float(right_world[0])
        left_z = float(left_world[2])
        right_z = float(right_world[2])
        qc.update({
            'motor_left_centroid_world': (left_x, float(left_world[1]), left_z),
            'motor_right_centroid_world': (right_x, float(right_world[1]), right_z),
            'left_centroid_right_of_midline': bool(left_x > midline_x),
            'right_centroid_left_of_midline': bool(right_x < midline_x),
            'motor_z_diff_mm': float(abs(left_z - right_z)),
            'midline_x': float(midline_x),
        })
    return qc


def warp_atlas_to_subject(
    atlas_img,
    mapping,
    subject_shape,
    subject_affine,
    mni_shape=None,
    mni_affine=None,
    interpolation='nearest',
    midline_x=0.0,
    verbose=True
):
    """
    Warp MNI atlas labels to subject native space.
    
    Applies the SyN diffeomorphic mapping (from registration) to transform
    atlas labels from MNI space into the subject's native coordinate system.
    
    If the atlas grid differs from the MNI template grid used for registration,
    the atlas is first resampled to the MNI grid before warping.
    
    Parameters
    ----------
    atlas_img : Nifti1Image
        Atlas label image in MNI space.
    mapping : DiffeomorphicMap
        Registration mapping from `register_mni_to_subject()`.
    subject_shape : tuple
        Shape of subject image (X, Y, Z).
    subject_affine : ndarray
        4x4 affine matrix of subject image.
    mni_shape : tuple, optional
        Shape of MNI template used for registration. Required if atlas
        grid differs from template grid.
    mni_affine : ndarray, optional
        4x4 affine of MNI template used for registration.
    interpolation : str, optional
        Interpolation method. Must be 'nearest' for label maps.
    midline_x : float, optional
        Subject-space midline world X (from ``compute_warped_midline``). Used by
        the motor-ROI centroid QC check (AU11). Defaults to 0.0 (backward
        compatible) but the pipeline always passes the warped-MNI value.
    verbose : bool, optional
        Print progress information.
        
    Returns
    -------
    warped_atlas : ndarray
        Atlas labels warped to subject space. Shape matches subject_shape.
    qc : dict
        Assertable atlas-warp QC (AU33). The previous implementation only
        ``print()``ed these checks, so label-count changes and motor-centroid
        side/Z warnings were assert-free. The dict always contains:
        - ``labels_original`` / ``labels_warped``: sorted lists of unique
          nonzero labels before/after warping.
        - ``labels_preserved``: True iff the warped label *set* equals the
          original (count change alone is necessary but not sufficient — a
          resample can split/merge labels while keeping the count).
        - ``label_counts``: ``{label: voxel_count}`` of the warped atlas.
        When both motor labels (7 and 107) survive warping it additionally
        carries ``motor_left_centroid_world``, ``motor_right_centroid_world``
        (xyz), ``left_centroid_right_of_midline``,
        ``right_centroid_left_of_midline``, ``motor_z_diff_mm`` and
        ``midline_x`` — the same values the verbose path prints as warnings.
    """
    if interpolation != 'nearest':
        raise ValueError(
            f"Interpolation must be 'nearest' for label maps, got '{interpolation}'. "
            "Linear interpolation creates invalid fractional labels."
        )
    
    # Use the *resampled* atlas data (if a resample happened) as the label-set
    # ground truth, so the label-preservation check is not confused by the
    # Harvard-Oxford grid differing from the MNI grid.
    atlas_data_raw = atlas_img.get_fdata()
    atlas_shape = atlas_data_raw.shape
    orig_labels = np.unique(atlas_data_raw[atlas_data_raw > 0])

    if verbose:
        print(f"  → Warping atlas to subject space...")
        print(f"    • Atlas shape: {atlas_shape}")
        print(f"    • Target shape: {subject_shape}")
        print(f"    • Unique labels: {len(orig_labels)}")
        print(f"    • Interpolation: {interpolation}")

    # Check if atlas needs resampling to match MNI template grid
    if mni_shape is not None and atlas_shape != mni_shape:
        if verbose:
            print(f"    ⚠️ Atlas grid {atlas_shape} differs from MNI template {mni_shape}")
        atlas_data = resample_atlas_to_mni_grid(atlas_img, mni_shape, mni_affine, verbose=verbose)
        # Resampling (even nearest-neighbour) can shift a label's voxel set, so
        # recompute the original label set on the resampled grid.
        orig_labels = np.unique(atlas_data[atlas_data > 0])
    else:
        atlas_data = atlas_data_raw

    # Apply the mapping with nearest-neighbor interpolation
    # Registration was: static=subject, moving=MNI template
    # mapping.transform() warps from domain (MNI) to codomain (subject)
    # Must provide image_world2grid to tell DIPY how to interpret the input coordinates
    image_world2grid = np.linalg.inv(mni_affine) if mni_affine is not None else None

    warped_atlas = mapping.transform(
        atlas_data,
        interpolation=interpolation,
        image_world2grid=image_world2grid
    )

    # Ensure integer labels
    warped_atlas = np.round(warped_atlas).astype(np.int16)

    if verbose:
        warped_unique = np.unique(warped_atlas[warped_atlas > 0])
        print(f"    • Warped labels: {len(warped_unique)}")

    # AU33: build the assertable QC dict via the shared helper so the same
    # checks are available to tests/callers without re-deriving them.
    qc = compute_atlas_warp_qc(
        warped_atlas, orig_labels, subject_affine, midline_x=midline_x
    )

    if verbose and not qc['labels_preserved']:
        print(f"    ⚠️ Label set changed "
              f"({len(qc['labels_original'])} → {len(qc['labels_warped'])})")

    # QC: Report motor ROI centroids and bounding boxes (if motor labels present)
    if 'motor_left_centroid_world' in qc:
        left_x, left_y, left_z = qc['motor_left_centroid_world']
        right_x, right_y, right_z = qc['motor_right_centroid_world']
        z_diff = qc['motor_z_diff_mm']
        left_right_of_mid = qc['left_centroid_right_of_midline']
        right_left_of_mid = qc['right_centroid_left_of_midline']

        # AU11: the anatomical midline is world X = midline_x (the warped MNI
        # midline), not necessarily 0. Compare centroids against that plane.
        if verbose:
            print("    • Motor ROI Diagnostics:")
            print(f"    ├─ Left centroid (world):  X={left_x:.1f}, Y={left_y:.1f}, Z={left_z:.1f}")
            print(f"    └─ Right centroid (world): X={right_x:.1f}, Y={right_y:.1f}, Z={right_z:.1f}")

            # Check for obvious issues
            if left_right_of_mid:
                print(f"    ⚠️ Left motor centroid is right of the midline (X={left_x:.1f} > {midline_x:.1f})")
            if right_left_of_mid:
                print(f"    ⚠️ Right motor centroid is left of the midline (X={right_x:.1f} < {midline_x:.1f})")

            if z_diff > 10:
                print(f"    ⚠️ Motor centroids differ by {z_diff:.1f}mm in Z (should be similar)")

    return warped_atlas, qc


def warp_harvard_oxford_to_subject(
    registration_result,
    output_dir=None,
    subject_id=None,
    save_warped=True,
    verbose=True
):
    """
    Complete pipeline: Fetch Harvard-Oxford and warp both atlases to subject space.
    
    Parameters
    ----------
    registration_result : dict
        Output from `register_mni_to_subject()` containing:
        - 'mapping': DiffeomorphicMap
        - 'subject_affine': Subject affine matrix
        - 'subject_shape': Subject image shape
    output_dir : str or Path, optional
        Directory for saving warped atlases. Required if save_warped=True.
    subject_id : str, optional
        Subject identifier for output filenames.
    save_warped : bool, optional
        Save warped atlases as NIfTI files. Default is True.
    verbose : bool, optional
        Print progress information. Default is True.
        
    Returns
    -------
    result : dict
        Dictionary containing:
        - 'cortical_warped': Warped cortical atlas (ndarray)
        - 'subcortical_warped': Warped subcortical atlas (ndarray)
        - 'cortical_warped_path': Path to saved cortical atlas (if saved)
        - 'subcortical_warped_path': Path to saved subcortical atlas (if saved)
        - 'subject_affine': Subject affine (for creating NIfTI)
        - 'roi_config': CST_ROI_CONFIG for downstream use
    """
    if save_warped and output_dir is None:
        raise ValueError("output_dir required when save_warped=True")
    
    if output_dir is not None:
        output_dir = Path(output_dir)
        output_dir.mkdir(parents=True, exist_ok=True)
    
    if verbose:
        print("=" * 60)
        print("WARP ATLAS: Harvard-Oxford → Subject Space")
        print("=" * 60)
    
    # Extract registration components
    mapping = registration_result['mapping']
    subject_shape = registration_result['subject_shape']
    subject_affine = registration_result['subject_affine']  # RAS affine (for warping)
    mni_shape = registration_result.get('mni_shape')
    mni_affine = registration_result.get('mni_affine')

    # AU11 midline (single source of truth for hemisphere splits)
    midline_x = registration_result.get('midline_x', 0.0)
    hemisphere_mask = registration_result.get('hemisphere_mask')
    midline_distance = registration_result.get('midline_distance')

    # Original orientation info (for saving outputs)
    original_subject_affine = registration_result.get('original_subject_affine', subject_affine)
    was_reoriented = registration_result.get('was_reoriented', False)
    reorientation_transform = registration_result.get('reorientation_transform')
    
    # Step 1: Fetch atlases
    if verbose:
        print("\n[Step 1/3] Fetching Harvard-Oxford atlases...")
    
    atlases = fetch_harvard_oxford(verbose=verbose)
    
    # Step 2: Warp subcortical atlas (contains brainstem)
    if verbose:
        print("\n[Step 2/3] Warping subcortical atlas...")
    
    subcortical_warped, subcortical_qc = warp_atlas_to_subject(
        atlas_img=atlases['subcortical_img'],
        mapping=mapping,
        subject_shape=subject_shape,
        subject_affine=subject_affine,
        mni_shape=mni_shape,
        mni_affine=mni_affine,
        midline_x=midline_x,
        verbose=verbose
    )

    # Step 3: Warp cortical atlas (contains precentral gyrus)
    if verbose:
        print("\n[Step 3/3] Warping cortical atlas...")

    # SPLIT HEMISPHERES IN MNI SPACE BEFORE WARPING
    # This fixes the asymmetry issue caused by splitting in subject space
    cortical_split = split_atlas_hemispheres_mni(atlases['cortical_img'], verbose=verbose)

    cortical_warped, cortical_qc = warp_atlas_to_subject(
        atlas_img=cortical_split,
        mapping=mapping,
        subject_shape=subject_shape,
        subject_affine=subject_affine,
        mni_shape=mni_shape,
        mni_affine=mni_affine,
        midline_x=midline_x,
        verbose=verbose
    )
    
    # Prepare result
    result = {
        'cortical_warped': cortical_warped,
        'subcortical_warped': subcortical_warped,
        'cortical_warped_path': None,
        'subcortical_warped_path': None,
        # AU33: assertable atlas-warp QC (label preservation + motor centroids)
        'cortical_qc': cortical_qc,
        'subcortical_qc': subcortical_qc,
        'subject_affine': subject_affine,  # RAS affine (for internal processing)
        'original_subject_affine': original_subject_affine,  # Original affine (for saving)
        'was_reoriented': was_reoriented,
        'reorientation_transform': reorientation_transform,
        'roi_config': CST_ROI_CONFIG,
        # AU11 midline (passed through for downstream hemisphere splits)
        'midline_x': midline_x,
        'hemisphere_mask': hemisphere_mask,
        'midline_distance': midline_distance,
    }
    
    # Save warped atlases
    if save_warped:
        if verbose:
            print("\nSaving warped atlases...")

        nifti_dir = output_dir / "nifti"
        nifti_dir.mkdir(parents=True, exist_ok=True)
        
        prefix = f"{subject_id}_" if subject_id else ""
        
        # Determine which affine and data to use for saving
        # If subject was reoriented for registration, transform data back to original orientation
        if was_reoriented and reorientation_transform is not None:
            if verbose:
                print("    Transforming to original orientation for saving...")
            subcortical_to_save = reorient_to_original(subcortical_warped, reorientation_transform)
            cortical_to_save = reorient_to_original(cortical_warped, reorientation_transform)
            save_affine = original_subject_affine
        else:
            subcortical_to_save = subcortical_warped
            cortical_to_save = cortical_warped
            save_affine = subject_affine
        
        # Save subcortical
        subcort_path = nifti_dir / f"{prefix}harvard_oxford_subcortical_warped.nii.gz"
        nib.save(
            nib.Nifti1Image(subcortical_to_save, save_affine),
            subcort_path
        )
        result['subcortical_warped_path'] = subcort_path
        if verbose:
            print(f"    ✓ Subcortical: {subcort_path}")
        
        # Save cortical
        cort_path = nifti_dir / f"{prefix}harvard_oxford_cortical_warped.nii.gz"
        nib.save(
            nib.Nifti1Image(cortical_to_save, save_affine),
            cort_path
        )
        result['cortical_warped_path'] = cort_path
        if verbose:
            print(f"    ✓ Cortical: {cort_path}")
    
    if verbose:
        print("\n" + "=" * 60)
        print("Atlas warping complete")
        print("=" * 60)
    
    return result


def verify_atlas_labels(warped_atlas, expected_labels, atlas_name="atlas", verbose=True):
    """
    Verify that expected labels exist in warped atlas.
    
    Parameters
    ----------
    warped_atlas : ndarray
        Warped atlas label array.
    expected_labels : list of int
        Label values that should be present.
    atlas_name : str, optional
        Name for verbose output.
    verbose : bool, optional
        Print verification results.
        
    Returns
    -------
    verification : dict
        Dictionary with 'success' bool and 'missing' list.
    """
    present_labels = np.unique(warped_atlas[warped_atlas > 0])
    missing = [l for l in expected_labels if l not in present_labels]
    
    if verbose:
        print(f"\nVerifying {atlas_name} labels:")
        for label in expected_labels:
            status = "✓" if label in present_labels else "✗ MISSING"
            voxel_count = np.sum(warped_atlas == label)
            print(f"    Label {label}: {status} ({voxel_count:,} voxels)")
    
    return {
        'success': len(missing) == 0,
        'missing': missing,
        'present': list(present_labels)
    }