"""
coordinate_validation.py

Validate tractogram and reference image coordinate compatibility.

This module addresses the critical risk of coordinate system mismatches,
which can produce anatomically plausible but incorrect results.
"""

import os

import numpy as np
import nibabel as nib
from dipy.io.streamline import load_tractogram
from dipy.io.stateful_tractogram import Space

# Affine-equality tolerances.
#
# AU22: validate_tractogram_coordinates previously checked only bounding-box
# overlap, not affine equality, so a tractogram and FA map in mismatched
# coordinate spaces could pass when their bounds happened to overlap (GLM §3.10,
# Qwen §3.9). The explicit check below reuses the exact tolerances from
# ``csttool.validation.bundle_comparison.check_spatial_compatibility`` so the
# two validators agree on what counts as a spatial mismatch.
_AFFINE_TOL_TRANS = 1.0   # mm — translation (last affine column)
_AFFINE_TOL_ROT = 1e-3    # rotation/scale (3x3 submatrix), elementwise


def _tractogram_affine(tractogram_path):
    """Return the tractogram's stored voxel→RASMM affine, or None if the
    format carries no spatial header.

    Only ``.trk`` (and ``.trx``) embed an affine; ``.tck`` and the text formats
    do not, so for those the reference image is the only source of truth and no
    affine equality can be asserted from the tractogram alone. Returning None
    lets the caller emit a documented warning rather than silently passing.
    """
    ext = os.path.splitext(str(tractogram_path))[1].lower()
    if ext == ".gz":
        ext = os.path.splitext(str(tractogram_path)[:-3])[1].lower()
    if ext == ".trk":
        # lazy_load avoids reading the full streamline array just for the header
        return nib.streamlines.load(str(tractogram_path), lazy_load=True).affine
    return None


def validate_tractogram_coordinates(
    tractogram_path: str,
    reference_path: str,
    strict: bool = True,
    verbose: bool = True
) -> dict:
    """
    Validate that tractogram coordinates are compatible with reference image.

    This is the primary safeguard against coordinate system mismatches,
    which can produce anatomically plausible but incorrect results.

    Parameters
    ----------
    tractogram_path : str
        Path to tractogram file (.trk, .tck)
    reference_path : str
        Path to reference NIfTI (typically FA map)
    strict : bool
        If True, raise error on critical mismatch. If False, return warnings only.
    verbose : bool
        Print validation details

    Returns
    -------
    result : dict
        Validation result with keys:
        - 'valid': bool - True if validation passed
        - 'warnings': list of warning messages
        - 'errors': list of error messages
        - 'tractogram_info': dict of tractogram properties
        - 'reference_info': dict of reference properties

    Raises
    ------
    ValueError
        If strict=True and critical validation fails
    """
    result = {
        'valid': True,
        'warnings': [],
        'errors': [],
        'tractogram_info': {},
        'reference_info': {}
    }

    # Load reference image
    try:
        ref_img = nib.load(reference_path)
    except Exception as e:
        result['errors'].append(f"Failed to load reference image: {e}")
        result['valid'] = False
        if strict:
            raise ValueError(f"Cannot validate: failed to load reference {reference_path}")
        return result

    ref_affine = ref_img.affine
    ref_shape = ref_img.shape[:3]

    # Compute reference volume bounds in world coordinates
    corners = np.array([
        [0, 0, 0],
        [ref_shape[0], 0, 0],
        [0, ref_shape[1], 0],
        [0, 0, ref_shape[2]],
        [ref_shape[0], ref_shape[1], ref_shape[2]]
    ])
    corners_h = np.hstack([corners, np.ones((corners.shape[0], 1))])
    world_corners = corners_h @ ref_affine.T

    ref_bounds = {
        'x': (float(world_corners[:, 0].min()), float(world_corners[:, 0].max())),
        'y': (float(world_corners[:, 1].min()), float(world_corners[:, 1].max())),
        'z': (float(world_corners[:, 2].min()), float(world_corners[:, 2].max()))
    }

    # Extract orientation
    ref_ornt = nib.orientations.aff2axcodes(ref_affine)

    result['reference_info'] = {
        'shape': tuple(int(s) for s in ref_shape),
        'voxel_size': tuple(float(v) for v in np.abs(np.diag(ref_affine)[:3])),
        'orientation': ''.join(ref_ornt),
        'bounds_mm': ref_bounds
    }

    # ------------------------------------------------------------------
    # AU22: explicit affine-equality assertion (not just bounding-box overlap).
    # The previous check relied on load_tractogram's implicit trk_header_check
    # returning False on a .trk header mismatch — a side effect that is silent
    # (logged, not raised) and does not apply to headerless formats (.tck) at
    # all. A tractogram and FA map in mismatched coordinate spaces can have
    # overlapping world bounds and pass the bounding-box check, then yield
    # anatomically-plausible-but-wrong extraction (GLM §3.10, Qwen §3.9).
    # Tolerances mirror bundle_comparison.check_spatial_compatibility.
    # ------------------------------------------------------------------
    trk_affine = _tractogram_affine(tractogram_path)
    if trk_affine is not None:
        trans_diff = float(np.linalg.norm(trk_affine[:3, 3] - ref_affine[:3, 3]))
        rot_diff = float(np.max(np.abs(trk_affine[:3, :3] - ref_affine[:3, :3])))
        affine_ok = trans_diff <= _AFFINE_TOL_TRANS and rot_diff <= _AFFINE_TOL_ROT
        if not affine_ok:
            msg = (
                "Tractogram and reference image affines do not match "
                f"(translation diff {trans_diff:.3f} mm > {_AFFINE_TOL_TRANS}, "
                f"rotation/scale max diff {rot_diff:.2e} > {_AFFINE_TOL_ROT}). "
                "A bounding-box overlap does not imply the same coordinate "
                "space; extraction on mismatched spaces gives anatomically-"
                "plausible-but-wrong results.\n"
                f"Tractogram affine:\n{trk_affine}\n"
                f"Reference affine:\n{ref_affine}"
            )
            result['errors'].append(msg)
            result['valid'] = False
            if verbose:
                print(f"  ✗ Affine mismatch: {msg}")
            if strict:
                raise ValueError(
                    f"Coordinate validation failed: tractogram and reference "
                    f"affines differ (translation {trans_diff:.3f} mm, "
                    f"rotation/scale {rot_diff:.2e}).\n"
                    f"Ensure the tractogram and FA map are in the same coordinate "
                    f"space (RASMM). Use --skip-coordinate-validation to bypass "
                    f"(NOT RECOMMENDED)."
                )
    else:
        # Headerless format (.tck, text): no embedded affine to assert against.
        # The reference affine is the only source of truth, so a mismatch is
        # not detectable from the files alone — surface this as a warning so
        # the user knows the affine was not verified, rather than a silent pass.
        result['warnings'].append(
            f"Tractogram format '{os.path.splitext(str(tractogram_path))[1]}' "
            "carries no spatial header; affine equality with the reference "
            "image could not be asserted. Verify the tractogram is in the same "
            "RASMM space as the FA map before extraction."
        )

    # Load tractogram with reference
    try:
        sft = load_tractogram(tractogram_path, ref_img, to_space=Space.RASMM)

        # Check if load_tractogram returned a valid SFT (it may return False on error)
        if sft is False or sft is None or not hasattr(sft, 'streamlines'):
            result['errors'].append(
                "Failed to load tractogram with reference: "
                "tractogram header may not match reference image"
            )
            result['valid'] = False
            if strict:
                raise ValueError(
                    f"Cannot validate: tractogram {tractogram_path} incompatible with reference"
                )
            return result

    except Exception as e:
        result['errors'].append(f"Failed to load tractogram with reference: {e}")
        result['valid'] = False
        if strict:
            raise ValueError(f"Cannot validate: failed to load tractogram {tractogram_path}")
        return result

    streamlines = sft.streamlines

    if len(streamlines) == 0:
        result['warnings'].append("Tractogram contains no streamlines")
        result['tractogram_info'] = {'n_streamlines': 0}
        return result

    # Compute streamline bounding box
    all_points = np.vstack([np.asarray(s) for s in streamlines])
    sl_bounds = {
        'x': (float(all_points[:, 0].min()), float(all_points[:, 0].max())),
        'y': (float(all_points[:, 1].min()), float(all_points[:, 1].max())),
        'z': (float(all_points[:, 2].min()), float(all_points[:, 2].max()))
    }

    coord_range = {
        'x': sl_bounds['x'][1] - sl_bounds['x'][0],
        'y': sl_bounds['y'][1] - sl_bounds['y'][0],
        'z': sl_bounds['z'][1] - sl_bounds['z'][0]
    }

    result['tractogram_info'] = {
        'n_streamlines': len(streamlines),
        'n_points': len(all_points),
        'bounds_mm': sl_bounds,
        'coordinate_range': coord_range
    }

    # VALIDATION 1: Check if streamlines appear to be in voxel space
    # Voxel indices typically: non-negative, range 0-256, no large negative values
    min_coord = float(all_points.min())
    max_coord = float(all_points.max())
    max_range = max(coord_range.values())

    if min_coord >= -1 and max_coord < 300 and max_range < 260:
        # Suspicious: looks like voxel indices, not mm
        result['warnings'].append(
            f"Streamline coordinates may be in voxel space "
            f"(range: {min_coord:.1f} to {max_coord:.1f}). "
            "Expected world coordinates in mm with typical range -100 to +100."
        )
        result['errors'].append(
            "Coordinate space mismatch detected: values suggest voxel indices, not mm"
        )
        result['valid'] = False

    # VALIDATION 2: Check if streamlines fall within reference volume bounds
    margin_mm = 15.0  # Allow margin for registration imperfections
    out_of_bounds_axes = []

    for axis in ['x', 'y', 'z']:
        ref_min, ref_max = ref_bounds[axis]
        sl_min, sl_max = sl_bounds[axis]

        if sl_min < ref_min - margin_mm or sl_max > ref_max + margin_mm:
            out_of_bounds_axes.append(axis)
            result['warnings'].append(
                f"Streamlines extend beyond reference {axis.upper()}-axis: "
                f"streamlines [{sl_min:.1f}, {sl_max:.1f}] mm vs "
                f"reference [{ref_min:.1f}, {ref_max:.1f}] mm"
            )

    if len(out_of_bounds_axes) > 1:
        result['errors'].append(
            f"Streamline bounding box significantly exceeds reference volume on "
            f"{len(out_of_bounds_axes)} axes ({', '.join(out_of_bounds_axes)}). "
            "This indicates a coordinate system mismatch."
        )
        result['valid'] = False

    # Print validation summary
    if verbose:
        _print_validation_summary(result, tractogram_path, reference_path)

    # Raise if strict mode and validation failed
    if strict and not result['valid']:
        error_msg = "; ".join(result['errors'])
        raise ValueError(
            f"Coordinate validation failed. This can lead to incorrect results.\n"
            f"Errors: {error_msg}\n"
            f"Ensure tractogram and FA map are in the same coordinate space (RASMM).\n"
            f"Use --skip-coordinate-validation to bypass (NOT RECOMMENDED)."
        )

    return result


def _print_validation_summary(result: dict, tractogram_path: str, reference_path: str):
    """Print formatted validation summary."""
    print()
    print("=" * 60)
    print("COORDINATE VALIDATION")
    print("=" * 60)

    ref = result['reference_info']
    print(f"\nReference: {reference_path}")
    print(f"  Shape: {ref['shape']}")
    print(f"  Orientation: {ref['orientation']}")
    bounds = ref['bounds_mm']
    print(f"  Bounds (mm): X[{bounds['x'][0]:.1f}, {bounds['x'][1]:.1f}], "
          f"Y[{bounds['y'][0]:.1f}, {bounds['y'][1]:.1f}], "
          f"Z[{bounds['z'][0]:.1f}, {bounds['z'][1]:.1f}]")

    tract = result['tractogram_info']
    print(f"\nTractogram: {tractogram_path}")
    print(f"  Streamlines: {tract.get('n_streamlines', 0):,}")

    if tract.get('bounds_mm'):
        bounds = tract['bounds_mm']
        print(f"  Bounds (mm): X[{bounds['x'][0]:.1f}, {bounds['x'][1]:.1f}], "
              f"Y[{bounds['y'][0]:.1f}, {bounds['y'][1]:.1f}], "
              f"Z[{bounds['z'][0]:.1f}, {bounds['z'][1]:.1f}]")

    if result['valid']:
        print("\n  ✓ Coordinate validation passed")
    else:
        print("\n  ✗ Coordinate validation failed:")
        for err in result['errors']:
            print(f"    • {err}")

    for warn in result['warnings']:
        if warn not in [e for e in result['errors']]:
            print(f"  ⚠️ {warn}")

    print("=" * 60)
