"""
plic_seeding.py

Warp the JHU ICBM-DTI-81 posterior-limb-of-internal-capsule (PLIC) labels into
subject space and turn them into validated bilateral deep-white-matter CST seed
regions. Also builds the independent inferior CST inclusion target (cerebral
peduncle / brainstem, per hemisphere) that accepted streamlines must reach.

The PLIC warp reuses the exact SyN mapping and grid that
``warp_harvard_oxford_to_subject`` uses for the motor/brainstem ROIs, so the
seed regions and the target ROIs live on one consistent grid.
"""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import nibabel as nib

from ...data.loader import (
    get_jhu_labels_path,
    JHU_PLIC_LEFT_LABEL,
    JHU_PLIC_RIGHT_LABEL,
    JHU_ATLAS_VERSION,
)
from .warp_atlas_to_subject import warp_atlas_to_subject


class PlicSeedError(ValueError):
    """Raised when the warped PLIC seeds fail anatomical validation."""


def _centroid_world(mask, affine):
    idx = np.argwhere(mask)
    if len(idx) == 0:
        return None
    ones = np.ones((len(idx), 1))
    world = (np.hstack([idx, ones]) @ np.asarray(affine).T)[:, :3]
    return world.mean(axis=0)


def _to_original_grid(data, reg_result):
    """Return a registration-grid array in the on-disk subject voxel order."""
    from .create_roi_masks import reorient_to_original

    transform = reg_result.get("reorientation_transform")
    result = np.asarray(data)
    if transform is not None:
        result = reorient_to_original(result, transform)
    expected = tuple(reg_result.get("original_subject_shape", result.shape))[:3]
    if result.shape[:3] != expected:
        raise PlicSeedError(
            f"Inverse-reoriented mask shape {result.shape[:3]} does not match "
            f"the tracking grid {expected}."
        )
    return result


def prepare_tracking_grid_masks(
    reg_result,
    roi_masks,
    plic,
    brain_mask,
    tracking_affine,
):
    """Move every registration-derived mask onto the DWI tracking grid.

    Registration operates in an internal RAS voxel order. LocalTracking uses
    the original on-disk DWI array and affine. Mixing those arrays can preserve
    shape while assigning every mask to the wrong physical coordinates, so the
    inverse orientation is mandatory before any intersection or tracking.
    """
    brain_mask = np.asarray(brain_mask, dtype=bool)
    arrays = {
        "plic_left": _to_original_grid(plic["plic_left"], reg_result).astype(bool),
        "plic_right": _to_original_grid(plic["plic_right"], reg_result).astype(bool),
        "motor_left": _to_original_grid(
            roi_masks["motor_left"], reg_result
        ).astype(bool),
        "motor_right": _to_original_grid(
            roi_masks["motor_right"], reg_result
        ).astype(bool),
        "brainstem": _to_original_grid(
            roi_masks["brainstem"], reg_result
        ).astype(bool),
        "hemisphere_mask": _to_original_grid(
            reg_result["hemisphere_mask"], reg_result
        ).astype(bool),
        "midline_distance": _to_original_grid(
            reg_result["midline_distance"], reg_result
        ),
    }

    for name in ("motor_left", "motor_right", "brainstem"):
        if not arrays[name].any():
            raise PlicSeedError(
                f"{name} is empty after conversion to the tracking grid."
            )
    for name, array in arrays.items():
        if array.shape != brain_mask.shape:
            raise PlicSeedError(
                f"{name} shape {array.shape} does not match tracking grid "
                f"{brain_mask.shape}."
            )

    left_raw = int(arrays["plic_left"].sum())
    right_raw = int(arrays["plic_right"].sum())
    arrays["plic_left"] &= brain_mask
    arrays["plic_right"] &= brain_mask
    if not arrays["plic_left"].any() or not arrays["plic_right"].any():
        raise PlicSeedError(
            "Warped PLIC is empty on one side after intersection with the "
            "brain mask on the tracking grid."
        )
    if (arrays["plic_left"] & arrays["plic_right"]).any():
        raise PlicSeedError("Left and right PLIC seeds overlap on the tracking grid.")

    midline_x = float(reg_result.get("midline_x", 0.0))
    plic_centroids = {
        side: _centroid_world(arrays[f"plic_{side}"], tracking_affine)
        for side in ("left", "right")
    }
    motor_centroids = {
        side: _centroid_world(arrays[f"motor_{side}"], tracking_affine)
        for side in ("left", "right")
    }
    if (
        plic_centroids["left"][0] >= midline_x
        or plic_centroids["right"][0] <= midline_x
        or motor_centroids["left"][0] >= midline_x
        or motor_centroids["right"][0] <= midline_x
    ):
        raise PlicSeedError(
            "Registration-derived PLIC or motor masks are on the wrong physical "
            "side after conversion to the tracking grid."
        )

    arrays["inferior_left"] = (
        arrays["brainstem"] & arrays["hemisphere_mask"]
    )
    arrays["inferior_right"] = (
        arrays["brainstem"] & (~arrays["hemisphere_mask"])
    )
    if not arrays["inferior_left"].any() or not arrays["inferior_right"].any():
        raise PlicSeedError("Inferior target is empty on one physical hemisphere.")

    registration_masks = (
        "plic_left", "plic_right", "motor_left", "motor_right",
        "brainstem", "hemisphere_mask", "midline_distance",
    )
    inverse_applied = reg_result.get("reorientation_transform") is not None
    original_affine = np.asarray(
        reg_result.get("original_subject_affine", tracking_affine)
    )
    affine_delta = float(
        np.max(np.abs(original_affine - np.asarray(tracking_affine)))
    )
    centroid_records = {
        "plic_left": plic_centroids["left"],
        "plic_right": plic_centroids["right"],
        "motor_left": motor_centroids["left"],
        "motor_right": motor_centroids["right"],
    }
    expected_sign = {
        "plic_left": -1, "motor_left": -1,
        "plic_right": 1, "motor_right": 1,
    }
    coordinate_qc = {
        "original_dwi_orientation": list(nib.aff2axcodes(tracking_affine)),
        "internal_registration_orientation": "RAS",
        "inverse_reorientation": {
            name: {
                "applied": bool(inverse_applied),
                "transform": (
                    np.asarray(reg_result["reorientation_transform"]).tolist()
                    if inverse_applied else None
                ),
            }
            for name in registration_masks
        },
        "physical_centroids_world_mm": {
            name: [float(value) for value in centroid]
            for name, centroid in centroid_records.items()
        },
        "left_right_signs": {
            name: {
                "x_relative_to_midline_mm": float(centroid[0] - midline_x),
                "observed_sign": (
                    -1 if centroid[0] < midline_x
                    else (1 if centroid[0] > midline_x else 0)
                ),
                "expected_sign": expected_sign[name],
                "pass": (
                    (-1 if centroid[0] < midline_x else 1)
                    == expected_sign[name]
                ),
            }
            for name, centroid in centroid_records.items()
        },
        "affine_compatibility": {
            "pass": bool(np.allclose(
                original_affine, np.asarray(tracking_affine),
                rtol=0.0, atol=1e-5,
            )),
            "maximum_absolute_difference": affine_delta,
        },
        "shape_compatibility": {
            name: {
                "shape": list(arrays[name].shape),
                "expected_shape": list(brain_mask.shape),
                "pass": arrays[name].shape == brain_mask.shape,
            }
            for name in arrays
        },
        "nonempty_masks": {
            name: {
                "voxel_count": int(np.count_nonzero(arrays[name])),
                "pass": bool(np.any(arrays[name])),
            }
            for name in (
                "plic_left", "plic_right", "motor_left", "motor_right",
                "brainstem", "inferior_left", "inferior_right",
            )
        },
    }
    coordinate_qc["overall_pass"] = bool(
        coordinate_qc["affine_compatibility"]["pass"]
        and all(
            item["pass"]
            for item in coordinate_qc["shape_compatibility"].values()
        )
        and all(
            item["pass"] for item in coordinate_qc["nonempty_masks"].values()
        )
        and all(
            item["pass"] for item in coordinate_qc["left_right_signs"].values()
        )
    )

    provenance = dict(plic.get("provenance", {}))
    provenance.update({
        "brain_intersected": True,
        "voxels_before_brain_intersection": {
            "left": left_raw,
            "right": right_raw,
        },
        "voxels_final": {
            "left": int(arrays["plic_left"].sum()),
            "right": int(arrays["plic_right"].sum()),
        },
        "centroid_world_left": [
            float(value) for value in plic_centroids["left"]
        ],
        "centroid_world_right": [
            float(value) for value in plic_centroids["right"]
        ],
        "tracking_grid": {
            "shape": list(brain_mask.shape),
            "orientation": list(nib.aff2axcodes(tracking_affine)),
            "inverse_reorientation_applied": bool(inverse_applied),
            "original_to_ras_orientation_transform": (
                np.asarray(reg_result["reorientation_transform"]).tolist()
                if inverse_applied else None
            ),
        },
    })
    arrays["plic_provenance"] = provenance
    arrays["coordinate_frame_qc"] = coordinate_qc
    return arrays



def save_preferred_seed_masks(
    left_mask,
    right_mask,
    affine,
    output_dir,
    subject_id,
    provenance,
):
    """Persist the exact physical seed masks consumed by LocalTracking."""
    nifti_dir = Path(output_dir) / "nifti"
    nifti_dir.mkdir(parents=True, exist_ok=True)
    paths = {}
    for side, mask in (("left", left_mask), ("right", right_mask)):
        path = (
            nifti_dir
            / f"{subject_id}_space-orig_desc-preferredSeed-{side}_mask.nii.gz"
        )
        nib.save(
            nib.Nifti1Image(np.asarray(mask, dtype=np.uint8), affine),
            path,
        )
        sidecar = path.with_name(path.name[:-7] + ".json")
        sidecar.write_text(
            json.dumps(
                {
                    "Description": (
                        "Exact seed mask consumed by preferred CST LocalTracking"
                    ),
                    "Hemisphere": side,
                    "VoxelCount": int(np.asarray(mask, dtype=bool).sum()),
                    "Provenance": provenance,
                },
                indent=2,
            )
            + "\n"
        )
        paths[side] = path
        paths[f"{side}_sidecar"] = sidecar
    return paths

def warp_plic_seeds(reg_result, brain_mask=None, *, resolution="1mm",
                    midline_tol_mm=8.0, verbose=True):
    """Warp PLIC L/R into subject space and validate them as CST seeds.

    Parameters
    ----------
    reg_result : dict
        Output of ``register_mni_to_subject`` (mapping, subject grid, MNI grid,
        hemisphere_mask, midline_x).
    brain_mask : ndarray, optional
        Subject brain mask on the warped grid; PLIC is intersected with it when
        shapes match so seeds never fall outside brain support.

    Returns
    -------
    dict with ``plic_left`` / ``plic_right`` (bool arrays on the warped grid) and
    ``provenance`` (atlas version, labels, transform, interpolation, hemisphere
    split source, seed voxel counts, centroids).
    """
    mapping = reg_result['mapping']
    subject_shape = reg_result['subject_shape']
    subject_affine = reg_result['subject_affine']
    mni_shape = reg_result.get('mni_shape')
    mni_affine = reg_result.get('mni_affine')
    midline_x = reg_result.get('midline_x', 0.0)
    hemisphere_mask = reg_result.get('hemisphere_mask')

    jhu_path = get_jhu_labels_path(resolution)
    jhu_img = nib.load(str(jhu_path))

    if verbose:
        print(f"  → Warping JHU PLIC labels ({JHU_ATLAS_VERSION})")

    warped, _qc = warp_atlas_to_subject(
        jhu_img, mapping, subject_shape, subject_affine,
        mni_shape=mni_shape, mni_affine=mni_affine,
        interpolation='nearest', midline_x=midline_x, verbose=False,
    )

    plic_left = warped == JHU_PLIC_LEFT_LABEL
    plic_right = warped == JHU_PLIC_RIGHT_LABEL
    n_left_raw, n_right_raw = int(plic_left.sum()), int(plic_right.sum())

    # Hemisphere clamp against the warped-MNI L/R label (gold standard), so a SyN
    # warp that bleeds a few voxels across the midline cannot put a "left" seed
    # in the right hemisphere.
    split_source = "none"
    if hemisphere_mask is not None and hemisphere_mask.shape == plic_left.shape:
        plic_left = plic_left & hemisphere_mask
        plic_right = plic_right & (~hemisphere_mask)
        split_source = "warped_mni_hemisphere_mask"

    # Intersect with brain support when the grids match.
    brain_intersected = False
    if brain_mask is not None and np.asarray(brain_mask).shape == plic_left.shape:
        bm = np.asarray(brain_mask, dtype=bool)
        plic_left = plic_left & bm
        plic_right = plic_right & bm
        brain_intersected = True

    n_left, n_right = int(plic_left.sum()), int(plic_right.sum())

    # ---- validation --------------------------------------------------------
    if n_left == 0 or n_right == 0:
        raise PlicSeedError(
            f"Warped PLIC empty on one side after clamping "
            f"(left={n_left}, right={n_right}). Registration or hemisphere "
            f"labelling is implausible; refusing to seed.")

    if plic_left.shape == plic_right.shape and int((plic_left & plic_right).sum()) > 0:
        raise PlicSeedError("Left and right PLIC seeds overlap after clamping.")

    cen_left = _centroid_world(plic_left, subject_affine)
    cen_right = _centroid_world(plic_right, subject_affine)
    # Left CST structures sit in the left hemisphere: world X < midline_x.
    if cen_left[0] >= midline_x or cen_right[0] <= midline_x:
        raise PlicSeedError(
            f"PLIC centroids on the wrong side of the midline "
            f"(left X={cen_left[0]:.1f}, right X={cen_right[0]:.1f}, "
            f"midline X={midline_x:.1f}).")

    provenance = {
        "atlas": JHU_ATLAS_VERSION,
        "atlas_path": str(jhu_path),
        "labels": {"left": JHU_PLIC_LEFT_LABEL, "right": JHU_PLIC_RIGHT_LABEL},
        "transform": "SyN diffeomorphic (register_mni_to_subject) + affine",
        "interpolation": "nearest",
        "hemisphere_split_source": split_source,
        "brain_intersected": brain_intersected,
        "voxels_raw": {"left": n_left_raw, "right": n_right_raw},
        "voxels_final": {"left": n_left, "right": n_right},
        "centroid_world_left": [float(x) for x in cen_left],
        "centroid_world_right": [float(x) for x in cen_right],
        "midline_x": float(midline_x),
    }

    if verbose:
        print(f"    • PLIC left:  {n_left:,} voxels  centroid X={cen_left[0]:.1f}")
        print(f"    • PLIC right: {n_right:,} voxels  centroid X={cen_right[0]:.1f}")

    return {"plic_left": plic_left, "plic_right": plic_right,
            "provenance": provenance}


def build_inferior_targets(brainstem_mask, reg_result, verbose=True):
    """Independent inferior CST inclusion target, split by hemisphere.

    The caudal target a valid CST branch must reach: the brainstem (cerebral
    peduncle / ventral pons the CST descends through), split L/R by the same
    warped-MNI hemisphere label the motor ROIs use. This reuses the existing
    brainstem ROI as an *inclusion* target, kept independent of the PLIC seed.
    """
    brainstem = np.asarray(brainstem_mask, dtype=bool)
    hemisphere_mask = reg_result.get('hemisphere_mask')
    if hemisphere_mask is not None and hemisphere_mask.shape == brainstem.shape:
        inf_left = brainstem & hemisphere_mask
        inf_right = brainstem & (~hemisphere_mask)
    else:
        # No hemisphere label: both sides may reach either brainstem half. Fall
        # back to the whole brainstem for each (still an inferior constraint).
        inf_left = brainstem
        inf_right = brainstem
    if verbose:
        print(f"  → Inferior targets: left {int(inf_left.sum()):,} / "
              f"right {int(inf_right.sum()):,} voxels (brainstem, hemisphere-split)")
    return inf_left, inf_right
