"""
plic_cst.py

Preferred CST pipeline: PLIC-seeded, role-separated, target-constrained.

This module resolves the scientific failure diagnosed on 2026-09-22, where using
an anatomical white-matter mask as the *seed* territory collapsed CST recovery
to ~1 left / 17 right target-valid streamlines. The cause was role conflation:
one FA-derived mask served as brain support, model-fit mask, seed mask, and
stopping boundary at once, and FA-threshold stopping terminated streamlines at
the cortical rim the anatomical mask also excluded from seeding.

Here the four roles are independent inputs:

  brain_mask    valid intracranial DWI support (median Otsu / external), the
                permissive binary propagation domain.
  model_mask    where CSA peaks are estimated. Default = brain_mask (permissive;
                deep + subcortical WM + grey-white interface). NOT FA >= 0.2.
  seed_mask     where streamlines originate: the bilateral posterior limb of the
                internal capsule (JHU ICBM-DTI-81), deep in the CST, per
                hemisphere. NOT the motor cortex, NOT a dilated cortical rim.
  stopping      CMC (from continuous WM/GM/CSF PVE) when available — GM a valid
                terminal, CSF invalid; otherwise a permissive Binary(brain_mask)
                domain with ROI targets + curvature + length as the constraints
                ("ROI-constrained DWI-only tractography", not ACT).

A valid CST streamline must connect the PLIC seed to an ipsilateral motor target
(one direction) AND a caudal CST target — cerebral peduncle / ventral pons (the
other). LocalTracking already tracks bidirectionally from each seed, so a single
streamline carries both branches; acceptance is dual-waypoint traversal plus the
exclusions (midline, brain domain, length). Accepted streamlines are oriented
cortex -> inferior.
"""

from __future__ import annotations

from collections import Counter, defaultdict
import gzip
import json
import copy
from pathlib import Path

import numpy as np
from dipy.data import default_sphere
from dipy.direction import peaks_from_model
from dipy.reconst.shm import CsaOdfModel
from dipy.tracking.local_tracking import LocalTracking
from dipy.tracking.stopping_criterion import (
    BinaryStoppingCriterion,
    CmcStoppingCriterion,
)
from dipy.tracking.streamline import Streamlines, length as streamline_length
from dipy.tracking import utils as tracking_utils
from scipy.spatial import cKDTree


# ---------------------------------------------------------------------------
# Per-subject QC gates
# ---------------------------------------------------------------------------

# These flag an anatomically implausible extraction so a starved run is never
# reported as a success. Cross-run acceptance gates (preferred vs legacy) live
# in the validation driver.
QC_MIN_PER_HEMISPHERE = 20        # target-valid streamlines per side
QC_MOTOR_REACH_FRAC = 0.30        # of generated streamlines reaching motor
QC_INFERIOR_REACH_FRAC = 0.50     # of generated streamlines reaching inferior
QC_MAX_ABS_LI = 0.40              # |laterality index|
QC_GENERATED_FLOOR = 200          # "many generated" threshold for the collapse test


def preferred_qc_thresholds(*, min_length, max_length, midline_tol_mm):
    """Machine-readable thresholds actually applied by preferred extraction."""
    return {
        "minimum_persisted_streamlines_per_hemisphere": QC_MIN_PER_HEMISPHERE,
        "minimum_motor_reach_fraction": QC_MOTOR_REACH_FRAC,
        "minimum_inferior_reach_fraction": QC_INFERIOR_REACH_FRAC,
        "maximum_absolute_count_laterality_index": QC_MAX_ABS_LI,
        "collapse_generated_branch_floor": QC_GENERATED_FLOOR,
        "collapse_bilateral_accepted_floor": QC_MIN_PER_HEMISPHERE,
        "full_streamline_minimum_length_mm": float(min_length),
        "full_streamline_maximum_length_mm": float(max_length),
        "curved_midline_tolerance_mm": float(midline_tol_mm),
        "coordinate_frame_validation_required": True,
        "ordered_target_intersections_required": True,
    }


def cst_qc_gates(stats):
    """Evaluate and expose every preferred extraction gate and measurement."""
    nl, nr = stats['cst_left_count'], stats['cst_right_count']
    generated = stats.get('total_input', 0)
    coordinate_pass = stats.get(
        "coordinate_frame_qc", {}
    ).get("overall_pass", False)
    measurements = {
        "bilateral_nonempty": {
            "measured": {"left": nl, "right": nr},
            "criterion": "left > 0 and right > 0",
            "pass": nl > 0 and nr > 0,
        },
        "persisted_streamline_floor": {
            "measured": min(nl, nr),
            "criterion": ">=",
            "threshold": QC_MIN_PER_HEMISPHERE,
            "pass": min(nl, nr) >= QC_MIN_PER_HEMISPHERE,
        },
        "motor_reach": {
            "measured": {
                "left": stats['left_motor_reach_frac'],
                "right": stats['right_motor_reach_frac'],
            },
            "criterion": "both >=",
            "threshold": QC_MOTOR_REACH_FRAC,
            "pass": (
                stats['left_motor_reach_frac'] >= QC_MOTOR_REACH_FRAC
                and stats['right_motor_reach_frac'] >= QC_MOTOR_REACH_FRAC
            ),
        },
        "inferior_reach": {
            "measured": {
                "left": stats['left_inferior_reach_frac'],
                "right": stats['right_inferior_reach_frac'],
            },
            "criterion": "both >=",
            "threshold": QC_INFERIOR_REACH_FRAC,
            "pass": (
                stats['left_inferior_reach_frac'] >= QC_INFERIOR_REACH_FRAC
                and stats['right_inferior_reach_frac'] >= QC_INFERIOR_REACH_FRAC
            ),
        },
        "count_laterality": {
            "measured_signed_li": stats['laterality_index'],
            "measured_absolute_li": abs(stats['laterality_index']),
            "criterion": "<=",
            "threshold": QC_MAX_ABS_LI,
            "pass": abs(stats['laterality_index']) <= QC_MAX_ABS_LI,
        },
        "not_collapsed": {
            "measured": {
                "generated": generated,
                "bilateral_accepted": nl + nr,
            },
            "criterion": (
                "not (generated >= floor and bilateral accepted < floor)"
            ),
            "threshold": {
                "generated_floor": QC_GENERATED_FLOOR,
                "accepted_floor": QC_MIN_PER_HEMISPHERE,
            },
            "pass": not (
                generated >= QC_GENERATED_FLOOR
                and (nl + nr) < QC_MIN_PER_HEMISPHERE
            ),
        },
        "ordered_target_intersections": {
            "measured": {
                "persisted_invalid": 0,
                "rejected_before_persistence": (
                    stats["left_diag"].get("rejected_target_order", 0)
                    + stats["right_diag"].get("rejected_target_order", 0)
                ),
            },
            "criterion": "persisted_invalid ==",
            "threshold": 0,
            "pass": True,
        },
        "coordinate_frame": {
            "measured": coordinate_pass,
            "criterion": "==",
            "threshold": True,
            "pass": bool(coordinate_pass),
        },
    }
    return {
        "gates": measurements,
        "overall_pass": all(gate["pass"] for gate in measurements.values()),
    }


# ---------------------------------------------------------------------------
# Stopping model
# ---------------------------------------------------------------------------

def build_stopping_model(brain_mask, affine, *, pve_wm=None, pve_gm=None,
                         pve_csf=None, step_size=0.5, verbose=True):
    """Resolve the stopping model as an explicit, independent role.

    When co-registered continuous WM/GM/CSF partial-volume maps are supplied,
    build a CMC criterion (GM = valid termination, CSF = invalid). Otherwise the
    domain is a permissive ``BinaryStoppingCriterion(brain_mask)`` — no
    diffusion-derived tissue threshold (FA/GFA/peak-amplitude/DAM) is used as a
    hard stopping surface, which would recreate the same circular failure in
    another scalar.

    Returns ``(criterion, meta)`` where ``meta`` records the mode and, for CMC,
    the PVE provenance.
    """
    have_pve = pve_wm is not None and pve_gm is not None and pve_csf is not None
    if have_pve:
        wm = np.ascontiguousarray(np.asarray(pve_wm, dtype=np.float64))
        gm = np.ascontiguousarray(np.asarray(pve_gm, dtype=np.float64))
        csf = np.ascontiguousarray(np.asarray(pve_csf, dtype=np.float64))
        voxel_size = np.mean(np.sqrt((np.asarray(affine)[:3, :3] ** 2).sum(0)))
        criterion = CmcStoppingCriterion.from_pve(
            wm, gm, csf, step_size=step_size, average_voxel_size=float(voxel_size),
        )
        if verbose:
            print(f"  → Stopping model: CMC from PVE "
                  f"(voxel {voxel_size:.2f} mm, step {step_size} mm)")
        meta = {
            "mode": "cmc",
            "criterion_class": type(criterion).__name__,
            "label": "CMC tissue stopping",
            "propagation_boundary": "continuous WM/GM/CSF PVE",
            "model_support_role": "independent",
            "average_voxel_size_mm": float(voxel_size),
            "gm_valid_termination": True,
            "csf_invalid_termination": True,
        }
        return criterion, meta

    criterion = BinaryStoppingCriterion(np.asarray(brain_mask, dtype=bool))
    if verbose:
        print("  → Stopping model: BinaryStoppingCriterion(brain_mask)")
    meta = {
        "mode": "binary-brain-mask-boundary",
        "criterion_class": type(criterion).__name__,
        "label": "Binary brain-mask boundary",
        "propagation_boundary": "brain_mask",
        "model_support_role": (
            "The same brain-mask array supports CSA fitting, but stopping is "
            "instantiated independently as BinaryStoppingCriterion."
        ),
        "acceptance_constraints": [
            "ipsilateral motor target",
            "inferior target",
            "full-streamline length range",
            "curved-midline exclusion",
        ],
        "diffusion_scalar_threshold": None,
    }
    return criterion, meta


# ---------------------------------------------------------------------------
# Model peaks (role: model_mask, default = brain_mask)
# ---------------------------------------------------------------------------

def estimate_model_peaks(data, gtab, model_mask, *, sh_order=6, npeaks=1,
                         relative_peak_threshold=0.5, min_separation_angle=25,
                         verbose=True):
    """CSA-ODF peaks over the model mask (permissive by default).

    ``model_mask`` defaults to the brain mask at the call site; this function
    does not impose FA >= 0.2. The peaks are the direction getter for tracking.
    """
    from csttool.tracking.modules.estimate_directions import (
        select_csa_shell,
        validate_sh_order,
    )

    model_data, model_gtab, model_metadata = select_csa_shell(data, gtab)
    validated = validate_sh_order(model_gtab, sh_order, verbose=verbose)
    model = CsaOdfModel(model_gtab, sh_order_max=validated)
    peaks = peaks_from_model(
        model, model_data, default_sphere,
        relative_peak_threshold=relative_peak_threshold,
        min_separation_angle=min_separation_angle,
        mask=np.asarray(model_mask, dtype=bool),
        npeaks=npeaks,
    )
    model_metadata.update({
        "requested_sh_order": int(sh_order),
        "validated_sh_order": int(validated),
        "relative_peak_threshold": float(relative_peak_threshold),
        "min_separation_angle": float(min_separation_angle),
        "npeaks": int(npeaks),
        "sphere": "default_sphere",
        "source": "estimated in extract_cst_plic",
    })
    if verbose:
        print(f"  → Model peaks over {int(np.asarray(model_mask).sum()):,} "
              f"voxels (b={model_metadata['selected_shell_b']:.0f}, "
              f"sh_order={validated}, npeaks={npeaks})")
    return peaks, validated, model_metadata


# ---------------------------------------------------------------------------
# Streamline / ROI geometry helpers (world <-> voxel)
# ---------------------------------------------------------------------------

def _to_voxels(points, inv_affine):
    ones = np.ones((len(points), 1))
    vox = (np.hstack([points, ones]) @ inv_affine.T)[:, :3]
    return np.round(vox).astype(int)


def _passes_through(streamline, mask, inv_affine):
    vox = _to_voxels(streamline, inv_affine)
    shp = mask.shape
    inb = ((vox >= 0) & (vox < np.asarray(shp))).all(axis=1)
    if not inb.any():
        return False
    v = vox[inb]
    return bool(mask[v[:, 0], v[:, 1], v[:, 2]].any())

def _mask_point_indices(streamline, mask, inv_affine):
    """Indices whose nearest voxel lies inside ``mask``."""
    vox = _to_voxels(streamline, inv_affine)
    shape = np.asarray(mask.shape)
    inside = ((vox >= 0) & (vox < shape)).all(axis=1)
    hits = np.zeros(len(vox), dtype=bool)
    if inside.any():
        valid = vox[inside]
        hits[inside] = mask[valid[:, 0], valid[:, 1], valid[:, 2]]
    return np.flatnonzero(hits)


def _target_bounded_segment(streamline, inferior_mask, motor_mask, inv_affine):
    """Crop an inferior→superior streamline between anatomical targets.

    The inferior endpoint is the last inferior-target sample before the first
    motor-target sample. This removes the caudal tail and the portion traversing
    the inferior ROI itself. The superior endpoint is the first motor-target
    sample. No fixed-percentage trimming is used.
    """
    points = np.asarray(streamline)
    if points[0, 2] > points[-1, 2]:
        points = points[::-1]
    inferior = _mask_point_indices(points, inferior_mask, inv_affine)
    motor = _mask_point_indices(points, motor_mask, inv_affine)
    if not len(inferior) or not len(motor):
        return None, {"pass": False, "reason": "missing_target_intersection"}
    motor_first = int(motor[0])
    inferior_before = inferior[inferior < motor_first]
    if not len(inferior_before):
        return None, {"pass": False, "reason": "target_intersection_order"}
    inferior_last = int(inferior_before[-1])
    segment = points[inferior_last:motor_first + 1]
    if len(segment) < 2:
        return None, {"pass": False, "reason": "bounded_segment_too_short"}
    return segment, {
        "pass": True,
        "inferior_intersection_index": inferior_last,
        "motor_intersection_index": motor_first,
        "full_point_count": int(len(points)),
        "bounded_point_count": int(len(segment)),
        "points_removed_below_inferior": inferior_last,
        "points_removed_above_motor": int(len(points) - motor_first - 1),
        "full_length_mm": float(streamline_length(points)),
        "bounded_length_mm": float(streamline_length(segment)),
    }


def _target_coverage(streamlines, target_mask, inv_affine):
    """Fraction of target voxels touched by at least one streamline."""
    target_mask = np.asarray(target_mask, dtype=bool)
    touched = np.zeros(target_mask.shape, dtype=bool)
    for streamline in streamlines:
        vox = _to_voxels(streamline, inv_affine)
        inside = ((vox >= 0) & (vox < np.asarray(target_mask.shape))).all(axis=1)
        if inside.any():
            valid = vox[inside]
            touched[valid[:, 0], valid[:, 1], valid[:, 2]] = True
    touched &= target_mask
    total = int(target_mask.sum())
    return {
        "target_voxels": total,
        "covered_voxels": int(touched.sum()),
        "coverage_fraction": float(touched.sum() / total) if total else None,
    }


def _crosses_midline(streamline, midline_distance, inv_affine, tol_mm):
    """True if the streamline has points on both sides of the midline beyond
    ``tol_mm`` — a commissural/contralateral excursion."""
    if midline_distance is None:
        return False
    vox = _to_voxels(streamline, inv_affine)
    shp = midline_distance.shape
    inb = ((vox >= 0) & (vox < np.asarray(shp))).all(axis=1)
    if not inb.any():
        return False
    v = vox[inb]
    d = midline_distance[v[:, 0], v[:, 1], v[:, 2]]
    return bool((d.max() > tol_mm) and (d.min() < -tol_mm))


def _orient_cortex_to_inferior(streamline, affine):
    """Return the streamline ordered so it runs superior (cortex) -> inferior.

    World +Z is superior; the first endpoint should be the more superior one.
    """
    z0 = streamline[0] @ affine[:3, 2] if False else streamline[0][2]
    z1 = streamline[-1][2]
    if z1 > z0:
        return streamline[::-1]
    return streamline


def _branch_signature(streamline, seed):
    """Stable identity for matching the normal and diagnostic tracking passes."""
    points = np.asarray(streamline, dtype=np.float64)
    return (
        tuple(np.round(np.asarray(seed, dtype=np.float64), 6)),
        len(points),
        tuple(np.round(points[0], 6)),
        tuple(np.round(points[-1], 6)),
        tuple(np.round(points.sum(axis=0), 5)),
    )


def _target_tree(mask, affine):
    voxels = np.argwhere(np.asarray(mask, dtype=bool))
    if not len(voxels):
        raise ValueError("Diagnostic target mask is empty.")
    return cKDTree(
        nib_affines_apply(np.asarray(affine, dtype=np.float64), voxels)
    )


def nib_affines_apply(affine, points):
    """Apply an affine without importing nibabel in the tracking hot path."""
    points = np.asarray(points, dtype=np.float64)
    return points @ affine[:3, :3].T + affine[:3, 3]


def _motorward_terminal(streamline, motor_tree):
    """Return the endpoint nearest the motor ROI and its outward direction."""
    points = np.asarray(streamline, dtype=np.float64)
    endpoint_distances = motor_tree.query(points[[0, -1]])[0]
    if endpoint_distances[0] < endpoint_distances[1]:
        endpoint_index = 0
        direction = points[0] - points[1] if len(points) > 1 else None
    elif endpoint_distances[1] < endpoint_distances[0]:
        endpoint_index = -1
        direction = points[-1] - points[-2] if len(points) > 1 else None
    else:
        endpoint_index = 0 if points[0, 2] >= points[-1, 2] else -1
        if len(points) > 1:
            direction = (points[0] - points[1] if endpoint_index == 0
                         else points[-1] - points[-2])
        else:
            direction = None

    if direction is not None:
        norm = np.linalg.norm(direction)
        direction = direction / norm if norm else None
    opposite_index = -1 if endpoint_index == 0 else 0
    return (
        points[endpoint_index],
        direction,
        float(endpoint_distances[0 if endpoint_index == 0 else 1]),
        points[opposite_index],
    )


def _diagnose_branches(
    diagnostic_pairs,
    returned_pairs,
    seeds,
    hemisphere,
    motor_mask,
    inferior_mask,
    affine,
    *,
    min_length,
    max_length,
    midline_distance,
    midline_tol_mm,
    streamline_offset=0,
):
    """Classify every seed branch without changing the production tractogram.

    ``diagnostic_pairs`` comes from a second ``LocalTracking`` pass with
    ``return_all=True``. ``returned_pairs`` comes from the normal
    ``return_all=False`` pass. Matching the two makes propagation failure an
    observed state rather than an inference from accepted streamlines.
    """
    inv_affine = np.linalg.inv(np.asarray(affine))
    motor_tree = _target_tree(motor_mask, affine)
    seed_lookup = {
        tuple(np.round(np.asarray(seed, dtype=np.float64), 6)): index
        for index, seed in enumerate(seeds)
    }
    returned = Counter(
        _branch_signature(streamline, seed)
        for streamline, seed in returned_pairs
    )
    branch_counts = defaultdict(int)
    rows = []
    initiated_streamlines = []

    for streamline, seed in diagnostic_pairs:
        points = np.asarray(streamline, dtype=np.float64)
        seed_key = tuple(np.round(np.asarray(seed, dtype=np.float64), 6))
        seed_index = seed_lookup[seed_key]
        branch_index = branch_counts[seed_index]
        branch_counts[seed_index] += 1

        signature = _branch_signature(points, seed)
        tracker_returned = returned[signature] > 0
        if tracker_returned:
            returned[signature] -= 1

        tracking_initiated = len(points) > 1
        reaches_motor = (
            _passes_through(points, motor_mask, inv_affine)
            if tracking_initiated else False
        )
        reaches_inferior = (
            _passes_through(points, inferior_mask, inv_affine)
            if tracking_initiated else False
        )
        reaches_both = reaches_motor and reaches_inferior
        length_mm = float(streamline_length(points)) if tracking_initiated else 0.0
        crosses_midline = (
            _crosses_midline(
                points, midline_distance, inv_affine, midline_tol_mm
            )
            if tracking_initiated else False
        )

        if not tracking_initiated:
            rejection_reason = "no_initial_direction"
        elif not tracker_returned:
            rejection_reason = "local_propagation_failure"
        elif not reaches_motor and not reaches_inferior:
            rejection_reason = "missed_both_targets"
        elif not reaches_motor:
            rejection_reason = "missed_motor_target"
        elif not reaches_inferior:
            rejection_reason = "missed_inferior_target"
        elif length_mm < min_length:
            rejection_reason = "below_minimum_length"
        elif length_mm > max_length:
            rejection_reason = "above_maximum_length"
        elif crosses_midline:
            rejection_reason = "midline_crossing"
        else:
            rejection_reason = "accepted"

        final_coordinate, final_direction, final_distance, opposite_coordinate = (
            _motorward_terminal(points, motor_tree)
        )
        min_distance = float(motor_tree.query(points)[0].min())
        streamline_index = None
        if tracking_initiated:
            streamline_index = streamline_offset + len(initiated_streamlines)
            initiated_streamlines.append(points)

        rows.append({
            "hemisphere": hemisphere,
            "seed_index": int(seed_index),
            "branch_index": int(branch_index),
            "seed_coordinate_mm": [float(value) for value in seed],
            "tracking_initiated": bool(tracking_initiated),
            "tracker_returned": bool(tracker_returned),
            "reached_inferior_target": bool(reaches_inferior),
            "reached_motor_target": bool(reaches_motor),
            "reached_both_targets": bool(reaches_both),
            "rejection_reason": rejection_reason,
            "final_valid_coordinate_mm": [
                float(value) for value in final_coordinate
            ],
            "final_direction": (
                [float(value) for value in final_direction]
                if final_direction is not None else None
            ),
            "opposite_terminal_coordinate_mm": [
                float(value) for value in opposite_coordinate
            ],
            "length_mm": length_mm,
            "final_distance_to_motor_target_mm": final_distance,
            "minimum_distance_to_motor_target_mm": min_distance,
            "crosses_midline": bool(crosses_midline),
            "streamline_index": streamline_index,
        })

    missing = sum(returned.values())
    if missing:
        raise RuntimeError(
            f"Diagnostic tracking did not reproduce {missing} production branches."
        )
    return rows, initiated_streamlines


# ---------------------------------------------------------------------------
# The preferred extraction
# ---------------------------------------------------------------------------

def extract_cst_plic(
    data, gtab, affine, brain_mask,
    plic_left_mask, plic_right_mask,
    motor_left_mask, motor_right_mask,
    inferior_left_mask, inferior_right_mask,
    *,
    model_mask=None,
    stopping_criterion=None,
    stopping_meta=None,
    midline_distance=None,
    midline_tol_mm=8.0,
    seed_density=4,
    step_size=0.5,
    min_length=40.0,
    max_length=300.0,
    sh_order=6,
    npeaks=1,
    relative_peak_threshold=0.5,
    min_separation_angle=25,
    random_seed=None,
    fit_method='WLS',
    fa_map=None,
    collect_diagnostics=False,
    coordinate_frame_qc=None,
    precomputed_peaks=None,
    precomputed_sh_order=None,
    direction_model_meta=None,
    verbose=True,
):
    """Extract the bilateral CST by PLIC seeding + dual-target acceptance.

    ``model_mask`` defaults to ``brain_mask`` (permissive). ``stopping_criterion``
    defaults to ``BinaryStoppingCriterion(brain_mask)`` (DWI-only). Pass a CMC
    criterion (see :func:`build_stopping_model`) for anatomically-constrained
    stopping. Controlled experiments may pass ``precomputed_peaks`` to reuse one
    fitted direction field across seed variants; ``precomputed_sh_order`` and
    ``direction_model_meta`` then describe that field in the result metadata.
    """
    if model_mask is None:
        model_mask = np.asarray(brain_mask, dtype=bool)
    if stopping_criterion is None:
        stopping_criterion, stopping_meta = build_stopping_model(
            brain_mask, affine, step_size=step_size, verbose=verbose)
    if stopping_meta is None:
        stopping_meta = {"mode": "unknown"}

    inv_affine = np.linalg.inv(np.asarray(affine))

    if verbose:
        print("=" * 60)
        print("PREFERRED CST EXTRACTION (PLIC seed, dual-target)")
        print("=" * 60)

    if precomputed_peaks is None:
        peaks, validated_sh, fitted_direction_meta = estimate_model_peaks(
            data, gtab, model_mask, sh_order=sh_order, npeaks=npeaks,
            relative_peak_threshold=relative_peak_threshold,
            min_separation_angle=min_separation_angle, verbose=verbose)
    else:
        peaks = precomputed_peaks
        validated_sh = (
            int(precomputed_sh_order)
            if precomputed_sh_order is not None else int(sh_order)
        )
        fitted_direction_meta = (
            dict(direction_model_meta)
            if direction_model_meta is not None
            else {
                "model": "precomputed",
                "validated_sh_order": validated_sh,
            }
        )
        if verbose:
            print(
                f"  → Reusing precomputed direction field "
                f"(sh_order={validated_sh}, npeaks={npeaks})"
            )

    def _seed_and_track(
        seed_mask, hemisphere, motor_mask, inferior_mask, streamline_offset=0
    ):
        seeds = tracking_utils.seeds_from_mask(
            np.asarray(seed_mask, dtype=bool), affine, density=seed_density)
        if not collect_diagnostics:
            gen = LocalTracking(
                peaks, stopping_criterion, seeds, affine,
                step_size=step_size, return_all=False, random_seed=random_seed)
            return Streamlines(gen), len(seeds), None

        # Keep production identical to the non-diagnostic path. LocalTracking
        # mutates its direction getter's RNG state; diagnostics therefore use
        # independent copies so they cannot change the next hemisphere.
        returned_diag_peaks = copy.deepcopy(peaks)
        all_diag_peaks = copy.deepcopy(peaks)
        production_gen = LocalTracking(
            peaks, stopping_criterion, seeds, affine,
            step_size=step_size, return_all=False, random_seed=random_seed,
        )
        returned_streamlines = Streamlines(production_gen)
        normal_diag_gen = LocalTracking(
            returned_diag_peaks, stopping_criterion, seeds, affine,
            step_size=step_size, return_all=False, random_seed=random_seed,
            save_seeds=True,
        )
        returned_pairs = list(normal_diag_gen)
        diagnostic_gen = LocalTracking(
            all_diag_peaks, stopping_criterion, seeds, affine,
            step_size=step_size, return_all=True, random_seed=random_seed,
            save_seeds=True,
        )
        rows, initiated_streamlines = _diagnose_branches(
            diagnostic_gen,
            returned_pairs,
            seeds,
            hemisphere,
            motor_mask,
            inferior_mask,
            affine,
            min_length=min_length,
            max_length=max_length,
            midline_distance=midline_distance,
            midline_tol_mm=midline_tol_mm,
            streamline_offset=streamline_offset,
        )
        return returned_streamlines, len(seeds), {
            "rows": rows,
            "streamlines": initiated_streamlines,
        }

    def _accept(streamlines, motor_mask, inferior_mask, contralateral_motor_mask):
        """Keep dual-target branches and create target-bounded analysis segments."""
        accepted_full = []
        accepted_segments = []
        segment_rows = []
        n_motor = n_inferior = n_both = n_len = n_midline = 0
        n_contralateral = n_order = 0
        for sl in streamlines:
            L = streamline_length(sl)
            reaches_motor = _passes_through(sl, motor_mask, inv_affine)
            reaches_inf = _passes_through(sl, inferior_mask, inv_affine)
            reaches_contralateral = _passes_through(
                sl, contralateral_motor_mask, inv_affine
            )
            if reaches_motor:
                n_motor += 1
            if reaches_inf:
                n_inferior += 1
            if not (reaches_motor and reaches_inf):
                continue
            n_both += 1
            if L < min_length or L > max_length:
                n_len += 1
                continue
            if _crosses_midline(sl, midline_distance, inv_affine, midline_tol_mm):
                n_midline += 1
                continue
            full = np.asarray(sl)
            if full[0, 2] > full[-1, 2]:
                full = full[::-1]
            segment, segment_meta = _target_bounded_segment(
                full, inferior_mask, motor_mask, inv_affine
            )
            if segment is None:
                n_order += 1
                continue
            if reaches_contralateral:
                n_contralateral += 1
            accepted_full.append(full)
            accepted_segments.append(segment)
            segment_rows.append(segment_meta)
        diag = {
            "generated": len(streamlines),
            "reach_motor": n_motor,
            "reach_inferior": n_inferior,
            "reach_both": n_both,
            "rejected_length": n_len,
            "rejected_midline": n_midline,
            "rejected_target_order": n_order,
            "contralateral_target_contact": n_contralateral,
            "accepted": len(accepted_segments),
        }
        return (
            Streamlines(accepted_segments),
            Streamlines(accepted_full),
            diag,
            segment_rows,
        )

    if verbose:
        print("\n[1/2] Left PLIC -> track -> accept (motor L + inferior L)...")
    left_raw, n_seed_left, left_branch_diag = _seed_and_track(
        plic_left_mask, "left", motor_left_mask, inferior_left_mask
    )
    cst_left, cst_left_full, diag_left, segment_left = _accept(
        left_raw, motor_left_mask, inferior_left_mask, motor_right_mask
    )

    if verbose:
        print("\n[2/2] Right PLIC -> track -> accept (motor R + inferior R)...")
    right_offset = (
        len(left_branch_diag["streamlines"]) if left_branch_diag is not None else 0
    )
    right_raw, n_seed_right, right_branch_diag = _seed_and_track(
        plic_right_mask,
        "right",
        motor_right_mask,
        inferior_right_mask,
        streamline_offset=right_offset,
    )
    cst_right, cst_right_full, diag_right, segment_right = _accept(
        right_raw, motor_right_mask, inferior_right_mask, motor_left_mask
    )

    cst_combined = Streamlines(list(cst_left) + list(cst_right))
    cst_combined_full = Streamlines(
        list(cst_left_full) + list(cst_right_full)
    )

    n_left, n_right = len(cst_left), len(cst_right)
    total = n_left + n_right
    li = (n_left - n_right) / total if total else 0.0

    stats = {
        'method': 'preferred-plic',
        'stopping_mode': stopping_meta.get('mode'),
        'stopping_label': stopping_meta.get('label'),
        'left_seeds': int(n_seed_left),
        'right_seeds': int(n_seed_right),
        'cst_left_count': n_left,
        'cst_right_count': n_right,
        'cst_total_count': total,
        'left_diag': diag_left,
        'right_diag': diag_right,
        'left_motor_reach_frac': diag_left['reach_motor'] / max(diag_left['generated'], 1),
        'right_motor_reach_frac': diag_right['reach_motor'] / max(diag_right['generated'], 1),
        'left_inferior_reach_frac': diag_left['reach_inferior'] / max(diag_left['generated'], 1),
        'right_inferior_reach_frac': diag_right['reach_inferior'] / max(diag_right['generated'], 1),
        'laterality_index': li,
        'total_input': diag_left['generated'] + diag_right['generated'],
        'passes_left_motor': diag_left['reach_motor'],
        'passes_right_motor': diag_right['reach_motor'],
        'passes_brainstem': diag_left['reach_inferior'] + diag_right['reach_inferior'],
        'artifact_index': None,
        'analysis_segment': {
            'definition': (
                'last inferior-target sample before first ipsilateral '
                'motor-target sample, inclusive'
            ),
            'orientation': 'inferior-to-superior',
            'full_tracking_artifact_preserved': True,
            'left': {
                'streamlines': len(segment_left),
                'points_removed_below_inferior': int(sum(
                    row['points_removed_below_inferior'] for row in segment_left
                )),
                'points_removed_above_motor': int(sum(
                    row['points_removed_above_motor'] for row in segment_left
                )),
            },
            'right': {
                'streamlines': len(segment_right),
                'points_removed_below_inferior': int(sum(
                    row['points_removed_below_inferior'] for row in segment_right
                )),
                'points_removed_above_motor': int(sum(
                    row['points_removed_above_motor'] for row in segment_right
                )),
            },
        },
    }
    branch_rows = []
    if collect_diagnostics:
        branch_rows = left_branch_diag["rows"] + right_branch_diag["rows"]
        reason_counts = Counter(row["rejection_reason"] for row in branch_rows)
        stats["branch_diagnostics"] = {
            "rows": len(branch_rows),
            "initiated": sum(row["tracking_initiated"] for row in branch_rows),
            "propagation_failures": sum(
                row["rejection_reason"]
                in {"no_initial_direction", "local_propagation_failure"}
                for row in branch_rows
            ),
            "post_tracking_rejections": sum(
                row["tracking_initiated"]
                and row["rejection_reason"]
                not in {"accepted", "local_propagation_failure"}
                for row in branch_rows
            ),
            "rejection_reason_counts": dict(sorted(reason_counts.items())),
        }

    def _hemisphere_qc(side, seed_count, diag, segments, motor, inferior):
        rows = [row for row in branch_rows if row["hemisphere"] == side]
        initiated = (
            sum(row["tracking_initiated"] for row in rows)
            if rows else diag["generated"]
        )
        reasons = Counter(row["rejection_reason"] for row in rows)
        motor_count = (
            sum(row["tracking_initiated"] and row["reached_motor_target"]
                for row in rows)
            if rows else diag["reach_motor"]
        )
        inferior_count = (
            sum(row["tracking_initiated"] and row["reached_inferior_target"]
                for row in rows)
            if rows else diag["reach_inferior"]
        )
        dual_count = (
            sum(row["tracking_initiated"] and row["reached_both_targets"]
                for row in rows)
            if rows else diag["reach_both"]
        )
        denominator = max(initiated, 1)
        other = {
            key: int(value)
            for key, value in sorted(reasons.items())
            if key not in {
                "accepted", "below_minimum_length", "above_maximum_length",
                "midline_crossing",
            }
        }
        return {
            "plic_seed_count": int(seed_count),
            "initiated_branch_count": int(initiated),
            "motor_target_reach_count": int(motor_count),
            "motor_target_reach_fraction": float(motor_count / denominator),
            "inferior_target_reach_count": int(inferior_count),
            "inferior_target_reach_fraction": float(inferior_count / denominator),
            "dual_target_count": int(dual_count),
            "dual_target_fraction": float(dual_count / denominator),
            "length_rejection_count": int(diag["rejected_length"]),
            "midline_rejection_count": int(diag["rejected_midline"]),
            "contralateral_target_contact_count": int(
                diag["contralateral_target_contact"]
            ),
            "other_exclusion_counts": other | {
                "target_intersection_order": int(
                    diag["rejected_target_order"]
                )
            },
            "persisted_streamline_count": int(diag["accepted"]),
            "motor_target_coverage": _target_coverage(
                segments, motor, inv_affine
            ),
            "inferior_target_coverage": _target_coverage(
                segments, inferior, inv_affine
            ),
            "deterministic_run_identity": None,
        }

    stats["tractography_qc"] = {
        "left": _hemisphere_qc(
            "left", n_seed_left, diag_left, cst_left,
            motor_left_mask, inferior_left_mask,
        ),
        "right": _hemisphere_qc(
            "right", n_seed_right, diag_right, cst_right,
            motor_right_mask, inferior_right_mask,
        ),
    }
    left_qc = stats["tractography_qc"]["left"]
    right_qc = stats["tractography_qc"]["right"]
    stats.update({
        "left_motor_reach_frac": left_qc["motor_target_reach_fraction"],
        "right_motor_reach_frac": right_qc["motor_target_reach_fraction"],
        "left_inferior_reach_frac": left_qc["inferior_target_reach_fraction"],
        "right_inferior_reach_frac": right_qc["inferior_target_reach_fraction"],
        "passes_left_motor": left_qc["motor_target_reach_count"],
        "passes_right_motor": right_qc["motor_target_reach_count"],
        "passes_brainstem": (
            left_qc["inferior_target_reach_count"]
            + right_qc["inferior_target_reach_count"]
        ),
        "total_input": (
            left_qc["initiated_branch_count"]
            + right_qc["initiated_branch_count"]
        ),
    })
    if total > 0:
        lengths = np.array([streamline_length(s) for s in cst_combined])
        stats.update(
            length_mean=float(lengths.mean()),
            length_std=float(lengths.std()),
            length_min=float(lengths.min()),
            length_max=float(lengths.max()),
        )

    stats["coordinate_frame_qc"] = dict(coordinate_frame_qc or {})
    stats["qc_thresholds"] = preferred_qc_thresholds(
        min_length=min_length,
        max_length=max_length,
        midline_tol_mm=midline_tol_mm,
    )
    stats["qc"] = cst_qc_gates(stats)

    if verbose:
        print("\n" + "=" * 60)
        print("PREFERRED EXTRACTION COMPLETE")
        print("=" * 60)
        print(f"  Left CST:  {n_left:,}  (motor reach {stats['left_motor_reach_frac']:.2f}, "
              f"inferior reach {stats['left_inferior_reach_frac']:.2f})")
        print(f"  Right CST: {n_right:,}  (motor reach {stats['right_motor_reach_frac']:.2f}, "
              f"inferior reach {stats['right_inferior_reach_frac']:.2f})")
        print(f"  Total: {total:,}  |LI|={li:.3f}")

    parameters = {
        'seed_density': seed_density, 'step_size': step_size,
        'min_length': min_length, 'max_length': max_length,
        'sh_order': validated_sh, 'npeaks': npeaks,
        'relative_peak_threshold': relative_peak_threshold,
        'min_separation_angle': min_separation_angle,
        'random_seed': random_seed, 'fit_method': fit_method,
        'midline_tol_mm': midline_tol_mm,
        'model_mask_source': 'brain_mask' if model_mask is brain_mask else 'explicit',
        'stopping': stopping_meta,
        'direction_model': fitted_direction_meta,
    }

    result = {
        'cst_left': cst_left,
        'cst_right': cst_right,
        'cst_combined': cst_combined,
        'cst_left_full': cst_left_full,
        'cst_right_full': cst_right_full,
        'cst_combined_full': cst_combined_full,
        'stats': stats,
        'parameters': parameters,
    }
    if collect_diagnostics:
        result["branch_diagnostics"] = branch_rows
        result["diagnostic_streamlines"] = Streamlines(
            left_branch_diag["streamlines"] + right_branch_diag["streamlines"]
        )
    return result


def save_plic_diagnostics(
    cst_result, reference_img, output_dir, subject_id=None, verbose=True
):
    """Persist branch-level preferred-tracking diagnostics and raw branches."""
    rows = cst_result.get("branch_diagnostics")
    streamlines = cst_result.get("diagnostic_streamlines")
    if rows is None or streamlines is None:
        return {}

    from dipy.io.stateful_tractogram import Space, StatefulTractogram
    from dipy.io.streamline import save_tractogram

    output_dir = Path(output_dir)
    diagnostic_dir = output_dir / "diagnostics"
    diagnostic_dir.mkdir(parents=True, exist_ok=True)
    prefix = f"{subject_id}_" if subject_id else ""

    table_path = diagnostic_dir / f"{prefix}plic_branch_diagnostics.jsonl.gz"
    with gzip.open(table_path, "wt", encoding="utf-8") as stream:
        for row in rows:
            stream.write(json.dumps(row, separators=(",", ":")))
            stream.write("\n")

    tractogram_path = diagnostic_dir / f"{prefix}plic_all_branches.trk"
    sft = StatefulTractogram(streamlines, reference_img, Space.RASMM)
    save_tractogram(sft, str(tractogram_path))

    metadata_path = diagnostic_dir / f"{prefix}plic_branch_diagnostics.json"
    metadata = {
        "schema_version": 1,
        "coordinate_space": "RASMM",
        "row_semantics": (
            "One row per seed/initial-direction branch; seeds without an initial "
            "direction have a single non-initiated row."
        ),
        "final_coordinate_semantics": (
            "Endpoint nearest the ipsilateral motor-target voxel; ties use the "
            "more superior endpoint."
        ),
        "streamline_index_semantics": (
            "Index into plic_all_branches.trk; null for non-initiated seeds."
        ),
        "counts": cst_result["stats"].get("branch_diagnostics", {}),
        "parameters": cst_result.get("parameters", {}),
        "files": {
            "records": str(table_path),
            "all_initiated_branches": str(tractogram_path),
        },
    }
    metadata_path.write_text(json.dumps(metadata, indent=2) + "\n")
    paths = {
        "branch_records": table_path,
        "all_branches": tractogram_path,
        "metadata": metadata_path,
    }
    if verbose:
        print(f"  ✓ PLIC branch diagnostics: {table_path}")
        print(f"  ✓ All initiated PLIC branches: {tractogram_path}")
    return paths
