"""Tests for the preferred, role-separated CST pipeline.

Covers the behavioural contracts that resolve the 2026-09-22 collapse:
role independence (seed vs model vs stopping), the stopping-model resolution
(DWI-only Binary vs CMC-from-PVE), PVE/mask validation, dual-target acceptance
geometry, orientation, midline exclusion, and the per-subject QC gates that must
flag a starved run as a failure.
"""
import numpy as np
import nibabel as nib
import pytest

from csttool.extract.modules import plic_cst as P
from csttool.extract.modules.plic_seeding import (
    _centroid_world,
    build_inferior_targets,
    prepare_tracking_grid_masks,
)
from dipy.tracking.stopping_criterion import (
    BinaryStoppingCriterion, CmcStoppingCriterion,
)

AFF = np.diag([2.0, 2.0, 2.0, 1.0])


# --------------------------------------------------------------------------
# QC gates — the load-bearing collapse detector
# --------------------------------------------------------------------------
class TestQcGates:
    def _stats(self, nl, nr, gen, lmf, rmf, lif, rif):
        total = nl + nr
        return {
            'cst_left_count': nl, 'cst_right_count': nr, 'total_input': gen,
            'left_motor_reach_frac': lmf, 'right_motor_reach_frac': rmf,
            'left_inferior_reach_frac': lif, 'right_inferior_reach_frac': rif,
            'laterality_index': (nl - nr) / total if total else 0.0,
            'left_diag': {}, 'right_diag': {},
            'coordinate_frame_qc': {'overall_pass': True},
        }

    def test_collapse_case_is_a_failure(self):
        gates = P.cst_qc_gates(
            self._stats(1, 17, 33000, 0.001, 0.9, 0.2, 0.9)
        )
        assert gates['gates']['not_collapsed']['pass'] is False
        assert gates['gates']['persisted_streamline_floor']['pass'] is False
        assert gates['overall_pass'] is False

    def test_healthy_bilateral_case_passes(self):
        gates = P.cst_qc_gates(
            self._stats(1030, 473, 3180, 0.62, 0.44, 0.96, 0.93)
        )
        assert all(gate['pass'] for gate in gates['gates'].values())
        assert gates['overall_pass'] is True

    @pytest.mark.parametrize("left,right", [(900, 20), (20, 900)])
    def test_gross_asymmetry_fails_with_signed_measurement(self, left, right):
        gates = P.cst_qc_gates(
            self._stats(left, right, 2000, 0.6, 0.6, 0.9, 0.9)
        )
        measured = gates['gates']['count_laterality']['measured_signed_li']
        assert np.sign(measured) == np.sign(left - right)
        assert gates['gates']['count_laterality']['pass'] is False
        assert gates['overall_pass'] is False


# --------------------------------------------------------------------------
# Stopping model — an independent role, DWI-only vs CMC
# --------------------------------------------------------------------------
class TestStoppingModel:
    def test_dwi_only_is_binary_on_brain_mask(self):
        bm = np.zeros((8, 8, 8), bool); bm[2:6, 2:6, 2:6] = True
        crit, meta = P.build_stopping_model(bm, AFF, verbose=False)
        assert isinstance(crit, BinaryStoppingCriterion)
        assert meta['mode'] == 'binary-brain-mask-boundary'
        assert meta['criterion_class'] == 'BinaryStoppingCriterion'
        assert meta['propagation_boundary'] == 'brain_mask'
        assert meta['diffusion_scalar_threshold'] is None

    def test_cmc_from_continuous_pve(self):
        bm = np.zeros((8, 8, 8), bool); bm[2:6, 2:6, 2:6] = True
        wm = np.zeros((8, 8, 8)); wm[2:6, 2:6, 2:6] = 0.8
        gm = np.zeros((8, 8, 8)); gm[2:6, 2:6, 2:6] = 0.2
        csf = np.zeros((8, 8, 8))
        crit, meta = P.build_stopping_model(bm, AFF, pve_wm=wm, pve_gm=gm,
                                            pve_csf=csf, verbose=False)
        assert isinstance(crit, CmcStoppingCriterion)
        assert meta['mode'] == 'cmc'
        assert meta['gm_valid_termination'] and meta['csf_invalid_termination']


# --------------------------------------------------------------------------
# Selection geometry — dual-target acceptance, orientation, midline exclusion
# --------------------------------------------------------------------------
class TestSelectionGeometry:
    def test_passes_through_requires_voxel_hit(self):
        mask = np.zeros((10, 10, 10), bool); mask[5, 5, 5] = True
        inv = np.linalg.inv(AFF)
        # World [10,10,10] -> voxel [5,5,5] with a 2 mm isotropic affine.
        through = np.array([[10.0, 10.0, 8.0], [10.0, 10.0, 10.0], [10.0, 10.0, 12.0]])
        miss = np.array([[0.0, 0.0, 0.0], [2.0, 2.0, 2.0]])
        assert P._passes_through(miss, mask, inv) is False

    def test_orientation_is_cortex_to_inferior(self):
        sl = np.array([[0, 0, -10.0], [0, 0, 10.0]])  # inferior-first
        out = P._orient_cortex_to_inferior(sl, AFF)
        assert out[0][2] > out[-1][2]  # first point is superior (cortex)

    def test_midline_crossing_detected_beyond_tolerance(self):
        # Signed distance: negative left, positive right.
        md = np.zeros((10, 10, 10))
        md[:5] = -20.0; md[5:] = 20.0
        inv = np.linalg.inv(AFF)
        # a streamline spanning both far sides (voxels 2 and 8 in x)
        crossing = np.array([[4.0, 10.0, 10.0], [16.0, 10.0, 10.0]])
        contained = np.array([[4.0, 10.0, 10.0], [6.0, 10.0, 10.0]])
        assert P._crosses_midline(crossing, md, inv, tol_mm=8.0) is True
        assert P._crosses_midline(contained, md, inv, tol_mm=8.0) is False

    def test_target_bounded_segment_is_inferior_to_motor_and_drops_tails(self):
        inferior = np.zeros((3, 3, 8), bool)
        motor = np.zeros_like(inferior)
        inferior[1, 1, 2] = True
        inferior[1, 1, 3] = True
        motor[1, 1, 6] = True
        streamline = np.array([
            [2.0, 2.0, 0.0],
            [2.0, 2.0, 4.0],
            [2.0, 2.0, 6.0],
            [2.0, 2.0, 8.0],
            [2.0, 2.0, 12.0],
            [2.0, 2.0, 14.0],
        ])
        segment, meta = P._target_bounded_segment(
            streamline, inferior, motor, np.linalg.inv(AFF)
        )
        assert np.array_equal(segment, streamline[2:5])
        assert meta['pass'] is True
        assert meta['inferior_intersection_index'] == 2
        assert meta['motor_intersection_index'] == 4
        assert meta['points_removed_below_inferior'] == 2
        assert meta['points_removed_above_motor'] == 1

    def test_target_bounded_segment_rejects_reverse_target_order(self):
        inferior = np.zeros((3, 3, 8), bool)
        motor = np.zeros_like(inferior)
        inferior[1, 1, 6] = True
        motor[1, 1, 2] = True
        streamline = np.array([
            [2.0, 2.0, 0.0],
            [2.0, 2.0, 4.0],
            [2.0, 2.0, 12.0],
        ])
        segment, meta = P._target_bounded_segment(
            streamline, inferior, motor, np.linalg.inv(AFF)
        )
        assert segment is None
        assert meta['reason'] == 'target_intersection_order'


# --------------------------------------------------------------------------
# Inferior target construction — hemisphere split, independent of the seed
# --------------------------------------------------------------------------
class TestInferiorTargets:
    def test_brainstem_split_by_hemisphere(self):
        bs = np.zeros((10, 10, 6), bool); bs[3:7, 3:7, 1:3] = True
        hemi = np.zeros((10, 10, 6), bool); hemi[:5] = True  # left = low x
        reg = {'hemisphere_mask': hemi}
        left, right = build_inferior_targets(bs, reg, verbose=False)
        assert left.sum() > 0 and right.sum() > 0
        assert int((left & right).sum()) == 0
        assert np.array_equal(left | right, bs)



def test_registration_masks_are_inverse_reoriented_before_tracking():
    shape = (6, 4, 3)
    las_affine = np.array([
        [-2.0, 0.0, 0.0, 5.0],
        [0.0, 2.0, 0.0, -3.0],
        [0.0, 0.0, 2.0, -2.0],
        [0.0, 0.0, 0.0, 1.0],
    ])
    current = nib.io_orientation(las_affine)
    target = nib.orientations.axcodes2ornt(("R", "A", "S"))
    original_to_ras = nib.orientations.ornt_transform(current, target)

    plic_left = np.zeros(shape, bool)
    plic_right = np.zeros(shape, bool)
    motor_left = np.zeros(shape, bool)
    motor_right = np.zeros(shape, bool)
    brainstem = np.zeros(shape, bool)
    plic_left[4, 2, 1] = True
    plic_right[1, 2, 1] = True
    motor_left[5, 2, 2] = True
    motor_right[0, 2, 2] = True
    brainstem[1, 2, 0] = True
    brainstem[4, 2, 0] = True
    hemisphere = np.zeros(shape, bool)
    hemisphere[3:, :, :] = True
    world_x = (
        np.indices(shape)[0] * las_affine[0, 0] + las_affine[0, 3]
    )

    to_ras = lambda array: nib.orientations.apply_orientation(  # noqa: E731
        array, original_to_ras
    )
    reg = {
        "reorientation_transform": original_to_ras,
        "original_subject_shape": shape,
        "original_orientation": ("L", "A", "S"),
        "hemisphere_mask": to_ras(hemisphere),
        "midline_distance": to_ras(world_x),
        "midline_x": 0.0,
    }
    plic = {
        "plic_left": to_ras(plic_left),
        "plic_right": to_ras(plic_right),
        "provenance": {},
    }
    rois = {
        "motor_left": to_ras(motor_left),
        "motor_right": to_ras(motor_right),
        "brainstem": to_ras(brainstem),
    }

    # The pre-fix contract silently paired this RAS array with the LAS affine,
    # putting the intended left seed in positive world X.
    assert _centroid_world(plic["plic_left"], las_affine)[0] > 0

    prepared = prepare_tracking_grid_masks(
        reg,
        rois,
        plic,
        np.ones(shape, bool),
        las_affine,
    )

    assert np.array_equal(prepared["plic_left"], plic_left)
    assert np.array_equal(prepared["plic_right"], plic_right)
    assert np.array_equal(prepared["motor_left"], motor_left)
    assert np.array_equal(prepared["motor_right"], motor_right)
    assert _centroid_world(prepared["plic_left"], las_affine)[0] < 0
    assert _centroid_world(prepared["plic_right"], las_affine)[0] > 0
    assert prepared["inferior_left"][4, 2, 0]
    assert prepared["inferior_right"][1, 2, 0]
    assert prepared["plic_provenance"]["tracking_grid"] == {
        "shape": [6, 4, 3],
        "orientation": ["L", "A", "S"],
        "inverse_reorientation_applied": True,
        "original_to_ras_orientation_transform": original_to_ras.tolist(),
    }
    frame_qc = prepared["coordinate_frame_qc"]
    assert frame_qc["overall_pass"] is True
    assert frame_qc["original_dwi_orientation"] == ["L", "A", "S"]
    assert frame_qc["internal_registration_orientation"] == "RAS"
    assert all(
        item["applied"]
        for item in frame_qc["inverse_reorientation"].values()
    )

# --------------------------------------------------------------------------
# Branch diagnostics — propagation failure versus post-tracking rejection
# --------------------------------------------------------------------------
class TestBranchDiagnostics:
    def _masks(self):
        motor = np.zeros((10, 10, 10), bool)
        inferior = np.zeros_like(motor)
        motor[5, 5, 8] = True
        inferior[5, 5, 2] = True
        return motor, inferior

    def test_records_every_seed_branch_and_classifies_rejection_stage(self):
        motor, inferior = self._masks()
        seed_a = np.array([10.0, 10.0, 10.0])
        seed_b = np.array([12.0, 10.0, 10.0])
        accepted = np.array([
            [10.0, 10.0, 4.0],
            [10.0, 10.0, 10.0],
            [10.0, 10.0, 16.0],
        ])
        missed_motor = np.array([
            [10.0, 10.0, 4.0],
            [10.0, 10.0, 8.0],
            [10.0, 10.0, 12.0],
        ])
        diagnostic_pairs = [
            (accepted, seed_a),
            (missed_motor, seed_a),
            (seed_b[None, :], seed_b),
        ]
        returned_pairs = diagnostic_pairs[:2]

        rows, streamlines = P._diagnose_branches(
            diagnostic_pairs,
            returned_pairs,
            np.array([seed_a, seed_b]),
            "left",
            motor,
            inferior,
            AFF,
            min_length=0.0,
            max_length=100.0,
            midline_distance=None,
            midline_tol_mm=8.0,
        )

        assert [row["rejection_reason"] for row in rows] == [
            "accepted", "missed_motor_target", "no_initial_direction",
        ]
        assert [row["branch_index"] for row in rows] == [0, 1, 0]
        assert [row["tracking_initiated"] for row in rows] == [True, True, False]
        assert [row["streamline_index"] for row in rows] == [0, 1, None]
        assert len(streamlines) == 2
        assert rows[0]["reached_both_targets"] is True
        assert rows[0]["final_valid_coordinate_mm"] == [10.0, 10.0, 16.0]
        assert rows[0]["final_direction"] == [0.0, 0.0, 1.0]
        assert rows[0]["final_distance_to_motor_target_mm"] == 0.0
        assert rows[1]["minimum_distance_to_motor_target_mm"] == 4.0

    def test_branch_dropped_by_local_tracking_is_propagation_failure(self):
        motor, inferior = self._masks()
        seed = np.array([10.0, 10.0, 10.0])
        branch = np.array([
            [10.0, 10.0, 8.0],
            [10.0, 10.0, 10.0],
            [10.0, 10.0, 12.0],
        ])
        rows, _ = P._diagnose_branches(
            [(branch, seed)],
            [],
            np.array([seed]),
            "right",
            motor,
            inferior,
            AFF,
            min_length=0.0,
            max_length=100.0,
            midline_distance=None,
            midline_tol_mm=8.0,
        )
        assert rows[0]["tracking_initiated"] is True
        assert rows[0]["tracker_returned"] is False
        assert rows[0]["rejection_reason"] == "local_propagation_failure"


def test_centroid_world_matches_affine():
    m = np.zeros((10, 10, 10), bool); m[5, 5, 5] = True
    c = _centroid_world(m, AFF)
    assert np.allclose(c, [10.0, 10.0, 10.0])
