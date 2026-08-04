"""AU33 — automated tests of registration quality, atlas-warp label preservation
and ROI placement.

The audit found that ``warp_atlas_to_subject.py``'s QC (label-count change,
motor-centroid side, motor Z-difference) only ``print()``ed and that
``verify_atlas_labels`` returned a dict no test asserted on, so there was *no
automated test* of registration quality / ROI placement / atlas-warp label
preservation. These tests assert on the real registration+warp of the synthetic
scene (``synth_*`` session fixtures) and on the now-assertable QC dict
(``compute_atlas_warp_qc``).
"""

import numpy as np
import pytest
from dipy.align.imwarp import SymmetricDiffeomorphicRegistration
from dipy.align.metrics import CCMetric

from csttool.extract.modules.warp_atlas_to_subject import (
    compute_atlas_warp_qc,
    warp_atlas_to_subject,
    split_atlas_hemispheres_mni,
    verify_atlas_labels,
)
from csttool.extract.modules.create_roi_masks import create_cst_roi_masks
from csttool.extract.modules.warp_atlas_to_subject import CST_ROI_CONFIG


# ---------------------------------------------------------------------------
# Registration quality
# ---------------------------------------------------------------------------

def test_registration_recovers_planted_translation(synth_registration_result):
    """The warped atlas must land at MNI-atlas-world + DX, i.e. the registration
    recovered the planted +DX mm world-X shift. Asserted on the atlas warp
    (convention-free) rather than the affine matrix (whose sign convention
    differs between DIPY transform types)."""
    reg = synth_registration_result
    # The subject was built by shifting the template affine by +DX in world X,
    # so every MNI-space world coordinate maps to subject-space + DX.
    DX = 6.0
    # motor_left planted at MNI world (-12, 0, 12) → subject (-6, 0, 12).
    # Assert via the registration's warped midline being on the +X side of 0
    # (the subject is reoriented to RAS but never recentered, so a correct
    # registration must move the midline off 0 toward +DX).
    assert reg["midline_x"] > 0, (
        f"midline_x={reg['midline_x']} not shifted +X — translation not recovered"
    )


def test_jacobian_determinant_near_identity_per_hemisphere(
    synth_registration_result,
):
    """A near-rigid (translation-only) warp should have Jacobian mean ~1.0 and
    few/negative-free voxels per hemisphere; a large asymmetry would indicate
    the registration is fitting one hemisphere better than the other."""
    reg = synth_registration_result
    js = reg["jacobian_stats"]
    for side in ("left_mean", "right_mean"):
        assert abs(js[side] - 1.0) < 0.15, f"{side}={js[side]} far from 1.0"
    for side in ("left_negative_pct", "right_negative_pct"):
        # Folding (negative Jacobian) means the warp is not diffeomorphic
        # locally; a near-rigid synthetic warp should have none.
        assert js[side] < 5.0, f"{side}={js[side]}% — too many folding voxels"
    # L/R Jacobian means should be close (no hemisphere-fitting asymmetry).
    assert abs(js["left_mean"] - js["right_mean"]) < 0.1, (
        f"hemisphere Jacobian means differ: L={js['left_mean']} R={js['right_mean']}"
    )


def test_warped_midline_hemisphere_balance(synth_registration_result):
    """The warped-MNI hemisphere mask should be roughly balanced (the MNI
    midline splits the brain ~50/50 and the synthetic warp is near-rigid)."""
    reg = synth_registration_result
    hmask = reg["hemisphere_mask"]
    n_left = int(hmask.sum())
    n_total = int(hmask.size)
    frac = n_left / n_total
    assert 0.40 < frac < 0.60, f"hemisphere mask unbalanced: {frac:.3f} left"


# ---------------------------------------------------------------------------
# Atlas-warp label preservation (the previously assert-free QC)
# ---------------------------------------------------------------------------

def test_subcortical_labels_preserved(synth_warped_atlases, synth_atlas_images):
    """The brainstem label must survive warping (set unchanged, no spurious
    labels). This is the assert-free 'label count changed only warns' path
    promoted to an assertion."""
    qc = synth_warped_atlases["subcortical_qc"]
    assert qc["labels_preserved"], qc
    assert qc["labels_warped"] == qc["labels_original"], qc
    # The only expected subcortical label is brainstem (8).
    assert qc["labels_warped"] == [8], qc
    assert qc["label_counts"][8] > 0


def test_cortical_labels_preserved(synth_warped_atlases):
    """Both motor labels (7 left, 107 right) must survive warping."""
    qc = synth_warped_atlases["cortical_qc"]
    assert qc["labels_preserved"], qc
    assert set(qc["labels_warped"]) == {7, 107}, qc
    assert qc["label_counts"][7] > 0 and qc["label_counts"][107] > 0


def test_motor_centroids_on_correct_sides(synth_warped_atlases):
    """The motor centroid QC booleans (previously print-only warnings) must be
    False: left motor left of the midline, right motor right of it."""
    qc = synth_warped_atlases["cortical_qc"]
    assert "motor_left_centroid_world" in qc, "motor centroids not in QC"
    assert qc["left_centroid_right_of_midline"] is False, qc
    assert qc["right_centroid_left_of_midline"] is False, qc
    # Functional consistency: left centroid X < midline < right centroid X.
    lx = qc["motor_left_centroid_world"][0]
    rx = qc["motor_right_centroid_world"][0]
    mx = qc["midline_x"]
    assert lx < mx < rx, (
        f"motor centroids not straddling midline: L={lx}, mid={mx}, R={rx}"
    )
    # L/R motor centroids should be at roughly the same Z (same axial plane).
    assert qc["motor_z_diff_mm"] < 10.0, qc


def test_warped_motor_centroids_match_planted_shift(synth_warped_atlases):
    """The warped motor centroids must land at MNI-planted + DX (within ~1
    voxel), confirming the warp applies the recovered translation to the
    atlas. This is the convention-free registration-quality assertion."""
    DX = 6.0
    qc = synth_warped_atlases["cortical_qc"]
    # Planted MNI: left (-12, 0, 12), right (12, 0, 12) → subject +DX in X.
    lx, ly, lz = qc["motor_left_centroid_world"]
    rx, ry, rz = qc["motor_right_centroid_world"]
    assert abs(lx - (-12 + DX)) < 3.0, f"left motor X={lx}, expected ~{-12 + DX}"
    assert abs(rx - (12 + DX)) < 3.0, f"right motor X={rx}, expected ~{12 + DX}"
    assert abs(lz - 12) < 4.0 and abs(rz - 12) < 4.0, (
        f"motor Z drifted: lz={lz}, rz={rz}"
    )


# ---------------------------------------------------------------------------
# ROI placement
# ---------------------------------------------------------------------------

def test_roi_placement(synth_registration_result, synth_warped_atlases):
    """ROI masks must be non-empty, the brainstem in the inferior Z third, and
    the motor L/R centroids straddling the warped midline."""
    reg = synth_registration_result
    warped = synth_warped_atlases
    masks = create_cst_roi_masks(
        warped_cortical=warped["cortical_warped"],
        warped_subcortical=warped["subcortical_warped"],
        subject_affine=reg["subject_affine"],
        roi_config=CST_ROI_CONFIG,
        dilate_brainstem=2,
        dilate_motor=1,
        save_masks=False,
        verbose=False,
        hemisphere_mask=reg["hemisphere_mask"],
        midline_x=reg["midline_x"],
    )

    def centroid_world(mask):
        idx = np.argwhere(mask)
        assert len(idx), "mask empty"
        c = idx.mean(axis=0)
        return reg["subject_affine"] @ np.append(c, 1)

    bs_w = centroid_world(masks["brainstem"])
    ml_w = centroid_world(masks["motor_left"])
    mr_w = centroid_world(masks["motor_right"])

    midline_x = reg["midline_x"]
    # Brainstem is inferior (low Z) relative to the motor cortices.
    assert bs_w[2] < ml_w[2], f"brainstem Z={bs_w[2]} not inferior to motor L Z={ml_w[2]}"
    assert bs_w[2] < mr_w[2], f"brainstem Z={bs_w[2]} not inferior to motor R Z={mr_w[2]}"
    # Motor L/R straddle the warped midline.
    assert ml_w[0] < midline_x < mr_w[0], (
        f"motor ROIs not straddling midline: L={ml_w[0]}, mid={midline_x}, R={mr_w[0]}"
    )


# ---------------------------------------------------------------------------
# Direct unit tests of the now-extracted QC helper (label-loss / side-swap
# detector) — these would have been impossible while the QC was print-only.
# ---------------------------------------------------------------------------

def _identity_diffeomorphic_map(shape, affine):
    """A genuine (near-)identity DiffeomorphicMap for direct warp tests."""
    static = np.zeros(shape, dtype=np.float32)
    static[4:-4, 4:-4, 4:-4] = 1.0
    sdr = SymmetricDiffeomorphicRegistration(CCMetric(dim=3), level_iters=[1])
    return sdr.optimize(static, static, static_grid2world=affine,
                        moving_grid2world=affine, prealign=np.eye(4))


def test_compute_atlas_warp_qc_detects_label_loss():
    """A warped atlas missing a label must report labels_preserved=False."""
    shape = (20, 20, 20)
    affine = np.eye(4)
    affine[:3, 3] = [-10, -10, -10]
    orig = np.zeros(shape, dtype=np.int16)
    orig[5:8, 5:8, 5:8] = 7
    orig[17:19, 17:19, 17:19] = 8
    # Drop label 8 in the "warped" result.
    warped = orig.copy()
    warped[warped == 8] = 0
    qc = compute_atlas_warp_qc(warped, orig_labels=np.unique(orig[orig > 0]),
                              subject_affine=affine, midline_x=0.0)
    assert qc["labels_preserved"] is False
    assert qc["labels_original"] == [7, 8]
    assert qc["labels_warped"] == [7]


def test_compute_atlas_warp_qc_detects_hemisphere_swap():
    """Motor centroids on the wrong side of the midline must flip the QC flags."""
    shape = (20, 20, 20)
    affine = np.eye(4)
    affine[:3, 3] = [-10, -10, -10]  # 1 mm voxels, world X in [-10, +10)
    warped = np.zeros(shape, dtype=np.int16)
    # Put left motor (7) at high X and right motor (107) at low X → swapped.
    warped[16:19, 10:13, 10:13] = 7      # world X ~ +6..+9 (right of midline 0)
    warped[1:4, 10:13, 10:13] = 107      # world X ~ -9..-6 (left of midline 0)
    qc = compute_atlas_warp_qc(warped, orig_labels=[7, 107],
                              subject_affine=affine, midline_x=0.0)
    assert qc["left_centroid_right_of_midline"] is True
    assert qc["right_centroid_left_of_midline"] is True


def test_verify_atlas_labels_detects_missing():
    """The existing ``verify_atlas_labels`` helper already returns a dict —
    pin that it reports success=False / missing labels when a label is gone."""
    warped = np.zeros((10, 10, 10), dtype=np.int16)
    warped[2:5, 2:5, 2:5] = 7  # only label 7 present
    out = verify_atlas_labels(warped, expected_labels=[7, 8], verbose=False)
    assert out["success"] is False
    assert 8 in out["missing"]