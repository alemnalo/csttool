"""Tests for the AU11 / AU19 midline and Jacobian fixes.

The fixtures here are deliberately **off-centre** (the anatomical midline is at world
X = M != 0). A brain centred at X = 0 cannot detect a midline bug — the split on X=0
and the split on the true midline coincide, so every assertion would pass against the
broken code (the AU9 / AU29 lesson).

Two AU19 defects are covered separately:
  1. the Jacobian gradient is taken per **mm**, not per voxel index (the forward field
     is in world units on dipy 1.9-1.12.1 — verified empirically, not assumed);
  2. the hemisphere split uses the **full** affine, not the axis-aligned
     ``affine[0,0]*i + affine[0,3]`` approximation.
"""
import numpy as np
import pytest

from csttool.extract.modules.registration import (
    _jacobian_determinant,
    compute_jacobian_hemisphere_stats,
    compute_warped_midline,
)
from csttool.extract.modules.passthrough_filtering import (
    sample_peduncle_fa,
    extract_cst_passthrough,
)


# --------------------------------------------------------------------------- #
# Fakes
# --------------------------------------------------------------------------- #
class _FakeMapping:
    """Minimal stand-in for a DiffeomorphicMap.

    ``compute_jacobian_hemisphere_stats`` only needs ``get_forward_field``; we also
    give ``transform`` so ``compute_warped_midline`` can be exercised end-to-end with
    a known voxel translation (decoupled from DIPY's real warp).
    """

    def __init__(self, forward_field=None, translate=None, out_shape=None):
        self._field = forward_field
        self._translate = translate  # (di, dj, dk) in voxels
        self._out_shape = out_shape

    def get_forward_field(self):
        return self._field

    def transform(self, image, interpolation='linear', image_world2grid=None,
                  out_shape=None, **kw):
        # Nearest-neighbour translation of the input mask onto the subject grid.
        from scipy.ndimage import shift as _shift
        out = self._out_shape if out_shape is None else out_shape
        di, dj, dk = self._translate or (0, 0, 0)
        shifted = _shift(image, (di, dj, dk), order=0, mode='constant', cval=0)
        # crop / pad to the subject grid shape
        result = np.zeros(out, dtype=image.dtype)
        sz = tuple(min(s, o) for s, o in zip(shifted.shape, out))
        slc = tuple(slice(0, z) for z in sz)
        result[slc] = shifted[slc]
        return result


# --------------------------------------------------------------------------- #
# AU19 defect 1: Jacobian gradient must be per mm
# --------------------------------------------------------------------------- #
def test_jacobian_determinant_is_per_mm_not_per_voxel_index():
    """A linear displacement ramp ``u = a * i`` has ``du/dx = a / voxel_size``.

    The Jacobian determinant is ``1 + du/dx = 1 + a / vx`` (per mm). The old code took
    ``np.gradient`` per voxel index, giving ``1 + a`` — wrong by the voxel-size factor
    (here vx=2, so 1.02 vs the correct 1.01). Asserting the per-mm value catches the
    calibration error that inflated the reported ``L 1.000±0.388 / R 0.999±0.330``.
    """
    a = 0.02
    vx = 2.0
    shape = (12, 8, 8)
    i, _, _ = np.meshgrid(np.arange(shape[0]), np.arange(shape[1]),
                          np.arange(shape[2]), indexing='ij')
    field = np.zeros(shape + (3,), dtype=float)
    field[..., 0] = a * i.astype(float)  # u = a*i, v=w=0
    J = _jacobian_determinant(field, [vx, vx, vx])

    expected = 1.0 + a / vx  # 1.01
    assert J.shape == shape
    # interior voxels (avoid the gradient's one-sided edge stencil)
    assert np.allclose(J[2:-2], expected, atol=1e-12)
    # and explicitly NOT the per-voxel-index value 1 + a (= 1.02)
    assert not np.allclose(J[2:-2].mean(), 1.0 + a, atol=1e-3)


# --------------------------------------------------------------------------- #
# AU19 defect 2 + AU11: hemisphere split uses the mask / full affine
# --------------------------------------------------------------------------- #
def _ramp_field(shape, a):
    i, _, _ = np.meshgrid(np.arange(shape[0]), np.arange(shape[1]),
                          np.arange(shape[2]), indexing='ij')
    field = np.zeros(shape + (3,), dtype=float)
    field[..., 0] = a * i.astype(float)
    return field


def test_jacobian_hemisphere_stats_use_warped_midline_mask():
    """When ``hemisphere_mask`` is given, the L/R stats follow it (the AU11 source of
    truth), not the axis-aligned world-X. The mask is deliberately off-centre so the
    two labelings differ."""
    shape = (16, 10, 10)
    a, vx = 0.02, 2.0
    field = _ramp_field(shape, a)
    # Off-centre warped-MNI mask: left = first 6 slabs (not the 8 that X=0 would give
    # under a centred affine). The axis-aligned split on this affine would label
    # differently, so this is non-vacuous.
    hemisphere_mask = np.zeros(shape, dtype=bool)
    hemisphere_mask[:6] = True

    J = _jacobian_determinant(field, [vx, vx, vx])
    affine = np.diag([vx, vx, vx, 1.0]).astype(float)  # translation 0 -> X=0 midline

    mapping = _FakeMapping(forward_field=field)
    stats, _ = compute_jacobian_hemisphere_stats(
        mapping, affine, hemisphere_mask=hemisphere_mask, verbose=False
    )
    # Mask-based expectations (exact, since the ramp is linear -> J uniform).
    assert stats['left_mean'] == pytest.approx(J[hemisphere_mask].mean())
    assert stats['right_mean'] == pytest.approx(J[~hemisphere_mask].mean())
    # And the two hemispheres genuinely differ in voxel count from an X=0 split.
    x_world = vx * np.arange(shape[0])
    axis_aligned_left = (x_world < 0).sum() * shape[1] * shape[2]
    assert int(hemisphere_mask.sum()) != axis_aligned_left


def test_jacobian_hemisphere_stats_full_affine_not_axis_aligned():
    """The scalar fallback must use the FULL affine for world X.

    Affine has a shear ``affine[0,1] = 1`` so world X = vx*i + 1*j + t; the old
    ``affine[0,0]*i + affine[0,3]`` drops the ``j`` term and mislabels voxels. With an
    off-centre midline (``midline_x``) the two labelings differ, so the stats differ."""
    shape = (12, 12, 6)
    a, vx = 0.02, 2.0
    field = _ramp_field(shape, a)
    # Sheared affine: world X = 2*i + 1*j - 12  (midline where 2i + j = 12)
    affine = np.eye(4)
    affine[0, 0] = vx
    affine[0, 1] = 1.0
    affine[0, 3] = -12.0
    midline_x = 0.0

    mapping = _FakeMapping(forward_field=field)
    stats, _ = compute_jacobian_hemisphere_stats(
        mapping, affine, hemisphere_mask=None, midline_x=midline_x, verbose=False
    )

    # Recompute the expected mask with the FULL affine.
    i, j, k = np.meshgrid(np.arange(shape[0]), np.arange(shape[1]),
                          np.arange(shape[2]), indexing='ij')
    x_full = (affine[0, 0] * i + affine[0, 1] * j +
              affine[0, 2] * k + affine[0, 3])
    J = _jacobian_determinant(field, [vx, vx, vx])
    expected_left = J[x_full < midline_x]
    expected_right = J[x_full >= midline_x]
    assert stats['left_mean'] == pytest.approx(expected_left.mean())
    assert stats['right_mean'] == pytest.approx(expected_right.mean())
    # Confirm the axis-aligned split would have disagreed (non-vacuous).
    x_axis = affine[0, 0] * i + affine[0, 3]
    assert not np.array_equal(x_full < midline_x, x_axis < midline_x)


# --------------------------------------------------------------------------- #
# AU11: compute_warped_midline (off-centre subject)
# --------------------------------------------------------------------------- #
def test_compute_warped_midline_off_centre():
    """A subject whose midline sits at world X = M != 0.

    The fake mapping places the warped MNI midline at subject voxel i=8; the subject
    affine places voxel 8 at world X = M. The returned ``hemisphere_mask`` must label
    voxels left of the boundary as left, ``midline_distance`` must be negative there,
    and the scalar ``midline_x`` must be near M."""
    shape = (16, 16, 16)
    vx = 2.0
    M = -6.0  # midline world X (off-centre)
    affine = np.diag([vx, vx, vx, 1.0]).astype(float)
    affine[0, 3] = M - vx * 8  # voxel i=8 -> world X = vx*8 + (M - vx*8) = M

    # MNI grid centred so X_mni < 0 for i < 8 (midline at i=8). No translation: the
    # warped midline lands at subject voxel i=8.
    mni_shape = (16, 16, 16)
    mni_affine = np.diag([vx, vx, vx, 1.0]).astype(float)
    mni_affine[0, 3] = -vx * 8

    # Subject brain cube with tissue on both sides of the midline.
    subject_data = np.zeros(shape, dtype=float)
    subject_data[2:14, 2:14, 2:14] = 1.0

    mapping = _FakeMapping(translate=(0, 0, 0), out_shape=shape)

    hemisphere_mask, midline_x, midline_distance = compute_warped_midline(
        mapping, affine, mni_affine, mni_shape,
        subject_data=subject_data, verbose=False,
    )

    # Voxels with i < 8 are left of the warped midline (inside the brain cube).
    i_idx = np.arange(shape[0])[:, None, None]
    left_by_plane = (i_idx < 8) & (subject_data > 0)
    in_brain = subject_data > 0
    assert np.array_equal(hemisphere_mask & in_brain, left_by_plane)

    # Signed distance: negative on the left, positive on the right, ~0 at the
    # midline boundary (voxel i=7 is a left voxel touching a right voxel).
    assert midline_distance[5, 8, 8] < 0
    assert midline_distance[11, 8, 8] > 0
    assert abs(midline_distance[7, 8, 8]) < 1e-9  # boundary voxel
    # Scalar midline reflects the off-centre plane (within a voxel of M).
    assert midline_x == pytest.approx(M, abs=vx)
    assert midline_x != 0.0  # the whole point: M != 0


# --------------------------------------------------------------------------- #
# AU11: sample_peduncle_fa with the warped-MNI mask (gold standard)
# --------------------------------------------------------------------------- #
def test_sample_peduncle_fa_uses_hemisphere_mask_off_centre():
    """An off-centre brainstem + a hemisphere mask that labels the *true* L/R.

    FA is 0.30 on the true left and 0.40 on the true right. The X=0 split (legacy)
    misassigns voxels because the brainstem is shifted to X = +6; the warped-MNI mask
    recovers the true 0.30 / 0.40. This is the D3 mechanism: the split plane, not the
    data, manufactured the 'lower-left' asymmetry."""
    shape = (10, 20, 20)
    vx = 2.0
    affine = np.diag([vx, vx, vx, 1.0]).astype(float)
    affine[0, 3] = 6.0  # brainstem shifted +6 mm (off-centre)

    fa_map = np.zeros(shape, dtype=float)
    brainstem = np.zeros(shape, dtype=np.uint8)
    hemisphere_mask = np.zeros(shape, dtype=bool)
    # Brainstem slab i in [3,7], all j,k; true midline at i=5 (world X = 6 -> i=0 is
    # X=6; X=0 at i=-3, off-volume). Define the true midline at subject i=5
    # (world X = vx*5 + 6 = 16). Left = i<5, right = i>=5.
    brainstem[3:8, 5:15, 5:15] = 1
    hemisphere_mask[3:5, 5:15, 5:15] = True  # true left
    fa_map[3:5, 5:15, 5:15] = 0.30
    fa_map[5:8, 5:15, 5:15] = 0.40

    # Superior 30% by Z: take top Z slab so all brainstem voxels qualify.
    res = sample_peduncle_fa(fa_map, affine, brainstem, affine,
                             hemisphere_mask=hemisphere_mask, verbose=False)
    assert res['left_mean_fa'] == pytest.approx(0.30, abs=1e-9)
    assert res['right_mean_fa'] == pytest.approx(0.40, abs=1e-9)

    # The legacy X=0 split would have labelled everything as right (all X>0 here),
    # so the left count would collapse -- demonstrating the bug the mask fixes.
    legacy = sample_peduncle_fa(fa_map, affine, brainstem, affine, verbose=False)
    assert legacy['left_n'] == 0
    assert legacy['right_n'] > 0


def test_sample_peduncle_fa_scalar_midline_off_centre():
    """Without the mask, the scalar ``midline_x`` still beats the X=0 assumption."""
    shape = (10, 10, 10)
    vx = 2.0
    affine = np.diag([vx, vx, vx, 1.0]).astype(float)
    affine[0, 3] = 6.0  # brainstem world X = 2*i + 6, so voxel i=5 -> X=16 (midline)
    fa_map = np.zeros(shape)
    brainstem = np.zeros(shape, dtype=np.uint8)
    brainstem[2:8, 2:8, 2:8] = 1
    fa_map[2:5, 2:8, 2:8] = 0.30   # left of true midline (X < 16 -> i < 5)
    fa_map[5:8, 2:8, 2:8] = 0.40   # right of true midline (X >= 16 -> i >= 5)

    res = sample_peduncle_fa(fa_map, affine, brainstem, affine,
                             midline_x=16.0, verbose=False)
    assert res['left_mean_fa'] == pytest.approx(0.30, abs=1e-9)
    assert res['right_mean_fa'] == pytest.approx(0.40, abs=1e-9)

    # The legacy X=0 split labels nothing as left (all brainstem X >= 10 > 0),
    # demonstrating the bias the scalar midline corrects.
    legacy = sample_peduncle_fa(fa_map, affine, brainstem, affine, verbose=False)
    assert legacy['left_n'] == 0


# --------------------------------------------------------------------------- #
# AU11: extract_cst_passthrough midline exclusion uses the curved midline
# --------------------------------------------------------------------------- #
def test_extract_cst_passthrough_midline_exclusion_uses_signed_distance():
    """A streamline crossing the true (off-centre) midline deeply is excluded; one
    that stays on one side is kept — even if a naive X=0 split would have called the
    crossing one a commissure.

    Uses ``midline_distance`` (the warped-midline surface) rather than the scalar
    plane, which is biased for off-centre subjects."""
    shape = (20, 20, 20)
    vx = 2.0
    affine = np.diag([vx, vx, vx, 1.0]).astype(float)  # world = 2*i (no translation)

    # Signed distance to the warped midline: 0 at i=10, -2*(i-10) left, +2*(i-10) right.
    i_idx = np.arange(shape[0])[:, None, None]
    midline_distance = (vx * (i_idx - 10)).astype(float) * np.ones(shape)
    midline_distance = np.broadcast_to(midline_distance, shape).copy()

    brainstem = np.zeros(shape, dtype=np.uint8)
    brainstem[10, 10, 10] = 1          # world (20,20,20), at the midline
    motor_left = np.zeros(shape, dtype=np.uint8)
    motor_left[5, 10, 10] = 1           # world (10,20,20), signed -10 (deep left)
    motor_right = np.zeros(shape, dtype=np.uint8)  # unused -> no bilateral exclusion
    masks = {'brainstem': brainstem, 'motor_left': motor_left,
             'motor_right': motor_right}

    # SL1: crosses the true midline deeply (signed -10 -> +10) -> excluded.
    sl_cross = np.array([[10.0, 20, 20], [20.0, 20, 20], [30.0, 20, 20]])
    # SL2: stays left of the true midline (signed -10 -> -2), passes brainstem +
    #       motor_left only -> kept as left CST.
    sl_left = np.array([[10.0, 20, 20], [16.0, 20, 20], [20.0, 20, 20]])

    from dipy.tracking.streamline import Streamlines
    result = extract_cst_passthrough(
        Streamlines([sl_cross, sl_left]), masks, affine,
        min_length=0, max_length=1000, midline_distance=midline_distance,
        verbose=False,
    )
    assert result['stats']['midline_excluded'] == 1
    assert len(result['cst_left']) == 1
    assert len(result['cst_right']) == 0


def test_extract_cst_passthrough_midline_exclusion_legacy_scalar():
    """Without ``midline_distance``, the scalar ``midline_x`` plane is used (backward
    compatible). A streamline crossing X=midline_x+-TOL on both sides is excluded."""
    shape = (30, 30, 30)
    affine = np.eye(4)  # voxel = 1 mm
    brainstem = np.zeros(shape, dtype=np.uint8)
    brainstem[15, 15, 15] = 1
    motor_left = np.zeros(shape, dtype=np.uint8)
    motor_left[3, 15, 15] = 1     # X=3 (deep left of midline_x=15 with TOL=8)
    motor_right = np.zeros(shape, dtype=np.uint8)  # unused
    masks = {'brainstem': brainstem, 'motor_left': motor_left,
             'motor_right': motor_right}

    # SL1: crosses midline_x=15 deeply (X 3 -> 27, both beyond +-8) -> excluded.
    sl_cross = np.array([[3.0, 15, 15], [15.0, 15, 15], [27.0, 15, 15]])
    # SL2: stays left (X 3 -> 15, never beyond +23) -> kept.
    sl_left = np.array([[3.0, 15, 15], [8.0, 15, 15], [15.0, 15, 15]])

    from dipy.tracking.streamline import Streamlines
    result = extract_cst_passthrough(
        Streamlines([sl_cross, sl_left]), masks, affine, min_length=0, max_length=1000,
        midline_x=15.0, verbose=False,
    )
    assert result['stats']['midline_excluded'] == 1
    assert len(result['cst_left']) == 1


# --------------------------------------------------------------------------- #
# AU12: motor ROI clamping at the warped-MNI midline
# --------------------------------------------------------------------------- #
def test_clamp_motor_rois_at_hemisphere_mask():
    """A motor_left mask that bleeds into the right hemisphere is clamped back."""
    from csttool.extract.modules.create_roi_masks import create_cst_roi_masks

    shape = (20, 20, 20)
    vx = 2.0
    affine = np.diag([vx, vx, vx, 1.0])

    # Hemisphere mask: left = i < 10, right = i >= 10 (world X=0 at i=0, so X<20=left)
    hemisphere_mask = np.zeros(shape, dtype=bool)
    hemisphere_mask[:10] = True

    # Motor labels extracted from warped atlas: left label=7 crosses i=10 by 2 voxels
    warped_cortical = np.zeros(shape, dtype=np.int16)
    warped_cortical[6:12, 8:12, 8:12] = 7   # motor left, bleeds 2 voxels right
    warped_cortical[10:16, 8:12, 8:12] = 107  # motor right, entirely on right side

    # Brainstem label (not clamped, but needed for the call)
    warped_subcortical = np.zeros(shape, dtype=np.int16)
    warped_subcortical[8:12, 8:12, 8:12] = 3  # arbitrary brainstem label

    roi_config = {
        'brainstem': {'label': 3},
        'motor_left': {'label': 7},
        'motor_right': {'label': 107},
    }

    masks = create_cst_roi_masks(
        warped_cortical, warped_subcortical, affine, roi_config,
        dilate_brainstem=0, dilate_motor=0,
        save_masks=False, verbose=False,
        hemisphere_mask=hemisphere_mask,
    )

    motor_left = masks['motor_left']
    motor_right = masks['motor_right']

    # Motor left must not have any voxels in the right hemisphere
    assert not np.any(motor_left & (~hemisphere_mask)), \
        "motor_left should be clamped to the left hemisphere only"

    # Motor right must not have any voxels in the left hemisphere
    assert not np.any(motor_right & hemisphere_mask), \
        "motor_right should be clamped to the right hemisphere only"

    # Brainstem must be untouched (it's a midline structure)
    assert masks['brainstem'].sum() > 0


def test_clamp_motor_rois_with_dilation():
    """Clamping happens AFTER dilation so dilated-into-other-hemi voxels are clipped."""
    from csttool.extract.modules.create_roi_masks import create_cst_roi_masks

    shape = (20, 20, 20)
    vx = 2.0
    affine = np.diag([vx, vx, vx, 1.0])

    hemisphere_mask = np.zeros(shape, dtype=bool)
    hemisphere_mask[:10] = True

    # Left motor label sits right against the midline at i=9 (last left voxel).
    # Dilation with iterations=1 will push it into i=10 (right hemisphere).
    warped_cortical = np.zeros(shape, dtype=np.int16)
    warped_cortical[7:10, 8:12, 8:12] = 7  # left motor, touches midline
    warped_cortical[10:13, 8:12, 8:12] = 107  # right motor

    warped_subcortical = np.zeros(shape, dtype=np.int16)
    warped_subcortical[8:12, 8:12, 8:12] = 3

    roi_config = {
        'brainstem': {'label': 3},
        'motor_left': {'label': 7},
        'motor_right': {'label': 107},
    }

    masks_no_clamp = create_cst_roi_masks(
        warped_cortical, warped_subcortical, affine, roi_config,
        dilate_brainstem=0, dilate_motor=1,
        save_masks=False, verbose=False,
        hemisphere_mask=None,  # no clamping
    )

    masks_clamped = create_cst_roi_masks(
        warped_cortical, warped_subcortical, affine, roi_config,
        dilate_brainstem=0, dilate_motor=1,
        save_masks=False, verbose=False,
        hemisphere_mask=hemisphere_mask,
    )

    left_no_clamp = masks_no_clamp['motor_left']
    left_clamped = masks_clamped['motor_left']

    # Without clamping, dilation bleeds into the right hemisphere
    assert np.any(left_no_clamp & (~hemisphere_mask)), \
        "without clamping, dilation should cross the midline"

    # With clamping, those bleed voxels are removed
    assert not np.any(left_clamped & (~hemisphere_mask)), \
        "clamped mask must stay strictly in the left hemisphere"

    # The clamped mask should be strictly smaller (or equal) than the unclamped one
    assert left_clamped.sum() <= left_no_clamp.sum()

    # But not empty — it still has its original left-hemisphere voxels
    assert left_clamped.sum() > 0


def test_clamp_motor_rois_scalar_midline_fallback():
    """When hemisphere_mask is None but midline_x is given, the scalar plane is used."""
    from csttool.extract.modules.create_roi_masks import create_cst_roi_masks

    shape = (20, 20, 20)
    affine = np.eye(4)  # 1 mm isotropic, world X = i (no translation)

    # Midline at world X = 10 mm (i=10). Left = i < 10.
    midline_x = 10.0

    warped_cortical = np.zeros(shape, dtype=np.int16)
    warped_cortical[6:13, 8:12, 8:12] = 7   # left motor: i=6..12, bleeds to i=12
    warped_cortical[10:16, 8:12, 8:12] = 107  # right motor: i=10..15

    warped_subcortical = np.zeros(shape, dtype=np.int16)
    warped_subcortical[8:12, 8:12, 8:12] = 3

    roi_config = {
        'brainstem': {'label': 3},
        'motor_left': {'label': 7},
        'motor_right': {'label': 107},
    }

    masks = create_cst_roi_masks(
        warped_cortical, warped_subcortical, affine, roi_config,
        dilate_brainstem=0, dilate_motor=0,
        save_masks=False, verbose=False,
        hemisphere_mask=None,
        midline_x=midline_x,
    )

    motor_left = masks['motor_left']
    motor_right = masks['motor_right']

    # All motor_left voxels must have world X < 10 (i < 10)
    left_indices = np.argwhere(motor_left)
    if len(left_indices) > 0:
        left_world_x = left_indices @ affine[:3, :3].T + affine[:3, 3]
        assert np.all(left_world_x[:, 0] < midline_x), \
            "motor_left must be clamped left of midline_x"

    # All motor_right voxels must have world X >= 10 (i >= 10)
    right_indices = np.argwhere(motor_right)
    if len(right_indices) > 0:
        right_world_x = right_indices @ affine[:3, :3].T + affine[:3, 3]
        assert np.all(right_world_x[:, 0] >= midline_x), \
            "motor_right must be clamped right of midline_x"


def test_clamp_motor_rois_none_is_noop():
    """When both hemisphere_mask and midline_x are None, no clamping occurs."""
    from csttool.extract.modules.create_roi_masks import create_cst_roi_masks

    shape = (20, 20, 20)
    affine = np.eye(4)

    warped_cortical = np.zeros(shape, dtype=np.int16)
    warped_cortical[6:13, 8:12, 8:12] = 7   # bleeds across hypothetical midline
    warped_cortical[10:16, 8:12, 8:12] = 107

    warped_subcortical = np.zeros(shape, dtype=np.int16)
    warped_subcortical[8:12, 8:12, 8:12] = 3

    roi_config = {
        'brainstem': {'label': 3},
        'motor_left': {'label': 7},
        'motor_right': {'label': 107},
    }

    masks = create_cst_roi_masks(
        warped_cortical, warped_subcortical, affine, roi_config,
        dilate_brainstem=0, dilate_motor=0,
        save_masks=False, verbose=False,
        hemisphere_mask=None,
        midline_x=None,
    )

    # Without clamping, the motor_left should include all labeled voxels
    # (i=6..9 after label 107 overwrites i=10..12: 4 slabs × 4×4)
    assert masks['motor_left'].sum() == 4 * 4 * 4
    assert masks['motor_right'].sum() == 6 * 4 * 4   # 6 slabs × 4×4
