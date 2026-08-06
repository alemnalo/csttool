"""Tests for the FA-grid ROI segmentation product.

The three ``roi_*.nii.gz`` masks the extraction stage has always written are
unreadable downstream: they land in ``extraction/nifti/``, which ``csttool run``
deletes wholesale. This product is the same label array, written so it survives
and so it can be sliced against FA without resampling — which is the only reason
the report can show what constrained its own extraction.

The grid is therefore the thing under test. Registration reorients the subject
to RAS for warping, but FA, the CST density volume and the streamlines all stay
on the on-disk FA grid (LAS for a scanner-native conversion, csttool's normal
case). A dseg written in the internal RAS order would carry the right labels on
the wrong grid and be rejected by the renderer.
"""

import json

import nibabel as nib
import numpy as np
import pytest
from nibabel.orientations import axcodes2ornt, ornt_transform

from csttool.extract.modules.create_roi_masks import (
    ROI_DSEG_LABELS,
    create_cst_roi_masks,
    reorient_to_original,
    save_roi_dseg,
)

SHAPE = (12, 14, 10)
RAS_AFFINE = np.diag([2.0, 2.0, 2.0, 1.0])
# A scanner-native LAS grid: the X axis is flipped relative to RAS.
LAS_AFFINE = np.array([[-2.0, 0, 0, 22.0],
                       [0, 2.0, 0, -14.0],
                       [0, 0, 2.0, -10.0],
                       [0, 0, 0, 1.0]])


def _las_to_ras_transform():
    return ornt_transform(nib.io_orientation(LAS_AFFINE), axcodes2ornt(("R", "A", "S")))


def _combined():
    """A label array with three disjoint, differently-sized labels."""
    combined = np.zeros(SHAPE, dtype=np.uint8)
    combined[2:5, 3:6, 1:3] = 1     # brainstem
    combined[6:8, 8:11, 5:8] = 2    # motor left
    combined[9:11, 8:11, 5:8] = 3   # motor right
    return combined


class TestSaveRoiDseg:
    def test_written_on_the_fa_grid_when_the_subject_was_reoriented(self, tmp_path):
        """Shape *and* affine must equal FA's, so no resampling is ever needed."""
        combined = _combined()
        fa = nib.Nifti1Image(np.zeros(SHAPE, np.float32), LAS_AFFINE)

        path = save_roi_dseg(
            combined, RAS_AFFINE, tmp_path, "sub-x",
            original_subject_affine=LAS_AFFINE,
            reorientation_transform=_las_to_ras_transform(),
            verbose=False,
        )
        img = nib.load(str(path))
        assert img.shape == fa.shape
        np.testing.assert_array_equal(img.affine, fa.affine)

    def test_is_not_the_internal_ras_array(self, tmp_path):
        """The reorientation is real: on a LAS subject the stored array differs
        from the in-memory RAS one. A test that only checked the labels would
        pass on a dseg written to the wrong grid."""
        combined = _combined()
        path = save_roi_dseg(
            combined, RAS_AFFINE, tmp_path, "sub-x",
            original_subject_affine=LAS_AFFINE,
            reorientation_transform=_las_to_ras_transform(),
            verbose=False,
        )
        stored = np.asarray(nib.load(str(path)).dataobj)
        assert not np.array_equal(stored, combined)
        # And it is exactly the inverse reorientation, not some other permutation.
        np.testing.assert_array_equal(
            stored, reorient_to_original(combined, _las_to_ras_transform())
        )

    def test_already_ras_subject_is_written_as_is(self, tmp_path):
        combined = _combined()
        path = save_roi_dseg(combined, RAS_AFFINE, tmp_path, "sub-x", verbose=False)
        img = nib.load(str(path))
        np.testing.assert_array_equal(np.asarray(img.dataobj), combined)
        np.testing.assert_array_equal(img.affine, RAS_AFFINE)

    def test_labels_are_exactly_the_documented_set(self, tmp_path):
        path = save_roi_dseg(_combined(), RAS_AFFINE, tmp_path, "sub-x", verbose=False)
        stored = np.asarray(nib.load(str(path)).dataobj)
        assert set(np.unique(stored).tolist()) == {0, 1, 2, 3}
        assert ROI_DSEG_LABELS == {1: "brainstem", 2: "motor_left", 3: "motor_right"}

    def test_label_counts_match_the_source_masks(self, tmp_path):
        combined = _combined()
        path = save_roi_dseg(combined, RAS_AFFINE, tmp_path, "sub-x", verbose=False)
        stored = np.asarray(nib.load(str(path)).dataobj)
        for value in (1, 2, 3):
            assert np.count_nonzero(stored == value) == np.count_nonzero(combined == value)

    def test_dtype_is_uint8(self, tmp_path):
        """A label map is categorical; float storage would invite interpolation."""
        path = save_roi_dseg(_combined(), RAS_AFFINE, tmp_path, "sub-x", verbose=False)
        assert nib.load(str(path)).get_data_dtype() == np.uint8

    def test_filename_follows_the_bids_dseg_convention(self, tmp_path):
        path = save_roi_dseg(_combined(), RAS_AFFINE, tmp_path, "sub-x", verbose=False)
        assert path.name == "sub-x_space-orig_desc-CSTroi_dseg.nii.gz"

    def test_sidecar_names_every_label_and_its_space(self, tmp_path):
        path = save_roi_dseg(_combined(), RAS_AFFINE, tmp_path, "sub-x", verbose=False)
        sidecar = json.loads(
            path.with_name(path.name.replace(".nii.gz", ".json")).read_text()
        )
        assert sidecar["Labels"] == {"1": "brainstem", "2": "motor_left",
                                     "3": "motor_right"}
        assert sidecar["Space"] == "orig (FA grid)"
        assert sidecar["Description"]
        assert sidecar["VoxelCounts"]["brainstem"] == 18

    def test_voxel_counts_describe_the_stored_array(self, tmp_path):
        """The sidecar must count what was written, not what was passed in."""
        path = save_roi_dseg(
            _combined(), RAS_AFFINE, tmp_path, "sub-x",
            original_subject_affine=LAS_AFFINE,
            reorientation_transform=_las_to_ras_transform(),
            verbose=False,
        )
        stored = np.asarray(nib.load(str(path)).dataobj)
        sidecar = json.loads(
            path.with_name(path.name.replace(".nii.gz", ".json")).read_text()
        )
        for value, name in ROI_DSEG_LABELS.items():
            assert sidecar["VoxelCounts"][name] == np.count_nonzero(stored == value)

    def test_a_write_failure_returns_none_rather_than_raising(self, tmp_path):
        """An additive product must never break the extraction that made it."""
        blocked = tmp_path / "blocked"
        blocked.write_text("not a directory")
        with pytest.warns(UserWarning):
            assert save_roi_dseg(
                _combined(), RAS_AFFINE, blocked, "sub-x", verbose=False
            ) is None


class TestCreateCstRoiMasksEmitsTheDseg:
    """The product is written by the same call that writes the three masks."""

    def _atlases(self):
        cortical = np.zeros(SHAPE, dtype=np.int16)
        subcortical = np.zeros(SHAPE, dtype=np.int16)
        subcortical[3:5, 4:6, 2:4] = 8      # brainstem label
        cortical[6:8, 9:11, 6:8] = 7        # motor left label
        cortical[9:11, 9:11, 6:8] = 107     # motor right label
        return cortical, subcortical

    ROI_CONFIG = {
        "brainstem": {"label": 8},
        "motor_left": {"label": 7},
        "motor_right": {"label": 107},
    }

    def test_dseg_path_is_returned_and_written(self, tmp_path):
        cortical, subcortical = self._atlases()
        masks = create_cst_roi_masks(
            warped_cortical=cortical, warped_subcortical=subcortical,
            subject_affine=RAS_AFFINE, roi_config=self.ROI_CONFIG,
            dilate_brainstem=0, dilate_motor=0,
            output_dir=tmp_path, subject_id="sub-x", verbose=False,
        )
        path = masks["roi_dseg_path"]
        assert path is not None and path.exists()

    def test_dseg_labels_agree_with_the_returned_masks(self, tmp_path):
        cortical, subcortical = self._atlases()
        masks = create_cst_roi_masks(
            warped_cortical=cortical, warped_subcortical=subcortical,
            subject_affine=RAS_AFFINE, roi_config=self.ROI_CONFIG,
            dilate_brainstem=0, dilate_motor=0,
            output_dir=tmp_path, subject_id="sub-x", verbose=False,
        )
        stored = np.asarray(nib.load(str(masks["roi_dseg_path"])).dataobj)
        np.testing.assert_array_equal(stored == 1, masks["brainstem"])
        np.testing.assert_array_equal(stored == 2, masks["motor_left"])
        np.testing.assert_array_equal(stored == 3, masks["motor_right"])

    def test_dseg_differs_from_the_original_orientation_masks_when_reoriented(
        self, tmp_path
    ):
        """Guards the inverse case of the grid test above: when the reorientation
        transform is non-identity, the dseg is *not* the internal array."""
        cortical, subcortical = self._atlases()
        masks = create_cst_roi_masks(
            warped_cortical=cortical, warped_subcortical=subcortical,
            subject_affine=RAS_AFFINE, roi_config=self.ROI_CONFIG,
            dilate_brainstem=0, dilate_motor=0,
            output_dir=tmp_path, subject_id="sub-x", verbose=False,
            original_subject_affine=LAS_AFFINE,
            reorientation_transform=_las_to_ras_transform(),
        )
        stored = np.asarray(nib.load(str(masks["roi_dseg_path"])).dataobj)
        internal = np.zeros(SHAPE, dtype=np.uint8)
        internal[masks["brainstem"]] = 1
        internal[masks["motor_left"]] = 2
        internal[masks["motor_right"]] = 3
        assert not np.array_equal(stored, internal)
        np.testing.assert_array_equal(nib.load(str(masks["roi_dseg_path"])).affine,
                                      LAS_AFFINE)
