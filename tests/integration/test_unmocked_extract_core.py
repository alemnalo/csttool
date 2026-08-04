"""AU28 — unmocked end-to-end coverage of the extraction scientific core.

The existing ``tests/integration/test_pipeline.py`` patches
``register_mni_to_subject``, ``load_mni_template``, ``fetch_harvard_oxford``,
``extract_cst_passthrough`` and ``validate_tractogram_coordinates`` with
``MagicMock``, so registration → warping → ROI construction → filtering have no
automated end-to-end coverage. This module exercises the *real* implementations
of those four stages on a small, fully-synthetic, self-contained scene (see the
``synth_*`` session fixtures in ``tests/conftest.py``). The only piece left
mocked anywhere in the pipeline is ``fetch_harvard_oxford``, which fetches the
FSL-licensed Tier-2 atlas that cannot be assumed present in CI; we substitute a
synthetic atlas of the same geometry instead.

What runs unmocked here:
  - ``register_mni_to_subject`` (affine + SyN)  [synth_registration_result]
  - ``warp_atlas_to_subject`` + ``split_atlas_hemispheres_mni``
    [synth_warped_atlases]
  - ``create_cst_roi_masks``
  - ``extract_cst_passthrough``
  - ``validate_tractogram_coordinates``
"""

import numpy as np

from csttool.extract.modules.create_roi_masks import create_cst_roi_masks
from csttool.extract.modules.passthrough_filtering import extract_cst_passthrough
from csttool.extract.modules.warp_atlas_to_subject import CST_ROI_CONFIG
from csttool.extract.modules.coordinate_validation import (
    validate_tractogram_coordinates,
)


def test_unmocked_extract_core_end_to_end(
    synth_registration_result,
    synth_warped_atlases,
    synth_wholebrain_tractogram,
    synth_subject_fa,
):
    """Run registration→warp→ROI→filter→coord-validation unmocked and assert.

    The synthetic scene plants two left-CST and two right-CST streamlines (each
    running from a precentral ROI through the brainstem) plus one junk
    streamline through an unrelated anterior region. A correct end-to-end run
    must recover the four CST streamlines, split them by the correct hemisphere,
    and exclude the junk — and the tractogram/FA coordinate validator must pass
    on the synthetic tractogram (one of the five sites AU28 names as mocked).
    """
    reg = synth_registration_result
    warped = synth_warped_atlases

    # --- Stage: ROI mask construction (real) ---
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
    for key in ("brainstem", "motor_left", "motor_right"):
        assert masks[key] is not None and masks[key].sum() > 0, (
            f"{key} mask is empty — warping/ROI construction produced nothing"
        )

    # --- Stage: pass-through filtering (real) ---
    res = extract_cst_passthrough(
        streamlines=synth_wholebrain_tractogram["streamlines"],
        masks=masks,
        affine=reg["subject_affine"],
        min_length=20.0,
        max_length=200.0,
        midline_x=reg["midline_x"],
        midline_distance=reg["midline_distance"],
        verbose=False,
    )

    stats = res["stats"]
    assert stats["extraction_rate"] > 0.0, "no streamlines extracted at all"
    # Four planted CST streamlines (2 L + 2 R); the junk streamline is excluded.
    assert stats["cst_left_count"] == 2, stats
    assert stats["cst_right_count"] == 2, stats
    assert stats["cst_total_count"] == 4, stats

    # --- L/R assignment must be anatomically correct (left < midline < right) ---
    midline_x = reg["midline_x"]

    def centroid_x(streamlines):
        pts = np.vstack([np.asarray(sl) for sl in streamlines])
        return float(np.mean(pts[:, 0]))

    assert len(res["cst_left"]) == 2 and len(res["cst_right"]) == 2
    left_cx = centroid_x(res["cst_left"])
    right_cx = centroid_x(res["cst_right"])
    assert left_cx < midline_x < right_cx, (
        f"L/R assignment wrong: left centroid X={left_cx:.2f}, "
        f"midline={midline_x:.2f}, right centroid X={right_cx:.2f}"
    )

    # --- Stage: coordinate validation (real) — one of AU28's mocked sites ---
    v = validate_tractogram_coordinates(
        synth_wholebrain_tractogram["path"],
        synth_subject_fa["path"],
        strict=True,
        verbose=False,
    )
    assert v["valid"], (
        f"coordinate validation failed on the synthetic tractogram: {v['errors']}"
    )
