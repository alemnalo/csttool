"""AU31 edge case: truncated NIfTI file.

A truncated .nii.gz loads its header (nib.load succeeds, shape looks correct)
but reading the data array raises EOFError deep in the pipeline. AU31 surfaces
this at the load boundary with a clear "file truncated" message naming the
path, rather than a raw EOFError three frames down.
"""

import pytest

from csttool.preprocess.modules.load_dataset import load_dataset


def test_truncated_nifti_load_raises_clear_error(truncated_nifti, valid_dwi_nifti):
    """load_dataset must reject a truncated NIfTI with a clear message.

    The truncated fixture shares the bval/bvec sidecars of the valid DWI, so
    the gradient table (AU21) validates fine; the failure must come from the
    data-read check with "truncated" in the message.
    """
    trunc_path = truncated_nifti
    data_dir = trunc_path.parent
    # Reuse the valid sidecars (dwi.bval / dwi.bvec) by naming the truncated
    # file to match the valid stem so load_dataset finds the gradients. The
    # rename overwrites the valid dwi.nii.gz (fine: tmp_path is per-test).
    trunc_path = trunc_path.rename(data_dir / "dwi.nii.gz")
    with pytest.raises(ValueError, match="truncated") as exc_info:
        load_dataset(str(data_dir), "dwi")
    assert "dwi.nii.gz" in str(exc_info.value)


def test_truncated_nifti_track_path_clear_error(truncated_nifti, capsys):
    """The track CLI path must also report truncation clearly, not a raw EOFError."""
    from csttool.cli.commands import track as track_mod

    trunc = truncated_nifti
    assert trunc.exists()
    args = type("A", (), {
        "nifti": trunc,
        "out": trunc.parent / "track_out",
        "subject_id": None,
        "rng_seed": 42,
        "fa_thr": 0.2,
        "sh_order": 2,
        "seed_density": 1,
        "step_size": 0.5,
        "fit_method": "WLS",
        "verbose": False,
        "show_plots": False,
        "save_visualizations": False,
        "use_brain_mask_stop": False,
    })()
    args.out.mkdir(parents=True, exist_ok=True)
    result = track_mod.cmd_track(args)
    assert result is None  # cmd_track returns None on failure
    captured = capsys.readouterr()
    assert "truncated" in captured.out.lower()
    assert trunc.name in captured.out
