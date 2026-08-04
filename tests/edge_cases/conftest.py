"""Shared fixtures for the AU31 edge-case suite.

Builds the six pathological-input classes the auditors named, reusing the
project's existing synthetic conventions (identity-ish RAS affines, unit DWI
bvecs) so the edge-case tests are consistent with the per-module suites.
"""

import numpy as np
import nibabel as nib
import pytest
from dipy.io.streamline import save_tractogram
from dipy.io.stateful_tractogram import StatefulTractogram, Space


# A small but physically-plausible 4D DWI shape used across the edge cases.
_DWI_SHAPE = (8, 8, 8, 7)  # 1 b0 + 6 DWI
_RAS_AFFINE = np.diag([2.0, 2.0, 2.0, 1.0])
_RAS_AFFINE[:3, 3] = [-8, -8, -8]


def _unit_bvecs_6dir():
    """Six unit DWI directions + a zero b0 row, as (N, 3)."""
    return np.array([
        [0, 0, 0],
        [1, 0, 0],
        [-1, 0, 0],
        [0, 1, 0],
        [0, -1, 0],
        [0, 0, 1],
        [0, 0, -1],
    ], dtype=float)


@pytest.fixture
def ras_affine():
    return _RAS_AFFINE.copy()


@pytest.fixture
def valid_bvals():
    return np.array([0, 1000, 1000, 1000, 1000, 1000, 1000])


@pytest.fixture
def valid_bvecs():
    return _unit_bvecs_6dir()


@pytest.fixture
def valid_dwi_nifti(tmp_path, valid_bvals, valid_bvecs):
    """A well-formed DWI NIfTI + sidecars on disk."""
    rng = np.random.default_rng(42)
    data = rng.random(_DWI_SHAPE).astype(np.float32) * 100
    img = nib.Nifti1Image(data, _RAS_AFFINE)
    nii = tmp_path / "dwi.nii.gz"
    nib.save(img, nii)
    np.savetxt(tmp_path / "dwi.bval", valid_bvals, fmt="%d")
    np.savetxt(tmp_path / "dwi.bvec", valid_bvecs.T, fmt="%.8f")
    return tmp_path, "dwi"


@pytest.fixture
def all_zero_dwi_data():
    """A 4D DWI volume that is entirely zero (pathological but well-shaped)."""
    return np.zeros(_DWI_SHAPE, dtype=np.float32)


@pytest.fixture
def single_direction_bvals():
    """1 b0 + 1 DWI — the minimum, and too few for any meaningful SH fit."""
    return np.array([0, 1000])


@pytest.fixture
def single_direction_bvecs():
    return np.array([[0, 0, 0], [1, 0, 0]], dtype=float)


@pytest.fixture
def single_direction_dwi_data():
    """4D DWI with 2 volumes (1 b0 + 1 DWI)."""
    rng = np.random.default_rng(7)
    return (rng.random((_DWI_SHAPE[0], _DWI_SHAPE[1], _DWI_SHAPE[2], 2))
            .astype(np.float32) * 100)


@pytest.fixture
def truncated_nifti(tmp_path, valid_dwi_nifti):
    """A NIfTI whose on-disk bytes are truncated mid-stream."""
    _dir, _name = valid_dwi_nifti
    src = _dir / f"{_name}.nii.gz"
    with open(src, "rb") as f:
        full = f.read()
    trunc = _dir / "truncated.nii.gz"
    # Keep the header but cut the gzip stream partway through the data.
    with open(trunc, "wb") as f:
        f.write(full[: len(full) // 2])
    return trunc


@pytest.fixture
def empty_tractogram_nifti(tmp_path):
    """A .trk containing zero streamlines, saved against a small FA-like ref."""
    aff = _RAS_AFFINE.copy()
    ref = nib.Nifti1Image(np.zeros(_DWI_SHAPE[:3], dtype=np.float32), aff)
    ref_path = tmp_path / "ref.nii.gz"
    nib.save(ref, ref_path)
    sft = StatefulTractogram([], ref, Space.RASMM)
    trk = tmp_path / "empty.trk"
    save_tractogram(sft, str(trk), bbox_valid_check=False)
    return trk, ref_path


@pytest.fixture
def missing_tag_dicom_dir(tmp_path):
    """A directory of synthetic DICOM stubs with NO metadata tags.

    Mirrors the existing tests/ingest/test_ingest.py convention: a 128-byte
    preamble + the ``DICM`` magic. These are not real DICOMs (no pixel data),
    which is exactly the "missing tags" / corrupt-DICOM edge case — pydicom can
    read the preamble but every series attribute is absent.
    """
    dcm_dir = tmp_path / "no_tags"
    dcm_dir.mkdir()
    for i in range(3):
        p = dcm_dir / f"img_{i:04d}.dcm"
        with open(p, "wb") as f:
            f.seek(128)
            f.write(b"DICM")
            f.write(b"\x00" * 50)
    return dcm_dir
