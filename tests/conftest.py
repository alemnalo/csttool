import pytest
import numpy as np
import nibabel as nib
from pathlib import Path
from dipy.io.stateful_tractogram import StatefulTractogram, Space
from dipy.io.utils import create_nifti_header
from dipy.io.streamline import save_trk
from dipy.tracking.streamline import Streamlines

# ---------------------------------------------------------------------------
# AU28 / AU33 synthetic registration+atlas+tractogram scene
#
# A small, fully self-contained scene (no Tier-2 fetched data) that lets the
# *real* scientific core run end-to-end in CI: affine+SyN registration, atlas
# warping, ROI construction and pass-through filtering. The bundled 197^3 MNI
# template and the FSL-licensed Harvard-Oxford/FMRIB58 atlases are deliberately
# NOT used — the scene ships its own 40^3 MNI template + synthetic atlas so the
# test never depends on `csttool fetch-data` having been run.
#
# Geometry: 40^3 grid, 2 mm isotropic, RAS, centred so voxel (20,20,20) ~= world
# (0,0,0). The "subject" is the template array with the affine shifted by +DX mm
# in world X (DX = 6 mm), so registration must recover a ~DX translation and the
# warped atlas lands at MNI-atlas-world-coords + DX. The atlas carries the
# Harvard-Oxford CST labels (brainstem=8; motor_left=7; motor_right=107 after the
# MNI-space hemisphere split). Streamlines are drawn in subject space through the
# expected warped ROI locations.
# ---------------------------------------------------------------------------
SYNTH_GRID_SHAPE = (40, 40, 40)
SYNTH_VOX_MM = 2.0
SYNTH_DX_MM = 6.0
SYNTH_LEVEL_ITERS_AFFINE = [100, 10, 1]
SYNTH_LEVEL_ITERS_SYN = [2, 2, 1]


def _synth_affine():
    """2 mm isotropic RAS affine centred on world (0,0,0)."""
    a = np.eye(4)
    a[:3, :3] = np.eye(3) * SYNTH_VOX_MM
    a[:3, 3] = [-(SYNTH_GRID_SHAPE[0] // 2) * SYNTH_VOX_MM,
                -(SYNTH_GRID_SHAPE[1] // 2) * SYNTH_VOX_MM,
                -(SYNTH_GRID_SHAPE[2] // 2) * SYNTH_VOX_MM]
    return a


def _world_to_vox(xyz):
    return (np.asarray(xyz) - _synth_affine()[:3, 3]) / SYNTH_VOX_MM


def _make_mni_template():
    """Ellipsoid 'brain' envelope + a few bright internal blobs for CC structure."""
    from scipy.ndimage import gaussian_filter
    shape = SYNTH_GRID_SHAPE
    i, j, k = np.meshgrid(np.arange(shape[0]), np.arange(shape[1]), np.arange(shape[2]),
                          indexing='ij')
    cx = np.array(shape) / 2
    r = (((i - cx[0]) / 16.0) ** 2 + ((j - cx[1]) / 14.0) ** 2
         + ((k - cx[2]) / 16.0) ** 2)
    data = np.zeros(shape, dtype=np.float32)
    data[r <= 1.0] = 1.0
    for c, amp in [((18, 20, 24), 1.5), ((22, 20, 16), 1.3),
                   ((20, 14, 20), 1.2), ((20, 26, 20), 1.2)]:
        d2 = (i - c[0]) ** 2 + (j - c[1]) ** 2 + (k - c[2]) ** 2
        data += amp * np.exp(-d2 / (2 * 3.0 ** 2))
    data = gaussian_filter(data, 1.0).astype(np.float32)
    return data


def _make_subject_fa(mni_data):
    """Subject = template array, affine shifted by +DX mm in world X."""
    affine = _synth_affine()
    subj_affine = affine.copy()
    subj_affine[0, 3] += SYNTH_DX_MM
    return mni_data.copy(), subj_affine


def _blob(arr, center_world, label, radius_vox=3):
    c = _world_to_vox(center_world)
    shape = arr.shape
    i, j, k = np.meshgrid(np.arange(shape[0]), np.arange(shape[1]), np.arange(shape[2]),
                          indexing='ij')
    d2 = (i - c[0]) ** 2 + (j - c[1]) ** 2 + (k - c[2]) ** 2
    arr[d2 <= radius_vox ** 2] = label


def _make_atlas():
    """Synthetic MNI-space atlas with CST labels (pre hemisphere split).

    brainstem=8 (subcortical), precentral=7 placed in BOTH hemispheres (the
    cortical atlas is split into 7/107 by ``split_atlas_hemispheres_mni``).
    """
    sub = np.zeros(SYNTH_GRID_SHAPE, dtype=np.int32)
    cort = np.zeros(SYNTH_GRID_SHAPE, dtype=np.int32)
    _blob(sub, [0, 0, -15], 8, radius_vox=3)      # brainstem (inferior-central)
    _blob(cort, [-12, 0, 12], 7, radius_vox=3)    # left precentral (MNI X<0)
    _blob(cort, [12, 0, 12], 7, radius_vox=3)     # right precentral (split adds 100)
    return sub, cort


def _make_streamlines():
    """Subject-space streamlines: 2 left CST, 2 right CST, 1 junk."""
    bs = np.array([0, 0, -15], dtype=float)
    mleft = np.array([-12, 0, 12], dtype=float)
    mright = np.array([12, 0, 12], dtype=float)
    shift = np.array([SYNTH_DX_MM, 0, 0], dtype=float)

    def curved(start, end, n=40, lateral=2.0):
        t = np.linspace(0, 1, n)
        base = start[None, :] * (1 - t)[:, None] + end[None, :] * t[:, None]
        base[:, 1] += lateral * np.sin(np.pi * t)
        return base.astype(np.float32)

    sls = [
        curved(mleft + shift, bs + shift),
        curved(mleft + shift + np.array([-1, 1, 0]), bs + shift + np.array([0, -1, 0])),
        curved(mright + shift, bs + shift, lateral=-2.0),
        curved(mright + shift + np.array([1, 1, 0]), bs + shift + np.array([0, -1, 0]),
               lateral=-2.0),
        curved(np.array([0, 18, 12]) + shift, np.array([0, 18, -16]) + shift),  # junk
    ]
    return Streamlines(sls)


@pytest.fixture(scope="session")
def synth_mni_template(tmp_path_factory):
    """Synthetic 40^3 MNI template written to disk + in-memory arrays."""
    d = tmp_path_factory.mktemp("synth_mni")
    data = _make_mni_template()
    affine = _synth_affine()
    img = nib.Nifti1Image(data, affine)
    path = d / "mni_template.nii.gz"
    nib.save(img, path)
    return {"path": path, "img": img, "data": data, "affine": affine}


@pytest.fixture(scope="session")
def synth_subject_fa(synth_mni_template, tmp_path_factory):
    """Subject FA = template array with +DX mm world-X affine shift."""
    d = tmp_path_factory.mktemp("synth_subject")
    data, subj_affine = _make_subject_fa(synth_mni_template["data"])
    img = nib.Nifti1Image(data, subj_affine)
    path = d / "subject_fa.nii.gz"
    nib.save(img, path)
    return {"path": path, "img": img, "data": data, "affine": subj_affine}


@pytest.fixture(scope="session")
def synth_atlas_images(tmp_path_factory):
    """Synthetic MNI-space subcortical + cortical atlas NIfTI images."""
    d = tmp_path_factory.mktemp("synth_atlas")
    affine = _synth_affine()
    sub_data, cort_data = _make_atlas()
    sub_img = nib.Nifti1Image(sub_data, affine)
    cort_img = nib.Nifti1Image(cort_data, affine)
    nib.save(sub_img, d / "subcortical.nii.gz")
    nib.save(cort_img, d / "cortical.nii.gz")
    return {"subcortical_img": sub_img, "cortical_img": cort_img,
            "subcortical_data": sub_data, "cortical_data": cort_data,
            "affine": affine}


@pytest.fixture(scope="session")
def synth_wholebrain_tractogram(synth_subject_fa, tmp_path_factory):
    """Subject-space whole-brain tractogram on disk + raw Streamlines."""
    d = tmp_path_factory.mktemp("synth_trk")
    sls = _make_streamlines()
    ref_img = synth_subject_fa["img"]
    sft = StatefulTractogram(list(sls), ref_img, Space.RASMM)
    path = d / "wholebrain.trk"
    save_trk(sft, str(path), bbox_valid_check=False)
    return {"streamlines": sls, "path": path, "ref_img": ref_img}


@pytest.fixture(scope="session")
def synth_registration_result(synth_subject_fa, synth_mni_template):
    """Real (unmocked) affine+SyN registration of the synthetic subject to MNI.

    Session-scoped because it is the most expensive piece (~2 s) and every AU28/
    AU33 test consumes its outputs (mapping, midline, Jacobian, affines).
    """
    from csttool.extract.modules.registration import register_mni_to_subject
    out_dir = synth_subject_fa["path"].parent / "reg_out"
    out_dir.mkdir(exist_ok=True)
    reg = register_mni_to_subject(
        subject_fa_path=str(synth_subject_fa["path"]),
        output_dir=str(out_dir),
        mni_template_path=str(synth_mni_template["path"]),
        level_iters_affine=SYNTH_LEVEL_ITERS_AFFINE,
        level_iters_syn=SYNTH_LEVEL_ITERS_SYN,
        use_fa_template=False,
        generate_qc=False,
        save_warped=False,
        verbose=False,
    )
    return reg


@pytest.fixture(scope="session")
def synth_warped_atlases(synth_registration_result, synth_atlas_images):
    """Real (unmocked) warp of the synthetic atlases to subject space.

    Returns the cortical/subcortical warped arrays, their QC dicts, the split
    cortical image, and the registration result.
    """
    from csttool.extract.modules.warp_atlas_to_subject import (
        warp_atlas_to_subject, split_atlas_hemispheres_mni,
    )
    reg = synth_registration_result
    sub_w, sub_qc = warp_atlas_to_subject(
        synth_atlas_images["subcortical_img"], reg["mapping"],
        reg["subject_shape"], reg["subject_affine"],
        mni_shape=reg["mni_shape"], mni_affine=reg["mni_affine"],
        midline_x=reg["midline_x"], verbose=False)
    cort_split = split_atlas_hemispheres_mni(synth_atlas_images["cortical_img"],
                                              verbose=False)
    cort_w, cort_qc = warp_atlas_to_subject(
        cort_split, reg["mapping"], reg["subject_shape"], reg["subject_affine"],
        mni_shape=reg["mni_shape"], mni_affine=reg["mni_affine"],
        midline_x=reg["midline_x"], verbose=False)
    return {"cortical_warped": cort_w, "subcortical_warped": sub_w,
            "cortical_qc": cort_qc, "subcortical_qc": sub_qc,
            "cortical_split_img": cort_split, "reg": reg}


@pytest.fixture
def synthetic_affine():
    """Returns a simple identity affine for testing."""
    return np.eye(4)

@pytest.fixture
def synthetic_image_data():
    """Returns a small 10x10x10 synthetic 3D image."""
    data = np.zeros((10, 10, 10), dtype=np.float32)
    # Create some "features"
    data[2:8, 2:8, 2:8] = 1.0  # Cube in the center
    return data

@pytest.fixture
def synthetic_nifti(synthetic_image_data, synthetic_affine):
    """Returns a synthetic nibabel Nifti1Image."""
    return nib.Nifti1Image(synthetic_image_data, synthetic_affine)

@pytest.fixture
def synthetic_tractogram(synthetic_nifti, synthetic_affine):
    """Returns a simple synthetic StatefulTractogram."""
    # Create a few straight streamlines
    # Streamline 1: straight line along Z axis
    sl1 = np.array([[5, 5, 2], [5, 5, 3], [5, 5, 4], [5, 5, 5], [5, 5, 6], [5, 5, 7], [5, 5, 8]], dtype=np.float32)
    # Streamline 2: slightly offset
    sl2 = np.array([[4, 4, 2], [4, 4, 3], [4, 4, 4], [4, 4, 5], [4, 4, 6], [4, 4, 7], [4, 4, 8]], dtype=np.float32)
    
    streamlines = [sl1, sl2]
    
    sft = StatefulTractogram(streamlines, synthetic_nifti, Space.RASMM)
    return sft

@pytest.fixture
def synthetic_z_gradient_data():
    """Returns a 10x10x10 volume ramping linearly from 0.0 to 1.0 along +Z.

    Unlike synthetic_image_data (a flat cube), this varies along the tract axis, so a
    profile computed on it changes if streamline orientation changes. A flat map cannot
    detect a flip, which is what let AU9 go unnoticed.
    """
    data = np.zeros((10, 10, 10), dtype=np.float32)
    data[:, :, :] = np.arange(10, dtype=np.float32) / 9.0
    return data

@pytest.fixture
def synthetic_z_gradient_nifti(synthetic_z_gradient_data, synthetic_affine):
    """Returns a Nifti1Image of the Z ramp. With an identity affine, world Z == voxel k."""
    return nib.Nifti1Image(synthetic_z_gradient_data, synthetic_affine)

@pytest.fixture
def synthetic_bvals():
    """Returns synthetic b-values (1 b0 and 6 DWIs)."""
    return np.array([0, 1000, 1000, 1000, 1000, 1000, 1000])

@pytest.fixture
def synthetic_bvecs():
    """Returns synthetic b-vectors (1 b0 and 6 directions)."""
    # 6 directions along axes + diagonals
    bvecs = np.array([
        [0, 0, 0],
        [1, 0, 0],
        [-1, 0, 0],
        [0, 1, 0],
        [0, -1, 0],
        [0, 0, 1],
        [0, 0, -1]
    ])
    # Normalize non-zero vectors
    norm = np.linalg.norm(bvecs[1:], axis=1, keepdims=True)
    bvecs[1:] = bvecs[1:] / norm
    return bvecs

@pytest.fixture
def synthetic_gtab(synthetic_bvals, synthetic_bvecs):
    """Returns a synthetic gradient table."""
    from dipy.core.gradients import gradient_table
    return gradient_table(synthetic_bvals, bvecs=synthetic_bvecs)
