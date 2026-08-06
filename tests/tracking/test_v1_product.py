"""End-to-end test of the V1 world-frame data product (visualization-refactor M2).

Verifies that ``save_tracking_outputs`` writes ``{stem}_v1.nii.gz`` with the
NIfTI vector intent, the (X,Y,Z,1,3) shape, float32 dtype, a byte-identical
affine to the FA map, and a sidecar carrying the load-bearing ``VectorFrame``
and frame-diagnostic keys. Uses DIPY's real ``TensorModel`` so the eigenvectors
are genuine, not a mock.
"""

import json
from pathlib import Path

import nibabel as nib
import numpy as np
import pytest
from dipy.reconst.dti import TensorModel, color_fa
from dipy.core.gradients import gradient_table
from dipy.io.image import load_nifti

from csttool.tracking.modules.save_tracking_outputs import save_tracking_outputs


def _make_dwi(seed=0, affine=None, shape=(8, 9, 7)):
    rng = np.random.default_rng(seed)
    # 6 b0 + 30 DWI on a small sphere.
    bvals = np.concatenate([np.zeros(6), np.full(30, 1000.0)])
    bvecs = rng.normal(size=(36, 3))
    bvecs[6:] /= np.linalg.norm(bvecs[6:], axis=1, keepdims=True)
    bvecs[:6] = 0.0
    gtab = gradient_table(bvals, bvecs)
    # A smooth signal with an anisotropic tensor in every voxel so the fit has a
    # well-defined principal eigenvector (superior-inferior, axis 2).
    data = np.zeros(shape + (36,), dtype=np.float32)
    evals = np.array([1.0e-3, 0.3e-3, 0.3e-3])  # lambda1 > lambda2 == lambda3
    # Make the principal axis vary: axis-2 in the upper half, axis-0 lower half,
    # to confirm the rotation preserves each direction separately.
    for ix in range(shape[0]):
        for iy in range(shape[1]):
            for iz in range(shape[2]):
                e = evals.copy()
                if iz < shape[2] // 2:
                    # principal along voxel axis 2 (superior-inferior)
                    pass
                else:
                    e = np.array([0.3e-3, 0.3e-3, 1.0e-3])  # along axis 0? swap
                    e = np.array([1.0e-3, 0.3e-3, 0.3e-3])
        # (signal synthesis below is vectorised)
    # Build signal via the analytic single-tensor formula for speed.
    S0 = 1000.0
    b = 1000.0
    S = np.zeros(shape + (36,), dtype=np.float32)
    # anisotropic direction field
    v1_field = np.zeros(shape + (3,), dtype=np.float32)
    for iz in range(shape[2]):
        if iz < shape[2] // 2:
            v1 = np.array([0.0, 0.0, 1.0])  # along voxel Z
        else:
            v1 = np.array([1.0, 0.0, 0.0])  # along voxel X
        v1_field[:, :, iz, :] = v1
    evals_diag = np.array([1.0e-3, 0.3e-3, 0.3e-3])
    for k in range(36):
        if bvals[k] == 0:
            S[..., k] = S0
            continue
        g = bvecs[k]
        # ADC = g^T D g with D = evals in the v1 basis (v1, e2, e3).
        # Build an orthonormal basis around v1 per voxel.
        # For axis-aligned v1 this is just diag(evals permuted).
        # Compute g^T D g vectorised.
        D = np.zeros(shape + (3, 3))
        # Construct a rotation-free tensor aligned with the cardinal axes.
        # Since v1 is a cardinal axis here, D is diagonal with evals permuted.
        for iz in range(shape[2]):
            if iz < shape[2] // 2:
                # v1 = z -> lambda1 on axis 2
                D[:, :, iz] = np.diag([0.3e-3, 0.3e-3, 1.0e-3])
            else:
                D[:, :, iz] = np.diag([1.0e-3, 0.3e-3, 0.3e-3])
        adc = np.einsum('i,...ij,j->...', g, D, g)
        S[..., k] = S0 * np.exp(-b * adc)
    S += rng.normal(scale=2.0, size=S.shape).astype(np.float32)  # small noise
    if affine is None:
        affine = np.diag([2.0, 2.0, 2.0, 1.0])
    img = nib.Nifti1Image(S, affine)
    return img, gtab


def _save_with_v1(tmp_path, affine):
    img, gtab = _make_dwi(affine=affine)
    data = img.get_fdata().astype(np.float32)
    tenmodel = TensorModel(gtab)
    tenfit = tenmodel.fit(data)
    fa = np.array(tenfit.fa, dtype=np.float32)
    md = np.array(tenfit.md, dtype=np.float32)
    stem = "sub-test"
    out_dir = tmp_path / "tracking"
    out_dir.mkdir()
    # Minimal streamlines object (empty is fine; V1 path is independent).
    streamlines = []
    outputs = save_tracking_outputs(
        streamlines, img, fa, md, img.affine, out_dir, stem,
        tracking_params={}, verbose=False, tenfit=tenfit,
    )
    v1_path = out_dir / "scalar_maps" / f"{stem}_v1.nii.gz"
    fa_path = out_dir / "scalar_maps" / f"{stem}_fa.nii.gz"
    return outputs, v1_path, fa_path


class TestV1Product:
    def test_v1_written_with_vector_intent_and_shape(self, tmp_path):
        outputs, v1_path, _ = _save_with_v1(tmp_path, np.diag([2.0, 2.0, 2.0, 1.0]))
        assert v1_path.exists()
        assert "v1_map" in outputs
        img = nib.load(str(v1_path))
        # (X, Y, Z, 1, 3), float32
        assert img.shape[:3] == (8, 9, 7)
        assert img.shape[-1] == 3
        assert img.shape[-2] == 1
        # On-disk dtype is float32 (get_fdata() upcasts for reading; check the header).
        assert img.header.get_data_dtype() == np.float32
        # Vector intent declared in the header.
        intent = img.header.get_intent()
        assert intent[0] == 'vector'
        # Unit-length vectors where the fit succeeded.
        v1 = img.get_fdata()
        flat = v1.reshape(-1, 3)
        norms = np.linalg.norm(flat[flat.sum(axis=1) != 0], axis=1)
        assert np.allclose(norms, 1.0, atol=1e-5)

    def test_v1_affine_byte_identical_to_fa(self, tmp_path):
        _, v1_path, fa_path = _save_with_v1(tmp_path, np.diag([2.0, 2.0, 2.0, 1.0]))
        v1_aff = nib.load(str(v1_path)).affine
        fa_aff = nib.load(str(fa_path)).affine
        assert np.array_equal(v1_aff, fa_aff)

    def test_v1_sidecar_has_frame_keys(self, tmp_path):
        _, v1_path, _ = _save_with_v1(tmp_path, np.diag([2.0, 2.0, 2.0, 1.0]))
        sidecar = v1_path.with_suffix("")  # .nii.gz -> need .json
        sidecar = v1_path.parent / (v1_path.name.replace(".nii.gz", ".json"))
        assert sidecar.exists()
        data = json.loads(sidecar.read_text())
        assert data["VectorFrame"] == "world-RAS"
        assert "AffineDeterminant" in data
        assert "ObliquityRad" in data and len(data["ObliquityRad"]) == 3
        assert "ShearMagnitude" in data
        assert "ShearWarning" in data
        assert data["EigenvectorConvention"].startswith("dipy.reconst.dti")

    def test_v1_skipped_without_tenfit_with_warning(self, tmp_path, recwarn):
        img, _ = _make_dwi()
        from csttool.tracking.modules.save_tracking_outputs import save_tracking_outputs
        out_dir = tmp_path / "t2"
        out_dir.mkdir()
        outputs = save_tracking_outputs(
            [], img, np.zeros((8, 9, 7), np.float32),
            np.zeros((8, 9, 7), np.float32),
            img.affine, out_dir, "s", tracking_params={}, verbose=False, tenfit=None,
        )
        assert "v1_map" not in outputs
        assert any("V1" in str(w.message) for w in recwarn.list)

    def test_v1_world_frame_matches_polar_rotation(self, tmp_path):
        # For an RAS affine the rotation is identity, so the stored world V1
        # equals the voxel-frame V1 exactly (sanity check the storage path).
        outputs, v1_path, _ = _save_with_v1(tmp_path, np.diag([2.0, 2.0, 2.0, 1.0]))
        img, gtab = _make_dwi(affine=np.diag([2.0, 2.0, 2.0, 1.0]))
        tenfit = TensorModel(gtab).fit(img.get_fdata().astype(np.float32))
        v1_voxel = tenfit.evecs[..., :, 0]
        stored = nib.load(str(v1_path)).get_fdata().reshape(v1_voxel.shape)
        # Only compare voxels where the fit produced a non-zero eigenvector.
        nonzero = np.linalg.norm(v1_voxel.reshape(-1, 3), axis=1) > 0
        stored_flat = stored.reshape(-1, 3)
        voxel_flat = v1_voxel.reshape(-1, 3)
        # RAS -> R = I, so they match in direction (sign may differ from DIPY's
        # arbitrary eigenvector sign convention; compare magnitudes / abs dot).
        dots = np.abs((stored_flat[nonzero] * voxel_flat[nonzero]).sum(axis=1))
        assert np.allclose(dots, 1.0, atol=1e-4)
