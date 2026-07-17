"""
Shared fixtures for reproducibility tests.
"""

import pytest
import numpy as np
from pathlib import Path

from dipy.data import get_fnames
from dipy.io.image import load_nifti
from dipy.io.gradients import read_bvals_bvecs
from dipy.core.gradients import gradient_table
from dipy.reconst.dti import TensorModel
from dipy.tracking.streamline import Streamlines
from dipy.io.stateful_tractogram import StatefulTractogram, Space
import nibabel as nib

from csttool.reproducibility import RunContext
from csttool.tracking.modules import (
    fit_tensors,
    estimate_directions,
    seed_and_stop,
    run_tractography,
)


@pytest.fixture
def tracking_config():
    """Standard tracking parameters for reproducibility tests."""
    return {
        "step_size": 0.5,
        "fa_threshold": 0.2,
        "seed_density": 1,  # Low density for faster tests
        "sh_order": 4,  # Lower SH order for faster tests
    }


@pytest.fixture
def synthetic_dwi_data(tmp_path):
    """Load synthetic DWI data from DIPY for testing.

    Returns dict with data, affine, bvals, bvecs, gtab, fa_map.
    """
    # Use DIPY's small test dataset
    fraw, fbval, fbvec = get_fnames('small_64D')

    data, affine = load_nifti(fraw)
    bvals, bvecs = read_bvals_bvecs(fbval, fbvec)
    gtab = gradient_table(bvals, bvecs)

    # Fit tensor model to get FA
    tenmodel = TensorModel(gtab, fit_method='WLS')
    tenfit = tenmodel.fit(data)
    fa = tenfit.fa

    return {
        "data": data,
        "affine": affine,
        "bvals": bvals,
        "bvecs": bvecs,
        "gtab": gtab,
        "fa": fa,
    }


@pytest.fixture(scope="module")
def repeated_run_tractograms(tmp_path_factory):
    """Generate 3 tractograms from identical tracking runs.

    This fixture runs tracking 3 times with the same seed (42) and
    returns the tractograms for comparison.

    Note: Uses module scope to avoid recomputing on every test.
    """
    tmp_dir = tmp_path_factory.mktemp("repeated_runs")

    # Load test data once
    fraw, fbval, fbvec = get_fnames('small_64D')
    data, affine = load_nifti(fraw)
    bvals, bvecs = read_bvals_bvecs(fbval, fbvec)
    gtab = gradient_table(bvals, bvecs)

    # Create simple brain mask (all non-zero voxels)
    brain_mask = (data[..., 0] > 0).astype(bool)

    # Fit tensors once
    tenfit, fa, md, rd, ad, white_matter = fit_tensors(data, gtab, brain_mask, fit_method='WLS')

    # Estimate directions once
    csapeaks = estimate_directions(
        data,
        gtab,
        white_matter,
        sh_order=4,  # Low for speed
        verbose=False
    )

    # Generate seeds and stopping criterion once
    seeds, stopping_criterion = seed_and_stop(
        fa,
        affine,
        fa_thresh=0.2,
        density=1,  # Low density for speed
        verbose=False
    )

    # Run tracking 3 times with same seed
    tractograms = []
    for run_num in range(3):
        ctx = RunContext(run_seed=42)  # Same seed each time

        streamlines = run_tractography(
            csapeaks,
            stopping_criterion,
            seeds,
            affine,
            step_size=0.5,
            random_seed=ctx.rng_tracking_seed(),
            verbose=False,
            visualize=False
        )

        # Create tractogram
        img = nib.Nifti1Image(data, affine)
        sft = StatefulTractogram(streamlines, img, Space.RASMM)

        tractograms.append({
            "streamlines": streamlines,
            "sft": sft,
            "fa": fa,
            "md": md,
            "rd": rd,
            "ad": ad,
            "affine": affine,
            "run_num": run_num,
        })

    return tractograms


@pytest.fixture
def tractogram_artifact(tmp_path, synthetic_dwi_data):
    """Single fixed tractogram + scalar maps for sensitivity tests.

    This fixture runs tracking once and saves the result.
    Sensitivity tests can load this artifact instead of retracking,
    isolating the perturbation effect from pipeline variability.
    """
    dwi = synthetic_dwi_data

    # Create simple brain mask
    brain_mask = (dwi["data"][..., 0] > 0).astype(bool)

    # Fit tensors
    tenfit, fa, md, rd, ad, white_matter = fit_tensors(dwi["data"], dwi["gtab"], brain_mask, fit_method='WLS')

    # Estimate directions
    csapeaks = estimate_directions(
        dwi["data"],
        dwi["gtab"],
        white_matter,
        sh_order=4,
        verbose=False
    )

    # Generate seeds and stopping criterion
    seeds, stopping_criterion = seed_and_stop(
        fa,
        dwi["affine"],
        fa_thresh=0.2,
        density=1,
        verbose=False
    )

    # Run tracking once with fixed seed
    ctx = RunContext(run_seed=42)
    streamlines = run_tractography(
        csapeaks,
        stopping_criterion,
        seeds,
        dwi["affine"],
        step_size=0.5,
        random_seed=ctx.rng_tracking_seed(),
        verbose=False,
        visualize=False
    )

    # Save artifact
    artifact_dir = tmp_path / "artifacts"
    artifact_dir.mkdir(exist_ok=True)

    # Save tractogram
    img = nib.Nifti1Image(dwi["data"], dwi["affine"])
    sft = StatefulTractogram(streamlines, img, Space.RASMM)
    trk_path = artifact_dir / "tractogram.trk"
    from dipy.io.streamline import save_tractogram
    save_tractogram(sft, str(trk_path))

    # Save scalar maps
    fa_path = artifact_dir / "fa.nii.gz"
    md_path = artifact_dir / "md.nii.gz"
    nib.save(nib.Nifti1Image(fa, dwi["affine"]), fa_path)
    nib.save(nib.Nifti1Image(md, dwi["affine"]), md_path)

    return {
        "tractogram_path": trk_path,
        "fa_path": fa_path,
        "md_path": md_path,
        "streamlines": streamlines,
        "fa": fa,
        "md": md,
        "affine": dwi["affine"],
    }
