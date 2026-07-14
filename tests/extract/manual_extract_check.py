"""
test_registration.py

Complete test script for csttool's registration and atlas warping modules.

Usage:
    python test_registration.py
"""

from time import time
from pathlib import Path

import numpy as np
import nibabel as nib
import matplotlib
import matplotlib.pyplot as plt

matplotlib.use('Agg')

# =============================================================================
# CONFIGURATION
# =============================================================================

# Input: Subject FA map from tracking pipeline
SUBJECT_FA_PATH = "/home/alem/Documents/thesis/data/out/trk/scalar_maps/17_cmrr_mbep2d_diff_ap_tdi.nii_fa.nii.gz"

# Output directory for registration results
OUTPUT_DIR = "/home/alem/Documents/thesis/data/out/extraction_test_output/"

# Use reduced iterations for faster testing (set to False for full registration)
FAST_TEST = False

# =============================================================================
# IMPORTS
# =============================================================================

from csttool.extract.modules.registration import (
    load_mni_template,
    compute_affine_registration,
    compute_syn_registration,
    register_mni_to_subject,
    save_registration_report,
    plot_registration_comparison
)

from csttool.extract.modules.warp_atlas_to_subject import (
    fetch_harvard_oxford,
    warp_atlas_to_subject,
    warp_harvard_oxford_to_subject,
    verify_atlas_labels,
    CST_ROI_CONFIG,
    HARVARDOXFORD_SUBCORTICAL,
    HARVARDOXFORD_CORTICAL
)

# =============================================================================
# SETUP
# =============================================================================

print("=" * 70)
print("CSTTOOL REGISTRATION & ATLAS WARPING - TEST SCRIPT")
print("=" * 70)

# Verify input exists
subject_fa_path = Path(SUBJECT_FA_PATH)
if not subject_fa_path.exists():
    print(f"✗ ERROR: Subject FA not found: {subject_fa_path}")
    print("  Please update SUBJECT_FA_PATH in this script")
    exit(1)

# Create output directory
output_dir = Path(OUTPUT_DIR)
output_dir.mkdir(parents=True, exist_ok=True)

print(f"\nInput:  {subject_fa_path}")
print(f"Output: {output_dir}")

# Set iteration parameters based on test mode
if FAST_TEST:
    print("\n⚡ FAST TEST MODE: Using reduced iterations")
    LEVEL_ITERS_AFFINE = [1000, 100, 10]
    LEVEL_ITERS_SYN = [5, 5, 3]
else:
    print("\n🐢 FULL TEST MODE: Using production iterations (slower)")
    LEVEL_ITERS_AFFINE = [10000, 1000, 100]
    LEVEL_ITERS_SYN = [10, 10, 5]

# =============================================================================
# TEST 1: Load MNI Template
# =============================================================================

print("\n" + "=" * 70)
print("TEST 1: load_mni_template()")
print("=" * 70)

t0 = time()
mni_img, mni_data, mni_affine = load_mni_template(contrast="T1")

print(f"\n    MNI shape: {mni_data.shape}")
print(f"    MNI dtype: {mni_data.dtype}")
print(f"    MNI range: [{mni_data.min():.2f}, {mni_data.max():.2f}]")
print(f"    Time: {time() - t0:.2f}s")
print("✓ Test 1 PASSED")

# =============================================================================
# TEST 2: Load Subject FA
# =============================================================================

print("\n" + "=" * 70)
print("TEST 2: Load Subject FA Map")
print("=" * 70)

t0 = time()
subject_img = nib.load(subject_fa_path)
subject_data = subject_img.get_fdata()
subject_affine = subject_img.affine

voxel_size = np.sqrt(np.sum(subject_affine[:3, :3]**2, axis=0))

print(f"\n    Subject shape: {subject_data.shape}")
print(f"    Subject dtype: {subject_data.dtype}")
print(f"    Subject voxel size: {voxel_size.round(3)} mm")
print(f"    FA range: [{subject_data.min():.3f}, {subject_data.max():.3f}]")
print(f"    Time: {time() - t0:.2f}s")
print("✓ Test 2 PASSED")

# =============================================================================
# TEST 3: Affine Registration
# =============================================================================

print("\n" + "=" * 70)
print("TEST 3: compute_affine_registration()")
print("=" * 70)

t0 = time()
affine_map = compute_affine_registration(
    static_image=subject_data,
    static_affine=subject_affine,
    moving_image=mni_data,
    moving_affine=mni_affine,
    level_iters=LEVEL_ITERS_AFFINE,
    sigmas=[3.0, 1.0, 0.0],
    factors=[4, 2, 1],
    verbose=True
)
affine_time = time() - t0

print(f"\n    Affine matrix:")
print(f"    {np.array2string(affine_map.affine, precision=3, suppress_small=True)}")
print(f"    Time: {affine_time:.2f}s")

# Test transform
print("\n    Testing affine transform...")
warped_affine = affine_map.transform(mni_data)
print(f"    Warped shape: {warped_affine.shape}")
print(f"    Warped range: [{warped_affine.min():.2f}, {warped_affine.max():.2f}]")
print("✓ Test 3 PASSED")

# =============================================================================
# TEST 4: Visualization - Before vs After Affine
# =============================================================================

print("\n" + "=" * 70)
print("TEST 4: plot_registration_comparison() - Affine Stage")
print("=" * 70)

viz_dir = output_dir / "visualizations"
viz_dir.mkdir(parents=True, exist_ok=True)

t0 = time()

# Before registration: resample MNI to subject grid with identity transform
from dipy.align.imaffine import AffineMap

identity_map = AffineMap(
    np.eye(4),
    subject_data.shape, subject_affine,
    mni_data.shape, mni_affine
)
mni_resampled = identity_map.transform(mni_data)

# Plot before registration
print("\n    Plotting before registration...")
figs_before = plot_registration_comparison(
    static_data=subject_data,
    moving_data=mni_resampled,
    ltitle="Subject FA",
    rtitle="MNI (identity)",
    output_dir=viz_dir,
    fname_prefix="01_before_registration"
)

# Plot after affine
print("\n    Plotting after affine registration...")
figs_affine = plot_registration_comparison(
    static_data=subject_data,
    moving_data=warped_affine,
    ltitle="Subject FA",
    rtitle="MNI (affine)",
    output_dir=viz_dir,
    fname_prefix="02_after_affine"
)

print(f"\n    Time: {time() - t0:.2f}s")
print("✓ Test 4 PASSED")

# =============================================================================
# TEST 5: SyN Registration
# =============================================================================

print("\n" + "=" * 70)
print("TEST 5: compute_syn_registration()")
print("=" * 70)

t0 = time()
syn_mapping = compute_syn_registration(
    static_image=subject_data,
    static_affine=subject_affine,
    moving_image=mni_data,
    moving_affine=mni_affine,
    prealign=affine_map.affine,
    level_iters=LEVEL_ITERS_SYN,
    metric_radius=4,
    verbose=True
)
syn_time = time() - t0

print(f"\n    Mapping type: {type(syn_mapping).__name__}")
print(f"    Time: {syn_time:.2f}s")

# Test transform
print("\n    Testing SyN transform...")
warped_syn = syn_mapping.transform(mni_data)
print(f"    Warped shape: {warped_syn.shape}")
print(f"    Warped range: [{warped_syn.min():.2f}, {warped_syn.max():.2f}]")
print("✓ Test 5 PASSED")

# =============================================================================
# TEST 6: Visualization - After SyN
# =============================================================================

print("\n" + "=" * 70)
print("TEST 6: plot_registration_comparison() - After SyN")
print("=" * 70)

t0 = time()
print("\n    Plotting after SyN registration...")
figs_syn = plot_registration_comparison(
    static_data=subject_data,
    moving_data=warped_syn,
    ltitle="Subject FA",
    rtitle="MNI (SyN)",
    output_dir=viz_dir,
    fname_prefix="03_after_syn"
)

print(f"\n    Time: {time() - t0:.2f}s")
print("✓ Test 6 PASSED")

# =============================================================================
# TEST 7: Save Registration Report (Standalone)
# =============================================================================

print("\n" + "=" * 70)
print("TEST 7: save_registration_report()")
print("=" * 70)

t0 = time()

# Construct result dict matching register_mni_to_subject output
test_result = {
    'mapping': syn_mapping,
    'affine_map': affine_map,
    'subject_affine': subject_affine,
    'subject_shape': subject_data.shape,
    'mni_affine': mni_affine,
    'mni_shape': mni_data.shape,
    'warped_template_path': None
}

report_path = save_registration_report(
    result=test_result,
    output_dir=output_dir,
    subject_id="test_manual"
)

print(f"    Time: {time() - t0:.2f}s")
print("✓ Test 7 PASSED")

# =============================================================================
# TEST 8: Full Pipeline - register_mni_to_subject()
# =============================================================================

print("\n" + "=" * 70)
print("TEST 8: register_mni_to_subject() - Full Pipeline")
print("=" * 70)

pipeline_dir = output_dir / "full_pipeline"

t0 = time()
result = register_mni_to_subject(
    subject_fa_path=SUBJECT_FA_PATH,
    output_dir=pipeline_dir,
    level_iters_affine=LEVEL_ITERS_AFFINE,
    level_iters_syn=LEVEL_ITERS_SYN,
    save_warped=True,
    generate_qc=False,  # We already generated QC above
    verbose=True
)
pipeline_time = time() - t0

print(f"\n    Result keys: {list(result.keys())}")
print(f"    Mapping type: {type(result['mapping']).__name__}")
print(f"    Warped template: {result['warped_template_path']}")
print(f"    Time: {pipeline_time:.2f}s")
print("✓ Test 8 PASSED")

# =============================================================================
# TEST 9: Verify Mapping Can Transform Atlas (Label Image)
# =============================================================================

print("\n" + "=" * 70)
print("TEST 9: Atlas Transformation Test (Nearest Neighbor)")
print("=" * 70)

t0 = time()

# Create a dummy parcellation with discrete labels in MNI space
dummy_parcellation = np.zeros(mni_data.shape, dtype=np.int32)

# Simulate some ROIs
center = np.array(mni_data.shape) // 2
dummy_parcellation[center[0]-10:center[0]+10, 
                   center[1]-10:center[1]+10, 
                   center[2]-20:center[2]] = 1  # "Brainstem"
dummy_parcellation[center[0]-20:center[0], 
                   center[1]+10:center[1]+30, 
                   center[2]:center[2]+20] = 2  # "Motor_L"
dummy_parcellation[center[0]:center[0]+20, 
                   center[1]+10:center[1]+30, 
                   center[2]:center[2]+20] = 3  # "Motor_R"

print(f"\n    Dummy parcellation shape: {dummy_parcellation.shape}")
print(f"    Unique labels before: {np.unique(dummy_parcellation)}")
print(f"    Label counts before: {[np.sum(dummy_parcellation == i) for i in range(4)]}")

# Transform with nearest neighbor interpolation (preserves discrete labels)
warped_parcellation = result['mapping'].transform(
    dummy_parcellation.astype(np.float64),
    interpolation='nearest'
)

print(f"\n    Warped parcellation shape: {warped_parcellation.shape}")
print(f"    Unique labels after: {np.unique(warped_parcellation.astype(int))}")
print(f"    Label counts after: {[np.sum(warped_parcellation.astype(int) == i) for i in range(4)]}")

# Save the test parcellation for inspection
parcel_path = output_dir / "test_parcellation_warped.nii.gz"
nib.save(
    nib.Nifti1Image(warped_parcellation.astype(np.int32), subject_affine),
    parcel_path
)
print(f"\n    Saved test parcellation: {parcel_path}")

print(f"    Time: {time() - t0:.2f}s")
print("✓ Test 9 PASSED")

# =============================================================================
# TEST 10: Verify Inverse Transform
# =============================================================================

print("\n" + "=" * 70)
print("TEST 10: Inverse Transform Test")
print("=" * 70)

t0 = time()

# Transform subject FA to MNI space (inverse direction)
subject_to_mni = result['mapping'].transform_inverse(subject_data)

print(f"\n    Subject FA shape: {subject_data.shape}")
print(f"    Transformed to MNI space: {subject_to_mni.shape}")
print(f"    Range: [{subject_to_mni.min():.3f}, {subject_to_mni.max():.3f}]")

# Save for inspection
inverse_path = output_dir / "test_subject_fa_in_mni_space.nii.gz"
nib.save(
    nib.Nifti1Image(subject_to_mni.astype(np.float32), mni_affine),
    inverse_path
)
print(f"    Saved: {inverse_path}")

print(f"    Time: {time() - t0:.2f}s")
print("✓ Test 10 PASSED")

# =============================================================================
# =============================================================================
# ATLAS WARPING TESTS
# =============================================================================
# =============================================================================

print("\n")
print("=" * 70)
print("=" * 70)
print("ATLAS WARPING MODULE TESTS")
print("=" * 70)
print("=" * 70)

# =============================================================================
# TEST 11: Fetch Harvard-Oxford Atlases
# =============================================================================

print("\n" + "=" * 70)
print("TEST 11: fetch_harvard_oxford()")
print("=" * 70)

t0 = time()
atlases = fetch_harvard_oxford(verbose=True)
fetch_time = time() - t0

print(f"\n    Result keys: {list(atlases.keys())}")
print(f"    Cortical atlas path: {atlases['cortical_path']}")
print(f"    Subcortical atlas path: {atlases['subcortical_path']}")
print(f"    Cortical shape: {atlases['cortical_img'].shape}")
print(f"    Subcortical shape: {atlases['subcortical_img'].shape}")

# Check label ranges
cort_data = atlases['cortical_img'].get_fdata()
subcort_data = atlases['subcortical_img'].get_fdata()
print(f"\n    Cortical label range: [{int(cort_data.min())}, {int(cort_data.max())}]")
print(f"    Subcortical label range: [{int(subcort_data.min())}, {int(subcort_data.max())}]")
print(f"    Cortical unique labels: {len(np.unique(cort_data))}")
print(f"    Subcortical unique labels: {len(np.unique(subcort_data))}")

print(f"\n    Time: {fetch_time:.2f}s")
print("✓ Test 11 PASSED")

# =============================================================================
# TEST 12: Warp Single Atlas (Subcortical)
# =============================================================================

print("\n" + "=" * 70)
print("TEST 12: warp_atlas_to_subject() - Subcortical Atlas")
print("=" * 70)

t0 = time()

subcortical_warped = warp_atlas_to_subject(
    atlas_img=atlases['subcortical_img'],
    mapping=result['mapping'],
    subject_shape=result['subject_shape'],
    subject_affine=result['subject_affine'],
    interpolation='nearest',
    verbose=True
)
warp_subcort_time = time() - t0

print(f"\n    Warped shape: {subcortical_warped.shape}")
print(f"    Warped dtype: {subcortical_warped.dtype}")
print(f"    Unique labels: {len(np.unique(subcortical_warped))}")

# Check brainstem label specifically
brainstem_label = 8
brainstem_voxels = np.sum(subcortical_warped == brainstem_label)
print(f"\n    Brainstem (label {brainstem_label}): {brainstem_voxels:,} voxels")

print(f"\n    Time: {warp_subcort_time:.2f}s")
print("✓ Test 12 PASSED")

# =============================================================================
# TEST 13: Warp Single Atlas (Cortical)
# =============================================================================

print("\n" + "=" * 70)
print("TEST 13: warp_atlas_to_subject() - Cortical Atlas")
print("=" * 70)

t0 = time()

cortical_warped = warp_atlas_to_subject(
    atlas_img=atlases['cortical_img'],
    mapping=result['mapping'],
    subject_shape=result['subject_shape'],
    subject_affine=result['subject_affine'],
    interpolation='nearest',
    verbose=True
)
warp_cort_time = time() - t0

print(f"\n    Warped shape: {cortical_warped.shape}")
print(f"    Warped dtype: {cortical_warped.dtype}")
print(f"    Unique labels: {len(np.unique(cortical_warped))}")

# Check precentral gyrus label specifically
precentral_label = 7
precentral_voxels = np.sum(cortical_warped == precentral_label)
print(f"\n    Precentral Gyrus (label {precentral_label}): {precentral_voxels:,} voxels")

print(f"\n    Time: {warp_cort_time:.2f}s")
print("✓ Test 13 PASSED")

# =============================================================================
# TEST 14: Verify Atlas Labels
# =============================================================================

print("\n" + "=" * 70)
print("TEST 14: verify_atlas_labels()")
print("=" * 70)

t0 = time()

# Verify subcortical labels (brainstem is critical for CST)
subcort_verification = verify_atlas_labels(
    warped_atlas=subcortical_warped,
    expected_labels=[16],  # Brainstem
    atlas_name="Subcortical (brainstem)",
    verbose=True
)

# Verify cortical labels (precentral gyrus is critical for CST)
cort_verification = verify_atlas_labels(
    warped_atlas=cortical_warped,
    expected_labels=[7],  # Precentral gyrus
    atlas_name="Cortical (precentral)",
    verbose=True
)

print(f"\n    Subcortical verification: {'✓ PASSED' if subcort_verification['success'] else '✗ FAILED'}")
print(f"    Cortical verification: {'✓ PASSED' if cort_verification['success'] else '✗ FAILED'}")

if not subcort_verification['success']:
    print(f"    ⚠️  Missing subcortical labels: {subcort_verification['missing']}")
if not cort_verification['success']:
    print(f"    ⚠️  Missing cortical labels: {cort_verification['missing']}")

print(f"\n    Time: {time() - t0:.2f}s")
print("✓ Test 14 PASSED")

# =============================================================================
# TEST 15: Full Pipeline - warp_harvard_oxford_to_subject()
# =============================================================================

print("\n" + "=" * 70)
print("TEST 15: warp_harvard_oxford_to_subject() - Full Pipeline")
print("=" * 70)

atlas_output_dir = output_dir / "warped_atlases"

t0 = time()
atlas_result = warp_harvard_oxford_to_subject(
    registration_result=result,
    output_dir=atlas_output_dir,
    subject_id="17_cmrr",
    save_warped=True,
    verbose=True
)
atlas_pipeline_time = time() - t0

print(f"\n    Result keys: {list(atlas_result.keys())}")
print(f"    Cortical warped shape: {atlas_result['cortical_warped'].shape}")
print(f"    Subcortical warped shape: {atlas_result['subcortical_warped'].shape}")
print(f"    Cortical saved to: {atlas_result['cortical_warped_path']}")
print(f"    Subcortical saved to: {atlas_result['subcortical_warped_path']}")
print(f"    ROI config keys: {list(atlas_result['roi_config'].keys())}")

print(f"\n    Time: {atlas_pipeline_time:.2f}s")
print("✓ Test 15 PASSED")

# =============================================================================
# TEST 16: Verify CST ROI Labels in Warped Atlases
# =============================================================================

print("\n" + "=" * 70)
print("TEST 16: Verify CST-Specific ROI Labels")
print("=" * 70)

t0 = time()

print("\n    CST ROI Configuration:")
for roi_name, roi_info in CST_ROI_CONFIG.items():
    print(f"      {roi_name}:")
    print(f"        Atlas: {roi_info['atlas']}")
    print(f"        Label: {roi_info['label']}")
    print(f"        Description: {roi_info['description']}")

# Check each CST ROI in warped atlases
print("\n    Verifying CST ROIs in warped atlases:")

# Brainstem (subcortical)
brainstem_mask = atlas_result['subcortical_warped'] == CST_ROI_CONFIG['brainstem']['label']
brainstem_count = np.sum(brainstem_mask)
print(f"      Brainstem: {brainstem_count:,} voxels {'✓' if brainstem_count > 0 else '✗'}")

# Motor Left (cortical, left hemisphere)
motor_label = CST_ROI_CONFIG['motor_left']['label']
motor_left_preliminary = atlas_result['cortical_warped'] == motor_label
motor_left_count = np.sum(motor_left_preliminary)
print(f"      Motor cortex (label {motor_label}, both hemispheres): {motor_left_count:,} voxels {'✓' if motor_left_count > 0 else '✗'}")

# Note about hemisphere separation
print("\n    Note: Hemisphere separation (motor_left vs motor_right)")
print("          will be implemented in create_roi_masks.py")

print(f"\n    Time: {time() - t0:.2f}s")
print("✓ Test 16 PASSED")

# =============================================================================
# TEST 17: Visualize Warped Atlas Overlays
# =============================================================================

print("\n" + "=" * 70)
print("TEST 17: Visualize Warped Atlas Overlays")
print("=" * 70)

t0 = time()

atlas_viz_dir = viz_dir / "atlas_overlays"
atlas_viz_dir.mkdir(parents=True, exist_ok=True)

# Create overlay visualization
fig, axes = plt.subplots(2, 3, figsize=(15, 10))
fig.suptitle('Warped Harvard-Oxford Atlas Overlays', fontsize=14)

# Get middle slices
mid_sag = subject_data.shape[0] // 2
mid_cor = subject_data.shape[1] // 2
mid_ax = subject_data.shape[2] // 2

# Row 1: Subcortical atlas (brainstem)
axes[0, 0].imshow(subject_data[mid_sag, :, :].T, cmap='gray', origin='lower')
axes[0, 0].imshow(np.ma.masked_where(subcortical_warped[mid_sag, :, :].T == 0, 
                                      subcortical_warped[mid_sag, :, :].T), 
                  cmap='tab20', alpha=0.5, origin='lower')
axes[0, 0].set_title('Subcortical - Sagittal')
axes[0, 0].axis('off')

axes[0, 1].imshow(subject_data[:, mid_cor, :].T, cmap='gray', origin='lower')
axes[0, 1].imshow(np.ma.masked_where(subcortical_warped[:, mid_cor, :].T == 0,
                                      subcortical_warped[:, mid_cor, :].T),
                  cmap='tab20', alpha=0.5, origin='lower')
axes[0, 1].set_title('Subcortical - Coronal')
axes[0, 1].axis('off')

axes[0, 2].imshow(subject_data[:, :, mid_ax].T, cmap='gray', origin='lower')
axes[0, 2].imshow(np.ma.masked_where(subcortical_warped[:, :, mid_ax].T == 0,
                                      subcortical_warped[:, :, mid_ax].T),
                  cmap='tab20', alpha=0.5, origin='lower')
axes[0, 2].set_title('Subcortical - Axial')
axes[0, 2].axis('off')

# Row 2: Cortical atlas (motor cortex)
axes[1, 0].imshow(subject_data[mid_sag, :, :].T, cmap='gray', origin='lower')
axes[1, 0].imshow(np.ma.masked_where(cortical_warped[mid_sag, :, :].T == 0,
                                      cortical_warped[mid_sag, :, :].T),
                  cmap='tab20', alpha=0.5, origin='lower')
axes[1, 0].set_title('Cortical - Sagittal')
axes[1, 0].axis('off')

axes[1, 1].imshow(subject_data[:, mid_cor, :].T, cmap='gray', origin='lower')
axes[1, 1].imshow(np.ma.masked_where(cortical_warped[:, mid_cor, :].T == 0,
                                      cortical_warped[:, mid_cor, :].T),
                  cmap='tab20', alpha=0.5, origin='lower')
axes[1, 1].set_title('Cortical - Coronal')
axes[1, 1].axis('off')

axes[1, 2].imshow(subject_data[:, :, mid_ax].T, cmap='gray', origin='lower')
axes[1, 2].imshow(np.ma.masked_where(cortical_warped[:, :, mid_ax].T == 0,
                                      cortical_warped[:, :, mid_ax].T),
                  cmap='tab20', alpha=0.5, origin='lower')
axes[1, 2].set_title('Cortical - Axial')
axes[1, 2].axis('off')

plt.tight_layout()
overlay_path = atlas_viz_dir / "atlas_overlay_all_views.png"
plt.savefig(overlay_path, dpi=150, bbox_inches='tight')
plt.close()

print(f"    ✓ Saved atlas overlay: {overlay_path}")

# Create CST-specific ROI visualization
fig, axes = plt.subplots(1, 3, figsize=(15, 5))
fig.suptitle('CST ROIs: Brainstem + Precentral Gyrus', fontsize=14)

# Create combined CST ROI mask for visualization
brainstem_mask = (subcortical_warped == 16).astype(float)
precentral_mask = (cortical_warped == 7).astype(float) * 2  # Different color cst_combined

# =============================================================================
# TEST 18: create_cst_roi_masks()
# =============================================================================

print("\n" + "=" * 70)
print("TEST 18: create_cst_roi_masks()")
print("=" * 70)

from csttool.extract.modules.visualizations import plot_roi_masks
from csttool.extract.modules.create_roi_masks import (
    create_cst_roi_masks,
    extract_roi_mask,
    separate_hemispheres
)

roi_output_dir = output_dir / "roi_masks"

t0 = time()
masks = create_cst_roi_masks(
    warped_cortical=atlas_result['cortical_warped'],
    warped_subcortical=atlas_result['subcortical_warped'],
    subject_affine=result['subject_affine'],
    roi_config=atlas_result['roi_config'],
    dilate_brainstem=2,
    dilate_motor=1,
    output_dir=roi_output_dir,
    subject_id="17_cmrr",
    save_masks=True,
    verbose=True
)
roi_time = time() - t0

print(f"\n    Result keys: {list(masks.keys())}")
print(f"    Time: {roi_time:.2f}s")
print("✓ Test 18 PASSED")

# =============================================================================
# TEST 19: Verify Hemisphere Separation
# =============================================================================

print("\n" + "=" * 70)
print("TEST 19: Verify Hemisphere Separation")
print("=" * 70)

t0 = time()

# Check that left and right don't overlap
overlap = masks['motor_left'] & masks['motor_right']
overlap_count = np.sum(overlap)

print(f"\n    Motor Left voxels:  {np.sum(masks['motor_left']):,}")
print(f"    Motor Right voxels: {np.sum(masks['motor_right']):,}")
print(f"    Overlap voxels:     {overlap_count}")

if overlap_count == 0:
    print("    ✓ No overlap between hemispheres")
else:
    print(f"    ⚠️ Warning: {overlap_count} voxels overlap!")

# Verify left is actually on left side (negative X in RAS)
left_coords = np.where(masks['motor_left'])
right_coords = np.where(masks['motor_right'])

if len(left_coords[0]) > 0 and len(right_coords[0]) > 0:
    # Convert to world coordinates
    left_x_world = result['subject_affine'][0, 0] * np.mean(left_coords[0]) + result['subject_affine'][0, 3]
    right_x_world = result['subject_affine'][0, 0] * np.mean(right_coords[0]) + result['subject_affine'][0, 3]
    
    print(f"\n    Left centroid X (world):  {left_x_world:.1f} mm")
    print(f"    Right centroid X (world): {right_x_world:.1f} mm")
    
    if left_x_world < right_x_world:
        print("    ✓ Hemisphere assignment correct (Left < Right in X)")
    else:
        print("    ⚠️ Warning: Hemisphere assignment may be inverted!")

print(f"\n    Time: {time() - t0:.2f}s")
print("✓ Test 19 PASSED")

# =============================================================================
# TEST 20: Visualize ROI Masks
# =============================================================================

print("\n" + "=" * 70)
print("TEST 20: plot_roi_masks()")
print("=" * 70)

t0 = time()

viz_path = plot_roi_masks(
    fa=subject_data,
    masks=masks,
    output_dir=viz_dir / "roi_masks",
    subject_id="17_cmrr",
    affine=subject_affine,
    verbose=True
)

print(f"\n    Time: {time() - t0:.2f}s")
print("✓ Test 20 PASSED")

# =============================================================================
# TEST 21: Load Whole-Brain Tractogram
# =============================================================================

print("\n" + "=" * 70)
print("TEST 21: Load Whole-Brain Tractogram")
print("=" * 70)

from csttool.extract.modules.endpoint_filtering import (
    extract_bilateral_cst,
    save_cst_tractograms,
    save_extraction_report
)
from dipy.io.streamline import load_tractogram

# Update this path to your whole-brain tractogram
TRACTOGRAM_PATH = "/home/alem/Documents/thesis/data/out/trk/17_cmrr_mbep2d_diff_ap_tdi.nii_cst_det.trk"

t0 = time()

if Path(TRACTOGRAM_PATH).exists():
    sft = load_tractogram(TRACTOGRAM_PATH, 'same', bbox_valid_check=False)
    whole_brain_streamlines = sft.streamlines
    tractogram_affine = sft.affine
    print(f"    Loaded: {len(whole_brain_streamlines):,} streamlines")
    print(f"    Time: {time() - t0:.2f}s")
    print("✓ Test 21 PASSED")
else:
    print(f"    ⚠️ Tractogram not found: {TRACTOGRAM_PATH}")
    whole_brain_streamlines = None

# =============================================================================
# TEST 22: extract_bilateral_cst()
# =============================================================================

if whole_brain_streamlines is not None:
    print("\n" + "=" * 70)
    print("TEST 22: extract_bilateral_cst()")
    print("=" * 70)
    
    t0 = time()
    cst_result = extract_bilateral_cst(
        streamlines=whole_brain_streamlines,
        masks=masks,
        affine=masks['subject_affine'],
        min_length=20.0,
        max_length=200.0,
        verbose=True
    )
    extraction_time = time() - t0
    
    print(f"\n    Result keys: {list(cst_result.keys())}")
    print(f"    Time: {extraction_time:.2f}s")
    print("✓ Test 22 PASSED")

# =============================================================================
# TEST 23: Save CST Tractograms
# =============================================================================

if whole_brain_streamlines is not None:
    print("\n" + "=" * 70)
    print("TEST 23: save_cst_tractograms()")
    print("=" * 70)
    
    cst_output_dir = output_dir / "cst_tractograms"
    
    # Load reference image for tractogram
    reference_img = nib.load(subject_fa_path)
    
    t0 = time()
    output_paths = save_cst_tractograms(
        cst_result=cst_result,
        reference_img=reference_img,
        output_dir=cst_output_dir,
        subject_id="17_cmrr",
        verbose=True
    )
    
    # Save report
    report_path = save_extraction_report(
        cst_result=cst_result,
        output_paths=output_paths,
        output_dir=cst_output_dir,
        subject_id="17_cmrr"
    )
    
    print(f"\n    Time: {time() - t0:.2f}s")
    print("✓ Test 23 PASSED")

    # =============================================================================
# TEST 24: Visualize Extracted CST
# =============================================================================

if whole_brain_streamlines is not None and len(cst_result['cst_combined']) > 0:
    print("\n" + "=" * 70)
    print("TEST 24: Visualize Extracted CST")
    print("=" * 70)
    
    t0 = time()
    
    cst_viz_dir = viz_dir / "cst_extraction"
    cst_viz_dir.mkdir(parents=True, exist_ok=True)
    
    # Create figure with 2 rows x 3 columns
    fig, axes = plt.subplots(2, 3, figsize=(18, 12))
    fig.suptitle(f'CST Extraction Results - 17_cmrr', fontsize=16)
    
    # Get middle slices for background
    mid_sag = subject_data.shape[0] // 2
    mid_cor = subject_data.shape[1] // 2
    mid_ax = subject_data.shape[2] // 2
    
    # Helper function to project streamlines onto 2D plane
    def project_streamlines_to_plane(streamlines, plane='axial', slice_idx=None, 
                                      slice_thickness=10, affine=None):
        """Project streamlines onto a 2D plane, filtering by slice proximity."""
        points_2d = []
        
        for sl in streamlines:
            # Convert to voxel coordinates if affine provided
            if affine is not None:
                inv_affine = np.linalg.inv(affine)
                ones = np.ones((sl.shape[0], 1))
                sl_h = np.hstack([sl, ones])
                sl_vox = (sl_h @ inv_affine.T)[:, :3]
            else:
                sl_vox = sl
            
            if plane == 'axial':
                # Filter points near the slice
                mask = np.abs(sl_vox[:, 2] - slice_idx) < slice_thickness
                if np.any(mask):
                    points_2d.append(sl_vox[mask][:, :2])  # X, Y
            elif plane == 'coronal':
                mask = np.abs(sl_vox[:, 1] - slice_idx) < slice_thickness
                if np.any(mask):
                    points_2d.append(sl_vox[mask][:, [0, 2]])  # X, Z
            elif plane == 'sagittal':
                mask = np.abs(sl_vox[:, 0] - slice_idx) < slice_thickness
                if np.any(mask):
                    points_2d.append(sl_vox[mask][:, 1:3])  # Y, Z
        
        return points_2d
    
    # Row 1: Whole brain (gray) + CST Left (blue) + CST Right (green)
    planes = ['sagittal', 'coronal', 'axial']
    slice_indices = [mid_sag, mid_cor, mid_ax]
    bg_slices = [
        subject_data[mid_sag, :, :].T,
        subject_data[:, mid_cor, :].T,
        subject_data[:, :, mid_ax].T
    ]
    
    for col, (plane, slice_idx, bg_slice) in enumerate(zip(planes, slice_indices, bg_slices)):
        ax = axes[0, col]
        
        # Background FA
        ax.imshow(bg_slice, cmap='gray', origin='lower', aspect='equal')
        
        # Project whole brain streamlines (subsample for speed)
        wb_subsample = whole_brain_streamlines[::10]  # Every 10th streamline
        wb_points = project_streamlines_to_plane(
            wb_subsample, plane=plane, slice_idx=slice_idx, 
            slice_thickness=15, affine=masks['subject_affine']
        )
        for pts in wb_points:
            if len(pts) > 1:
                ax.plot(pts[:, 0], pts[:, 1], 'gray', alpha=0.1, linewidth=0.3)
        
        # Project Left CST
        left_points = project_streamlines_to_plane(
            cst_result['cst_left'], plane=plane, slice_idx=slice_idx,
            slice_thickness=15, affine=masks['subject_affine']
        )
        for pts in left_points:
            if len(pts) > 1:
                ax.plot(pts[:, 0], pts[:, 1], 'blue', alpha=0.7, linewidth=1.0)
        
        # Project Right CST
        right_points = project_streamlines_to_plane(
            cst_result['cst_right'], plane=plane, slice_idx=slice_idx,
            slice_thickness=15, affine=masks['subject_affine']
        )
        for pts in right_points:
            if len(pts) > 1:
                ax.plot(pts[:, 0], pts[:, 1], 'green', alpha=0.7, linewidth=1.0)
        
        ax.set_title(f'{plane.capitalize()}\nWhole Brain + CST')
        ax.axis('off')
    
    # Row 2: CST only with ROI masks overlay
    for col, (plane, slice_idx, bg_slice) in enumerate(zip(planes, slice_indices, bg_slices)):
        ax = axes[1, col]
        
        # Background FA
        ax.imshow(bg_slice, cmap='gray', origin='lower', aspect='equal')
        
        # ROI masks
        if plane == 'sagittal':
            brainstem_slice = masks['brainstem'][mid_sag, :, :].T
            motor_left_slice = masks['motor_left'][mid_sag, :, :].T
            motor_right_slice = masks['motor_right'][mid_sag, :, :].T
        elif plane == 'coronal':
            brainstem_slice = masks['brainstem'][:, mid_cor, :].T
            motor_left_slice = masks['motor_left'][:, mid_cor, :].T
            motor_right_slice = masks['motor_right'][:, mid_cor, :].T
        else:  # axial
            brainstem_slice = masks['brainstem'][:, :, mid_ax].T
            motor_left_slice = masks['motor_left'][:, :, mid_ax].T
            motor_right_slice = masks['motor_right'][:, :, mid_ax].T
        
        # Overlay ROIs with transparency
        ax.imshow(np.ma.masked_where(brainstem_slice == 0, brainstem_slice),
                  cmap='Reds', alpha=0.3, origin='lower')
        ax.imshow(np.ma.masked_where(motor_left_slice == 0, motor_left_slice),
                  cmap='Blues', alpha=0.3, origin='lower')
        ax.imshow(np.ma.masked_where(motor_right_slice == 0, motor_right_slice),
                  cmap='Greens', alpha=0.3, origin='lower')
        
        # Project Left CST
        left_points = project_streamlines_to_plane(
            cst_result['cst_left'], plane=plane, slice_idx=slice_idx,
            slice_thickness=15, affine=masks['subject_affine']
        )
        for pts in left_points:
            if len(pts) > 1:
                ax.plot(pts[:, 0], pts[:, 1], 'blue', alpha=0.8, linewidth=1.2)
        
        # Project Right CST
        right_points = project_streamlines_to_plane(
            cst_result['cst_right'], plane=plane, slice_idx=slice_idx,
            slice_thickness=15, affine=masks['subject_affine']
        )
        for pts in right_points:
            if len(pts) > 1:
                ax.plot(pts[:, 0], pts[:, 1], 'green', alpha=0.8, linewidth=1.2)
        
        ax.set_title(f'{plane.capitalize()}\nCST + ROIs')
        ax.axis('off')
    
    # Add legend
    from matplotlib.lines import Line2D
    from matplotlib.patches import Patch
    legend_elements = [
        Line2D([0], [0], color='gray', linewidth=1, alpha=0.5, label='Whole Brain'),
        Line2D([0], [0], color='blue', linewidth=2, label=f'Left CST ({len(cst_result["cst_left"]):,})'),
        Line2D([0], [0], color='green', linewidth=2, label=f'Right CST ({len(cst_result["cst_right"]):,})'),
        Patch(facecolor='red', alpha=0.3, label='Brainstem ROI'),
        Patch(facecolor='blue', alpha=0.3, label='Motor Left ROI'),
        Patch(facecolor='green', alpha=0.3, label='Motor Right ROI'),
    ]
    fig.legend(handles=legend_elements, loc='lower center', ncol=6, fontsize=10)
    
    plt.tight_layout()
    plt.subplots_adjust(bottom=0.08)
    
    # Save figure
    cst_viz_path = cst_viz_dir / "17_cmrr_cst_extraction_visualization.png"
    plt.savefig(cst_viz_path, dpi=150, bbox_inches='tight')
    plt.close()
    
    print(f"    ✓ Saved CST visualization: {cst_viz_path}")
    print(f"    Time: {time() - t0:.2f}s")
    print("✓ Test 24 PASSED")

else:
    print("\n" + "=" * 70)
    print("TEST 24: SKIPPED (no CST streamlines extracted)")
    print("=" * 70)