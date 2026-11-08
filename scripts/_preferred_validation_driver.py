"""Direct validation driver for the preferred CST pipeline (no full re-run).

Calls run_preferred_extraction with the raw multi-shell DWI + an existing FA so
registration + PLIC seeding + CSA-peak tracking run without re-doing the whole
whole-brain track stage. Prints the gate quantities.
"""
import sys, json, argparse, time
from pathlib import Path
sys.path.insert(0, 'src')
from csttool.cli.commands.extract import run_preferred_extraction

SUB = sys.argv[1] if len(sys.argv) > 1 else "sub_005"
FA = Path(sys.argv[2]) if len(sys.argv) > 2 else Path(
    f"/home/alem/neurodata/out/20260922/wm-dam-calibration-v2/{SUB}-dam-0-5/"
    f"tracking/scalar_maps/{SUB}-dam-0-5_fa.nii.gz")
DWI = Path(f"/home/alem/neurodata/in/cdmd/{SUB}/dwi/{SUB}_dwi.nii.gz")
OUT = Path(f"/home/alem/neurodata/out/20260922/preferred-validation/{SUB}")

args = argparse.Namespace(
    fast_registration=True, dilate_brainstem=2, dilate_motor=1, rng_seed=42,
    seed_mask=None, model_mask=None, wm_pve=None, gm_pve=None, csf_pve=None,
    midline_tolerance_mm=8.0, seed_density=4, step_size=0.5,
    min_length=40.0, max_length=300.0, sh_order=6, npeaks=2, fit_method='WLS',
    save_visualizations=False,
)
t0 = time.time()
res = run_preferred_extraction(preproc_path=DWI, fa_path=FA, output_dir=OUT,
                               subject_id=f"{SUB}-preferred", args=args, verbose=True)
dt = time.time() - t0
if res is None:
    print("PREFERRED RUN RETURNED None (failed)"); sys.exit(1)
s = res['stats']
print("\n===== PREFERRED VALIDATION RESULT =====")
print(json.dumps({k: s[k] for k in (
    'cst_left_count', 'cst_right_count', 'cst_total_count',
    'left_motor_reach_frac', 'right_motor_reach_frac',
    'left_inferior_reach_frac', 'right_inferior_reach_frac',
    'laterality_index', 'left_seeds', 'right_seeds',
    'stopping_mode', 'seed_source', 'model_mask_source') if k in s}, indent=2))
print("left_diag:", json.dumps(s['left_diag']))
print("right_diag:", json.dumps(s['right_diag']))
print(f"elapsed {dt/60:.1f} min")
