# `csttool.preprocess`

The preprocessing module exposes the high-level functions that the CLI [`preprocess`](../cli/preprocess.md) command orchestrates. Use it directly when you want to script preprocessing from Python.

```python
from csttool.preprocess import run_preprocessing

result = run_preprocessing(
    input_dir="./raw",          # directory holding sub-01.nii.gz + .bval/.bvec
    output_dir="./preproc",
    filename="sub-01",          # stem, without extension
    denoise_method="mppca",
    apply_motion_correction=True,
    external_correction="topup-eddy",   # declared, never verified
)

result["output_paths"]["data"]   # preprocessed NIfTI
result["bvecs_rotated"]          # True when motion correction rotated the gradients
```

::: csttool.preprocess
