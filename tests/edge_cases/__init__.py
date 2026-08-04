"""
Formal edge-case suite (AU31).

Three independent audits (DeepSeek §4.5, GLM item 8, Qwen item 10) found no
systematic tests for pathological inputs: zero streamlines, all-zero DWI,
single direction, misordered bvec/bval, truncated NIfTI, DICOM missing tags.
The handling lived only in scattered try/except blocks.

This package provides characterisation tests that pin the *current* behaviour
for each class, plus targeted hardening (clearer error messages / warnings on
degenerate-but-valid inputs) implemented alongside in the relevant modules.

Per the roadmap meta-lesson, every "current behaviour is X" claim was verified
against running code before the test was written, and each test that claims to
guard something is checked to fail against the unhardened code where applicable
(the AU9/AU29 lesson: a test that passes vacuously against broken code is worse
than no test).
"""
