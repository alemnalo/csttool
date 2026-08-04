"""Generate golden figure images for perceptual-hash visual regression.

Run manually (NOT by pytest) to (re)create the committed golden PNGs under
``tests/metrics/golden/`` from the deterministic fixtures used by
``test_report_layout.py::TestVisualRegressionGolden``. Regeneration is an
explicit, reviewed action; the tests skip when the goldens are absent.

    python tests/metrics/generate_golden.py
"""

from pathlib import Path

import sys
from pathlib import Path

import matplotlib
matplotlib.use("Agg")  # noqa: E402

# Make the test package importable when run as a script.
sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

# Reuse the deterministic fixtures defined alongside the tests.
from tests.metrics.test_report_layout import (
    make_comparison,
    make_fa_background,
    make_streamlines,
    RAS_AFFINE,
)
from csttool.metrics.modules.visualizations import (
    plot_profile_matrix,
    plot_tractogram_qc_triptych,
)


def main() -> None:
    out = Path(__file__).parent / "golden"
    out.mkdir(parents=True, exist_ok=True)

    comparison = make_comparison()
    pm = plot_profile_matrix(
        comparison["left"], comparison["right"], out, "_golden"
    )
    pm.replace(out / "profile_matrix.png")
    pm = out / "profile_matrix.png"

    fa = make_fa_background()
    sl_l = make_streamlines(20, 42)
    sl_r = make_streamlines(20, 7)
    tri = plot_tractogram_qc_triptych(
        sl_l, sl_r, fa, RAS_AFFINE, out, "_golden", background_kind="fa"
    )
    tri.replace(out / "tractogram_qc_triptych.png")
    tri = out / "tractogram_qc_triptych.png"

    # Remove the temporary subject-prefixed files the generators wrote.
    for f in out.glob("_golden_*"):
        f.unlink()

    print(f"goldens written: {pm}, {tri}")


if __name__ == "__main__":
    main()
