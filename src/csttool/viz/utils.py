"""
utils.py - general visualization helpers.

Utilities that are neither style policy (``viz.style``) nor spatial geometry
(``viz.geometry``): deterministic subsampling and small conveniences.
"""

import numpy as np

from ..reproducibility.context import DEFAULT_SEED, derive_seed

# Documented visualization seed, derived from the project default seed by the same
# ``derive_seed`` call ``RunContext.rng_viz`` uses, so the two agree by construction.
# Keeps figure sampling reproducible and independent of tracking randomness, without
# mutating global state.
#
# This was previously ``hash(f"{DEFAULT_SEED}:viz")``. Python salts ``hash`` of a str
# per process, so VIZ_SEED took a new value on every invocation and every figure that
# subsamples drew a different subset each run — while the data behind it was bitwise
# identical. ``derive_seed`` is stable across processes.
VIZ_SEED = derive_seed(DEFAULT_SEED, "viz")


def viz_rng(seed=None):
    """Return a local ``numpy.random.Generator`` for visualization sampling.

    Never mutates the global NumPy random state. Pass an explicit ``seed`` to
    override the default documented visualization seed (e.g. from a RunContext).
    """
    return np.random.default_rng(VIZ_SEED if seed is None else seed)


def deterministic_subsample(items, max_n, rng=None):
    """Deterministically subsample a sequence to at most ``max_n`` elements.

    Returns a list. Order of the returned subset is sorted by original index so
    the result is stable across runs. If ``len(items) <= max_n`` the input is
    returned as a list unchanged.

    Parameters
    ----------
    items : sequence (streamlines, arrays, ...)
    max_n : int, maximum number to keep.
    rng : numpy.random.Generator, optional; defaults to ``viz_rng()``.
    """
    n = len(items)
    if max_n is None or n <= max_n:
        return list(items)
    if rng is None:
        rng = viz_rng()
    idx = np.sort(rng.choice(n, size=max_n, replace=False))
    return [items[i] for i in idx]
