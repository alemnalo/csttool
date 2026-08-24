"""
Cross-process stability of derived sub-seeds.

These tests must spawn subprocesses. Python salts ``hash()`` of a ``str``
per process (PEP 456), but the salt is fixed *within* a process, so an
in-process assertion that two calls agree passes even when the value changes
on every invocation. That is exactly how the original defect survived: every
figure that subsamples drew a different subset each run while the data behind
it was bitwise identical, and no test noticed.

Each test therefore runs the expression in fresh interpreters with differing
``PYTHONHASHSEED`` values and asserts the results agree.
"""

import os
import subprocess
import sys

import numpy as np
import pytest

from csttool.reproducibility.context import DEFAULT_SEED, RunContext, derive_seed
from csttool.viz.utils import VIZ_SEED, deterministic_subsample, viz_rng


HASH_SEEDS = ("0", "1", "12345", "random")


def in_subprocesses(expression: str) -> list[str]:
    """Evaluate ``expression`` in fresh interpreters under differing hash salts.

    Returns one stdout string per interpreter.
    """
    outputs = []
    for hash_seed in HASH_SEEDS:
        env = {**os.environ, "PYTHONHASHSEED": hash_seed}
        result = subprocess.run(
            [sys.executable, "-c", f"print({expression})"],
            env=env, text=True, capture_output=True, check=True,
        )
        outputs.append(result.stdout.strip())
    return outputs


def test_builtin_hash_is_actually_salted():
    """Guard the premise: if this fails, the other tests prove nothing."""
    values = in_subprocesses('hash("42:viz") & 0xFFFFFFFF')
    assert len(set(values)) > 1, (
        "hash() of a str was stable across processes, so PYTHONHASHSEED is "
        "being pinned somewhere and these tests cannot detect a regression"
    )


def test_derive_seed_is_stable_across_processes():
    values = in_subprocesses(
        "__import__('csttool.reproducibility.context', fromlist=['x'])"
        ".derive_seed(42, 'viz')"
    )
    assert len(set(values)) == 1, f"derive_seed differs across processes: {values}"


def test_viz_seed_is_stable_across_processes():
    values = in_subprocesses(
        "__import__('csttool.viz.utils', fromlist=['x']).VIZ_SEED"
    )
    assert len(set(values)) == 1, f"VIZ_SEED differs across processes: {values}"


def test_deterministic_subsample_is_stable_across_processes():
    """The property the name promises, checked where it actually broke."""
    values = in_subprocesses(
        "__import__('csttool.viz.utils', fromlist=['x'])"
        ".deterministic_subsample(list(range(10_000)), 8)"
    )
    assert len(set(values)) == 1, (
        f"deterministic_subsample drew different subsets across processes: {values}"
    )


def test_run_context_rng_viz_is_stable_across_processes():
    values = in_subprocesses(
        "__import__('csttool.reproducibility.context', fromlist=['x'])"
        ".RunContext(run_seed=42).rng_viz().integers(0, 1_000_000, size=5).tolist()"
    )
    assert len(set(values)) == 1, f"rng_viz differs across processes: {values}"


def test_run_context_rng_perturb_is_stable_across_processes():
    values = in_subprocesses(
        "__import__('csttool.reproducibility.context', fromlist=['x'])"
        ".RunContext(run_seed=42).rng_perturb(3).integers(0, 1_000_000, size=5).tolist()"
    )
    assert len(set(values)) == 1, f"rng_perturb differs across processes: {values}"


def test_viz_seed_matches_run_context_default():
    """The module constant and the RunContext method must not drift apart."""
    assert VIZ_SEED == derive_seed(DEFAULT_SEED, "viz")
    expected = np.random.default_rng(VIZ_SEED).integers(0, 1_000_000, size=5)
    actual = RunContext(run_seed=DEFAULT_SEED).rng_viz().integers(0, 1_000_000, size=5)
    np.testing.assert_array_equal(actual, expected)


def test_derive_seed_is_in_range_and_label_order_matters():
    assert 0 <= derive_seed(42, "viz") < 2**32
    assert derive_seed(42, "viz") != derive_seed(42, "perturb")
    assert derive_seed(42, "perturb", 0) != derive_seed(42, "perturb", 1)
    assert derive_seed(1, "viz") != derive_seed(2, "viz")


def test_derive_seed_values_are_pinned():
    """Changing these silently reshuffles every seeded subsample."""
    assert derive_seed(DEFAULT_SEED, "viz") == derive_seed(42, "viz")
    assert derive_seed(42, "viz") == int.from_bytes(
        __import__("hashlib").blake2b(b"42:viz", digest_size=4).digest(), "big"
    )


@pytest.mark.parametrize("seed", [0, 1, 42, 2**31])
def test_viz_rng_accepts_explicit_seed(seed):
    a = viz_rng(seed).integers(0, 1_000_000, size=5)
    b = viz_rng(seed).integers(0, 1_000_000, size=5)
    np.testing.assert_array_equal(a, b)


def test_deterministic_subsample_preserves_order_and_size():
    items = list(range(1000))
    subset = deterministic_subsample(items, 10)
    assert len(subset) == 10
    assert subset == sorted(subset)
    assert deterministic_subsample(items, 5000) == items
