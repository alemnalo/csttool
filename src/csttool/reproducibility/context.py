"""
RunContext: Encapsulate run-level state to avoid passing many parameters.

The RunContext object carries run_seed, provenance, and timing information
through the pipeline, and provides hierarchical RNG methods for different
subsystems (tracking, visualization, perturbation).
"""

import hashlib
from dataclasses import dataclass, field
from datetime import datetime
import numpy as np

from .provenance import get_provenance_dict


DEFAULT_SEED = 42


def derive_seed(*parts) -> int:
    """Derive a stable 32-bit sub-seed from a label.

    Sub-seeds let each subsystem draw independently of the others while all of
    them remain a pure function of ``run_seed``.

    Uses BLAKE2b rather than the builtin ``hash``. ``hash`` of a ``str`` is
    salted per process (PEP 456) unless ``PYTHONHASHSEED`` is set, so a seed
    derived from it takes a different value on every invocation. Deriving
    sub-seeds that way silently made every subsampled QC figure irreproducible
    across runs while the underlying data was bitwise identical.

    Parameters
    ----------
    *parts
        Label components, joined with ``:``. Order is significant.

    Returns
    -------
    int
        A seed in ``[0, 2**32)``, stable across processes, platforms and
        Python versions.
    """
    label = ":".join(str(part) for part in parts).encode("utf-8")
    return int.from_bytes(hashlib.blake2b(label, digest_size=4).digest(), "big")


@dataclass
class RunContext:
    """Context object for a single csttool run.

    Attributes:
        run_seed: Random seed for this run (default: 42)
        provenance: Dict containing git hash, Python version, dependencies, platform
        start_time: Timestamp when run started
    """
    run_seed: int = DEFAULT_SEED
    provenance: dict = field(default_factory=get_provenance_dict)
    start_time: datetime = field(default_factory=datetime.now)

    def rng_tracking_seed(self) -> int:
        """Return seed for DIPY LocalTracking (legacy API requires int)."""
        return self.run_seed

    def rng_viz(self) -> np.random.Generator:
        """Return RNG for visualization subsampling.

        Derives a sub-seed via :func:`derive_seed` so visualization randomness
        is independent of tracking randomness, yet reproducible across runs.
        """
        return np.random.default_rng(derive_seed(self.run_seed, "viz"))

    def rng_perturb(self, offset: int) -> np.random.Generator:
        """Return RNG for sensitivity tests with replicate offset.

        Args:
            offset: Replicate number (0, 1, 2, ...) to ensure different
                    perturbations across replicates

        Returns:
            numpy.random.Generator with unique seed for this replicate
        """
        return np.random.default_rng(
            derive_seed(self.run_seed, "perturb", offset)
        )
