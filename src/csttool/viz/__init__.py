"""
csttool visualization package.

Shared, presentation-only infrastructure for every figure csttool emits:

- ``style``    — typography, rcParams, canonical colours/colormaps, savefig policy,
                 and small presentation helpers (colorbars).
- ``geometry`` — affine/orientation handling, voxel<->world transforms, image-display
                 orientation, slice/bounds calculations, and L/R marker placement.
- ``utils``    — general helpers that are neither style policy nor spatial geometry
                 (deterministic subsampling, misc).

These modules contain no pipeline logic and must not import from the pipeline
packages, to keep the dependency graph acyclic.
"""

from . import style, geometry, utils

__all__ = ["style", "geometry", "utils"]
