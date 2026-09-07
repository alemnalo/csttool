"""
csttool visualization package.

Shared, presentation-only infrastructure for every figure csttool emits:

- ``style``    — typography, rcParams, canonical colours/colormaps, the single
                 savefig policy, and small presentation helpers (colorbars).
- ``geometry`` — affine/orientation handling, voxel<->world transforms, image-display
                 orientation, slice/bounds calculations, and L/R marker placement.
- ``render``   — draws one layer into one existing Axes. Creates no Figure, saves
                 nothing.
- ``layout``   — sizes Figures and Axes in millimetres, lays out panel rows, and
                 draws the shared key/caption furniture. Creates Figures, saves
                 nothing. Holds no document dimensions: callers pass the width
                 their page needs.
- ``utils``    — general helpers that are neither style policy nor spatial geometry
                 (deterministic subsampling, misc).

The division of labour is deliberate: ``render`` and ``layout`` are *primitives*,
and figure **composition** — which panels appear, in what order, at what final
size — belongs to the caller. That is what lets the operational QC figures and a
publication figure share one rendering foundation while keeping different
compositions, without either growing conditional branches for the other.

These modules contain no pipeline logic and must not import from the pipeline
packages, to keep the dependency graph acyclic.
"""

from . import style, geometry, utils, render, layout

__all__ = ["style", "geometry", "utils", "render", "layout"]
