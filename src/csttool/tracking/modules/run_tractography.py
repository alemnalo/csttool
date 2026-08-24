def run_tractography(csapeaks, stopping_criterion, seeds, affine, step_size=0.5, random_seed=None, verbose=False, visualize=False):
    """Run deterministic local tractography.

    Args:
        csapeaks (PeaksAndMetrics): Direction field from estimate_directions().
            Note: PeaksAndMetrics inherits from EuDXDirectionGetter, so it serves
            as a valid direction getter for LocalTracking.
        stopping_criterion: Stopping criterion from seed_and_stop().
        seeds (ndarray): Seed points in world coordinates (N, 3).
        affine (ndarray): 4x4 affine transformation matrix.
        step_size (float): Tracking step size in mm (default 0.5).
        random_seed (int, optional): Random seed for reproducible tracking.
            If None, results may vary between runs.
        verbose (bool): Print processing details.
        visualize (bool): Show 2D projection plots of streamlines.

    Returns:
        Streamlines: Generated streamlines in RASMM space.
    """
    from dipy.tracking.local_tracking import LocalTracking
    from dipy.tracking.streamline import Streamlines, length
    import numpy as np
    import matplotlib.pyplot as plt

    from ...viz.utils import viz_rng

    if verbose:
        print(f"  → Running tractography (step={step_size}mm)...")
        if random_seed is not None:
            print(f"    • Random seed: {random_seed}")

    streamline_generator = LocalTracking(
        csapeaks,
        stopping_criterion,
        seeds,
        affine=affine,
        step_size=step_size,
        random_seed=random_seed
    )
    streamlines = Streamlines(streamline_generator)

    if verbose:
        print(f"    • Generated: {len(streamlines):,} streamlines")
        if len(streamlines) > 0:
            lengths = np.array([length(s) for s in streamlines])
            print(f"    • Length statistics:")
            print(f"    ├─ Mean:   {lengths.mean():.1f} mm")
            print(f"    ├─ Median: {np.median(lengths):.1f} mm")
            print(f"    └─ Range:  [{lengths.min():.1f}, {lengths.max():.1f}] mm")
    
    # Visualization: Native space streamlines (2D projections)
    if visualize and len(streamlines) > 0:
        fig, axes = plt.subplots(1, 3, figsize=(15, 5))
        fig.suptitle(f"Whole-Brain Tractography ({len(streamlines):,} streamlines)", 
                     fontsize=14)
        
        # Sample streamlines for visualization (max 5000).
        # viz_rng, not np.random: the global RNG is unseeded here, which made
        # this panel differ between runs, and drawing from it would also
        # perturb the global stream for anything downstream.
        n_vis = min(5000, len(streamlines))
        vis_indices = np.sort(
            viz_rng().choice(len(streamlines), n_vis, replace=False)
        )
        
        views = [
            (0, 'Sagittal (Y-Z)', 1, 2, 'Y (mm)', 'Z (mm)', 'blue'),
            (1, 'Coronal (X-Z)', 0, 2, 'X (mm)', 'Z (mm)', 'green'),
            (2, 'Axial (X-Y)', 0, 1, 'X (mm)', 'Y (mm)', 'red'),
        ]
        
        for ax_idx, title, dim1, dim2, xlabel, ylabel, color in views:
            ax = axes[ax_idx]
            for idx in vis_indices:
                s = streamlines[idx]
                ax.plot(s[:, dim1], s[:, dim2], alpha=0.1, linewidth=0.5, color=color)
            ax.set_xlabel(xlabel)
            ax.set_ylabel(ylabel)
            ax.set_title(title)
            ax.set_aspect('equal')
            ax.grid(True, alpha=0.3)
        
        plt.tight_layout()
        plt.show()
    
    return streamlines