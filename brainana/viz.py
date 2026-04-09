"""Brain visualization using nilearn on fsaverage5.

Generates publication-quality brain surface maps from vertex-level predictions.
"""

from __future__ import annotations

import logging
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

logger = logging.getLogger(__name__)

N_VERTICES_LEFT = 10242


def plot_brain_map(
    brain_map: np.ndarray,
    title: str = "",
    output_path: str | Path | None = None,
    cmap: str = "RdBu_r",
    threshold: float | None = None,
    vmax: float | None = None,
) -> str | None:
    """Plot brain activation on fsaverage5 surface (both hemispheres).

    Parameters
    ----------
    brain_map : np.ndarray of shape (n_vertices,) where n = 20484
    title : Plot title
    output_path : Where to save the figure. If None, returns None.
    cmap : Matplotlib colormap
    threshold : Minimum absolute value to display
    vmax : Maximum value for colormap scaling

    Returns
    -------
    str path to saved figure, or None
    """
    from nilearn import datasets, plotting

    fsaverage = datasets.fetch_surf_fsaverage("fsaverage5")
    left_data = brain_map[:N_VERTICES_LEFT]
    right_data = brain_map[N_VERTICES_LEFT:]

    if vmax is None:
        vmax = max(np.abs(left_data).max(), np.abs(right_data).max())
        vmax = max(vmax, 0.01)

    fig, axes = plt.subplots(2, 2, figsize=(14, 8), subplot_kw={"projection": "3d"})

    views = [
        (fsaverage["infl_left"], left_data, "lateral", "Left lateral"),
        (fsaverage["infl_left"], left_data, "medial", "Left medial"),
        (fsaverage["infl_right"], right_data, "lateral", "Right lateral"),
        (fsaverage["infl_right"], right_data, "medial", "Right medial"),
    ]

    for ax, (mesh, data, view, subtitle) in zip(axes.flat, views):
        plotting.plot_surf_stat_map(
            mesh,
            data,
            view=view,
            cmap=cmap,
            threshold=threshold,
            vmax=vmax,
            axes=ax,
            colorbar=False,
        )
        ax.set_title(subtitle, fontsize=10)

    fig.suptitle(title, fontsize=14, fontweight="bold", y=0.98)
    fig.tight_layout(rect=[0, 0, 1, 0.95])

    if output_path:
        output_path = Path(output_path)
        output_path.parent.mkdir(parents=True, exist_ok=True)
        fig.savefig(output_path, dpi=150, bbox_inches="tight")
        plt.close(fig)
        logger.info(f"Saved brain map to {output_path}")
        return str(output_path)

    plt.close(fig)
    return None


def plot_score_progression(
    scores: list[float],
    title: str = "Brain Activation Over Iterations",
    output_path: str | Path | None = None,
    target_region: str = "",
) -> str | None:
    """Plot the fitness curve (activation score over iterations)."""
    fig, ax = plt.subplots(figsize=(10, 5))

    iterations = list(range(len(scores)))
    ax.plot(iterations, scores, "o-", color="#2196F3", linewidth=2, markersize=8)

    best_idx = int(np.argmax(scores))
    ax.plot(best_idx, scores[best_idx], "*", color="#FF5722", markersize=20,
            label=f"Best: {scores[best_idx]:.4f} (iter {best_idx})")

    running_best = [max(scores[:i+1]) for i in range(len(scores))]
    ax.plot(iterations, running_best, "--", color="#4CAF50", linewidth=1.5,
            alpha=0.7, label="Running best")

    ax.set_xlabel("Iteration", fontsize=12)
    ax.set_ylabel(f"{target_region} Activation", fontsize=12)
    ax.set_title(title, fontsize=14, fontweight="bold")
    ax.legend(fontsize=10)
    ax.grid(True, alpha=0.3)

    if output_path:
        output_path = Path(output_path)
        output_path.parent.mkdir(parents=True, exist_ok=True)
        fig.savefig(output_path, dpi=150, bbox_inches="tight")
        plt.close(fig)
        return str(output_path)

    plt.close(fig)
    return None
