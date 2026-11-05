"""Plotting helpers for the notebooks (Stage 2b)."""
from __future__ import annotations

import numpy as np


def tilt_heatmap(ax, tilts_1, tilts_2, values, title: str = "", mark_argmax: bool = True):
    """Heatmap of a joint objective ``values[i, j]`` = J(t_1[i], t_2[j]) on the axes ``ax``.

    Sector 1's tilt runs along x and sector 2's along y. The dotted diagonal is the equal-tilt line
    t_1 = t_2 (the mirror axis of the facing-sites layout); a star marks the (first) maximum.
    Returns the mesh, so the caller can add a colour bar with ``fig.colorbar(mesh, ax=ax)``.
    """
    t1 = np.asarray(tilts_1, dtype=float)
    t2 = np.asarray(tilts_2, dtype=float)
    values = np.asarray(values, dtype=float)
    if values.shape != (len(t1), len(t2)):
        raise ValueError(f"values must have shape ({len(t1)}, {len(t2)}), got {values.shape}")
    mesh = ax.pcolormesh(t1, t2, values.T, shading="nearest", cmap="viridis")
    lo, hi = max(t1.min(), t2.min()), min(t1.max(), t2.max())
    ax.plot([lo, hi], [lo, hi], color="white", lw=0.8, ls=":")
    if mark_argmax:
        i, j = np.unravel_index(np.argmax(values), values.shape)
        ax.plot(t1[i], t2[j], marker="*", color="white", mec="k", ms=14, ls="none")
    ax.set_xlabel("$t_1$ [deg]")
    ax.set_ylabel("$t_2$ [deg]")
    ax.set_title(title)
    ax.set_aspect("equal")
    return mesh
