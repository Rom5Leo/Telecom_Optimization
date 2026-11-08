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


def grid_image(values, nx: int, ny: int, valid=None):
    """Flat per-cell values of a :func:`telecomopt.coverage.sample_rectangle` grid -> an image array.

    ``sample_rectangle`` orders the cells x-major (index = i * ny + j for x-column i and y-row j). An
    image has rows = y and columns = x, so the result has shape ``(ny, nx)`` with ``image[j, i]`` the
    value of cell (i, j); draw it with ``imshow(..., origin="lower")``. Cells where ``valid`` is False
    (e.g. the D09 tower floors) become NaN, so they are not drawn. Returns a float array.
    """
    values = np.asarray(values, dtype=float)
    if values.size != nx * ny:
        raise ValueError(f"need nx * ny = {nx * ny} values, got {values.size}")
    if valid is not None and np.asarray(valid).shape != values.shape:
        raise ValueError("valid must have the shape of values")
    # if a mask is given, put NaN where it is False; reshape to (nx, ny) (axis 0 = x, as
    # sample_rectangle made it); transpose so rows are y.
    # e.g. nx = 3, ny = 2, values [0, 1, 2, 3, 4, 5] -> [[0, 2, 4], [1, 3, 5]].

    if valid is not None:
        values = np.where(np.array(valid), np.array(values), np.nan)

    image_reshape = values.reshape(nx, ny).T

    return image_reshape


def field_map(ax, values, nx: int, ny: int, extent, valid=None, masts=(), **imshow_kw):
    """Draw one per-cell field of a ``sample_rectangle`` grid on ``ax`` as an image.

    ``extent = (x_min, x_max, y_min, y_max)`` of the region; ``masts`` is a sequence of (x, y) mast
    positions drawn as black triangles; further keyword arguments go to ``imshow`` (``cmap``, ``vmin``,
    ``vmax``, ...). Returns the image, for a colour bar.
    """
    image = ax.imshow(grid_image(values, nx, ny, valid), origin="lower", extent=list(extent),
                      aspect="equal", **imshow_kw)
    for mx, my in masts:
        ax.plot(mx, my, marker="^", color="k", mec="white", ms=7, ls="none")
    return image
