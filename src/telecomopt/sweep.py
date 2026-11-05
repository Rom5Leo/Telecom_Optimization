"""Joint tilt sweep over a rectangular region with uniform demand (Stage 2b).

Chains the Stage 2b building blocks into one call: cell-centred sampling of the region
(:func:`telecomopt.coverage.sample_rectangle`), area weights with the near-tower floor of every mast
excluded (D09), one received-power table per sector (:func:`telecomopt.network.power_table`), the
spectral-efficiency cube of every tilt combination (:func:`telecomopt.network.se_cube`), and the 1b
metrics on each combination (:func:`telecomopt.coverage.sweep_metrics`).

The cube does not depend on demand, so it is returned: a different demand field only needs new weights
and a new :func:`telecomopt.coverage.sweep_metrics` call.
"""
from __future__ import annotations

import numpy as np

from telecomopt.rf import RFParams, DEFAULT_PARAMS
from telecomopt.coverage import sample_rectangle, sweep_metrics
from telecomopt.network import exclusion_mask, power_table, se_cube


def tilt_sweep(sectors, tilts_e_deg, x_range=(0.0, 500.0), y_range=(-250.0, 250.0),
               nx: int = 100, ny: int = 100, d_min: float = 35.0,
               params: RFParams = DEFAULT_PARAMS):
    """Uniform-demand sweep over every combination of one candidate tilt per sector.

    Parameters
    ----------
    sectors     : sequence of :class:`telecomopt.network.Sector`.
    tilts_e_deg : one 1-D sequence of candidate electrical tilts (deg) per sector, same order as ``sectors``.
    x_range, y_range : region bounds (m); the region is cut into ``nx`` x ``ny`` equal cells.
    d_min       : near-tower exclusion radius (m, D09); cells whose centre is closer to any mast get zero weight.

    Returns a dict:
      ``x``, ``y``       flat cell-centre coordinates (n_points,),
      ``weights``        uniform-demand aggregation weight per point = cell area x validity (n_points,),
      ``cube``           SE of every combination at every point, shape ``(n_1, ..., n_N, n_points)``,
      ``mean_se``, ``p5_se``  demand-weighted metrics per combination, shape ``(n_1, ..., n_N)``,
      ``tilts``          the tilt arrays used, one per sector.
    """
    if len(tilts_e_deg) != len(sectors):
        raise ValueError(f"need one tilt list per sector, got {len(tilts_e_deg)} for {len(sectors)} sectors")
    x, y, area = sample_rectangle(x_range[0], x_range[1], y_range[0], y_range[1], nx, ny)
    weights = area * exclusion_mask(x, y, sectors, d_min)
    tilts = [np.asarray(t, dtype=float) for t in tilts_e_deg]
    tables = [power_table(x, y, s, t, params=params) for s, t in zip(sectors, tilts)]
    cube = se_cube(tables, 10.0 ** (params.n0_dbm / 10.0))
    metrics = sweep_metrics(cube, weights)
    return {"x": x, "y": y, "weights": weights, "cube": cube,
            "mean_se": metrics["mean_se"], "p5_se": metrics["p5_se"], "tilts": tilts}
