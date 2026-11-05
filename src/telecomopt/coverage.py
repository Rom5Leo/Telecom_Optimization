"""Demand-weighted coverage/capacity metrics for a single sector (Stage 1b).

Turns the single-sector spatial radio model (telecomopt.rf) into the numbers an optimizer
can target: a demand field rho(x,y) over the sector's service region, and three rho-weighted
metrics computed from per-point spectral efficiency.

    served fraction         - share of demand clearing a service threshold (coverage/outage)
    5th-percentile SE        - the SE *threshold* marking the lower 5% of the demand-weighted
                               distribution (a boundary value, not the mean of the worst 5%; "cell-edge" by
                               convention; NOT necessarily the outermost points, and distinct from
                               [E]'s 5th-percentile-of-path-gain coverage)
    mean spectral efficiency - a capacity *proxy* (Shannon; true capacity also needs bandwidth,
                               scheduling and load, none modelled here)

Single sector => no interferers => SINR reduces to SNR (via path_gain_2d_db). The service
region is the sector's +/- half-azimuth wedge out to d_max, excluding the near-tower floor
d_min (D09). Demand rho is a field: uniform (area metrics), or user-density hotspots (OQ9).
"""
from __future__ import annotations

import numpy as np

from telecomopt.rf import RFParams, DEFAULT_PARAMS, path_gain_2d_db


def _as_weighted(values, weights):
    """Shared contract for the weighted metrics: matching shapes, finite non-negative weights,
    and a strictly positive total (else the weighted distribution is undefined)."""
    values = np.asarray(values, float); weights = np.asarray(weights, float)
    if values.shape != weights.shape:
        raise ValueError(f"values and weights must share shape, got {values.shape} vs {weights.shape}")
    if not np.all(np.isfinite(weights)) or np.any(weights < 0):
        raise ValueError("weights must be finite and non-negative")
    total = float(weights.sum())
    if not total > 0.0:
        raise ValueError("total weight must be > 0 (zero demand leaves the metric undefined)")
    return values, weights, total


def sample_wedge(d_min=35.0, d_max=250.0, half_az_deg=60.0, n_r=60, n_az=61):
    """Polar grid over the sector wedge. Returns (dist, az_deg, area) flat arrays.

    Cell-centred (midpoint) sampling, so `area` (the polar element r*dr*dphi at each point)
    integrates correctly: a uniform rho then sums to the wedge area, and area-based weighting
    falls out of rho=1. Excludes the near-tower floor d_min (D09).
    """
    dr = (d_max - d_min) / n_r
    dphi_deg = 2 * half_az_deg / n_az
    r = d_min + (np.arange(n_r) + 0.5) * dr                 # cell centres (midpoint rule)
    az = -half_az_deg + (np.arange(n_az) + 0.5) * dphi_deg
    R, A = np.meshgrid(r, az, indexing="ij")
    dist = R.ravel()
    az_deg = A.ravel()
    area = dist * dr * np.radians(dphi_deg)                 # polar area element r*dr*dphi
    return dist, az_deg, area


def sample_rectangle(x_min, x_max, y_min, y_max, nx, ny):
    """Cell-centred grid over an axis-aligned rectangle. Returns (x, y, area) flat arrays.

    The Cartesian counterpart of :func:`sample_wedge`: [x_min, x_max] x [y_min, y_max] is split
    into ``nx`` x ``ny`` equal cells; every point is a cell centre (midpoint rule) and ``area`` is
    the cell area, so a uniform demand sums to the rectangle area. Bounds are arbitrary
    (e.g. y in [-a, b]); symmetric bounds give a mirror-symmetric point set.

    Ordering: x-major, ``k = i * ny + j`` is the centre of column ``i`` (x) and row ``j`` (y), so
    ``x.reshape(nx, ny)`` is the map layout (axis 0 = x, axis 1 = y).

    No sample lies on the boundary. For odd ``nx`` a column of samples falls exactly on the
    mid-line x = (x_min + x_max) / 2 — the tie line of the two-site layout (D11) — so use even ``nx``
    there. The near-tower exclusion (D09) is not applied here: it enters as zero weight.
    """
    for name, v in (("x_min", x_min), ("x_max", x_max), ("y_min", y_min), ("y_max", y_max)):
        if not np.isfinite(v):
            raise ValueError(f"{name} must be finite, got {v}")
    if not (x_max > x_min and y_max > y_min):
        raise ValueError("need x_max > x_min and y_max > y_min")
    for name, n in (("nx", nx), ("ny", ny)):
        if not (isinstance(n, (int, np.integer)) and n >= 1):
            raise ValueError(f"{name} must be an integer >= 1, got {n!r}")

    width, height = x_max - x_min, y_max - y_min
    dx, dy = width / nx, height / ny

    xc = x_min + np.arange(nx) * dx + 0.5 * dx
    yc = y_min + np.arange(ny) * dy + 0.5 * dy
    x, y = np.meshgrid(xc, yc, indexing="ij")
    x = x.ravel()
    y = y.ravel()
    area = np.full(len(x), dx * dy)

    return (x, y, area)


def to_xy(dist, az_deg):
    """Ground coordinates from polar (boresight = +x)."""
    phi = np.radians(az_deg)
    return dist * np.cos(phi), dist * np.sin(phi)


def uniform_demand(dist, az_deg):
    """Uniform demand field rho = 1 (metrics become area-based)."""
    return np.ones_like(np.asarray(dist, dtype=float))


def weighted_mean(values, weights):
    """Weighted mean of `values` with finite non-negative `weights` (positive total)."""
    values, weights, _ = _as_weighted(values, weights)
    return float(np.average(values, weights=weights))


def hotspot_demand(x, y, centers, sigmas, peaks, floor=0.0):
    """Sum-of-Gaussians user-density field. centers=[(cx,cy),...], one sigma and peak per hotspot.

    rho(x,y) = floor + sum_k  peak_k * exp( -((x-cx_k)^2 + (y-cy_k)^2) / (2 sigma_k^2) ).
    """
    x = np.asarray(x, dtype=float); y = np.asarray(y, dtype=float)
    if not (len(centers) == len(sigmas) == len(peaks)):
        raise ValueError("centers, sigmas and peaks must have equal length")
    if any(sig <= 0 for sig in sigmas):
        raise ValueError("hotspot sigmas must be positive")
    # start from the constant `floor`, then add one Gaussian bump per hotspot.
    rho = np.full_like(x, float(floor))
    for (cx, cy), s, p in zip(centers, sigmas, peaks):
        rho = rho + p * np.exp(-((x - cx)**2 + (y - cy)**2) / (2.0 * s * s))

    return rho


def weighted_percentile(values, weights, q):
    """The q-th weighted percentile (q in [0,100]): the smallest value whose cumulative weight,
    over `values` sorted ascending, reaches q% of the total weight. Inverse-empirical-CDF (lower)
    convention, no interpolation — a different convention can give a slightly different discrete value.
    """
    if not (0.0 <= q <= 100.0):
        raise ValueError("q must be in [0, 100]")
    values, weights, _ = _as_weighted(values, weights)
    # sort by value, take the cumulative weight, find where it crosses q% of the total.
    order = np.argsort(values); v = values[order]; cw = np.cumsum(weights[order])
    cutoff = (q / 100.0) * cw[-1]; idx = int(np.searchsorted(cw, cutoff))
    return float(v[min(idx, len(v) - 1)])
    


def served_fraction(values, weights, threshold):
    """Weighted fraction of points with value >= threshold."""
    values, weights, total = _as_weighted(values, weights)
    # total weight of points at/above the threshold, divided by the total weight.
    return float(weights[values >= threshold].sum() / total)


def spectral_efficiency_field(tilt_e_deg, dist, az_deg, params=DEFAULT_PARAMS, tilt_m_deg=0.0):
    """Per-point spectral efficiency (bps/Hz). Single sector: SINR = SNR = P*g/N0 (no interferers)."""
    # path gain (dB) at each point, then SNR (linear) = 10^((P_dBm + g_dB - N0_dBm)/10), then Shannon.
    pg = np.array([path_gain_2d_db(d, a, tilt_e_deg, tilt_m_deg, params) for d, a in zip(dist, az_deg)])
    # turn the path-gain array into spectral efficiency.
    snr_lin = 10.0 ** ((params.p_dbm + pg - params.n0_dbm) / 10.0)
    return np.log2(1.0 + snr_lin)


def coverage_report(tilt_e_deg, dist, az_deg, demand, area, threshold=1.0,
                    params=DEFAULT_PARAMS, tilt_m_deg=0.0):
    """The three rho-weighted metrics at a given tilt. Aggregation weight = demand * area."""
    se = spectral_efficiency_field(tilt_e_deg, dist, az_deg, params, tilt_m_deg)
    # aggregation weight is demand density times area element; then the three metrics.
    w = np.asarray(demand, float) * np.asarray(area, float)
    return {"mean_se": weighted_mean(se, w),
            "p5_se": weighted_percentile(se, w, 5.0),
            "served_fraction": served_fraction(se, w, threshold)}


def sweep_metrics(cube, weights):
    """Demand-weighted mean SE and 5th-percentile SE for every tilt combination (Stage 2b).

    Parameters
    ----------
    cube    : array ``(n_1, ..., n_N, n_points)`` — the spectral efficiency of every tilt combination at
              every point (:func:`telecomopt.network.se_cube`).
    weights : array ``(n_points,)`` — aggregation weight of each point, demand x cell area, with
              ``0`` at points outside the valid region (D09). Finite, non-negative, positive total.

    Returns a dict ``{"mean_se": ..., "p5_se": ...}``; each value is an array of shape
    ``(n_1, ..., n_N)`` holding the 1b metrics (:func:`weighted_mean`, :func:`weighted_percentile` at 5)
    of that combination's SE over the points. The metric definitions are unchanged from Stage 1b.
    """
    cube = np.asarray(cube, dtype=float)
    weights = np.asarray(weights, dtype=float)
    if cube.ndim < 2:
        raise ValueError(f"cube must have shape (..., n_points) with at least one tilt axis, got {cube.shape}")
    if weights.shape != (cube.shape[-1],):
        raise ValueError(f"weights must have shape ({cube.shape[-1]},), got {weights.shape}")
    # two empty result arrays shaped like the cube without its last axis -> for every
    # combination of tilt numbers, the two 1b metrics over the points -> return both in a dict.

    shape = cube.shape[:-1]
    box_means, box_p5 = np.empty(shape), np.empty(shape)

    for idx in np.ndindex(*shape):
        box_means[idx] = weighted_mean(cube[idx], weights)
        box_p5[idx] = weighted_percentile(cube[idx], weights, q = 5.0)

    return {"mean_se": box_means, "p5_se": box_p5}


def non_dominated(*objectives):
    """Pareto mask: True for the options that no other option dominates (all objectives MAXIMIZED).

    Option j **dominates** option i when it is no worse on every objective and strictly better on at
    least one. The non-dominated set is the Pareto front — the genuine trade-off; every other option can
    be discarded before trading objectives off. Exact ties are not dominated (both are kept).

    Parameters
    ----------
    *objectives : one array per objective, all the same shape (e.g. mean SE and 5th-pct SE over a tilt
                  sweep, or over a (t1, t2) grid). Minimize an objective by passing its negative.

    Returns a boolean array with the shape of the inputs. Pairwise comparison: O(n^2) time and memory,
    fine up to a few thousand options.
    """
    if not objectives:
        raise ValueError("need at least one objective")
    arrays = [np.asarray(o, dtype=float) for o in objectives]
    shape = arrays[0].shape
    if any(a.shape != shape for a in arrays):
        raise ValueError(f"all objectives must share shape, got {[a.shape for a in arrays]}")
    F = np.column_stack([a.ravel() for a in arrays])          # (n_options, n_objectives)
    if not np.all(np.isfinite(F)):
        raise ValueError("objective values must be finite")
    no_worse = np.all(F[:, None, :] >= F[None, :, :], axis=2)   # [j, i]: j >= i on every objective
    better = np.any(F[:, None, :] > F[None, :, :], axis=2)      # [j, i]: j > i on at least one
    dominated = np.any(no_worse & better, axis=0)               # some j dominates i
    return (~dominated).reshape(shape)
