"""Demand-weighted coverage/capacity metrics for a single sector (Stage 1b).

Turns the single-sector spatial radio model (telecomopt.rf) into the numbers an optimizer
can target: a demand field rho(x,y) over the sector's service region, and three rho-weighted
metrics computed from per-point spectral efficiency.

    served fraction         - share of demand clearing a service threshold (coverage/outage)
    5th-percentile SE        - worst-served 5% of the demand-weighted distribution ("cell-edge" by
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


def to_xy(dist, az_deg):
    """Ground coordinates from polar (boresight = +x)."""
    phi = np.radians(az_deg)
    return dist * np.cos(phi), dist * np.sin(phi)


def uniform_demand(dist, az_deg):
    """Uniform demand field rho = 1 (metrics become area-based)."""
    return np.ones_like(np.asarray(dist, dtype=float))


def weighted_mean(values, weights):
    """Weighted mean of `values` with non-negative `weights`."""
    return float(np.average(np.asarray(values, float), weights=np.asarray(weights, float)))


def hotspot_demand(x, y, centers, sigmas, peaks, floor=0.0):
    """Sum-of-Gaussians user-density field. centers=[(cx,cy),...], one sigma and peak per hotspot.

    rho(x,y) = floor + sum_k  peak_k * exp( -((x-cx_k)^2 + (y-cy_k)^2) / (2 sigma_k^2) ).
    """
    x = np.asarray(x, dtype=float); y = np.asarray(y, dtype=float)
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
    values = np.asarray(values, float); weights = np.asarray(weights, float)
    # sort by value, take the cumulative weight, find where it crosses q% of the total.
    order = np.argsort(values); v = values[order]; cw = np.cumsum(weights[order])
    cutoff = (q / 100.0) * cw[-1]; idx = int(np.searchsorted(cw, cutoff))
    return float(v[min(idx, len(v) - 1)])
    


def served_fraction(values, weights, threshold):
    """Weighted fraction of points with value >= threshold."""
    values = np.asarray(values, float); weights = np.asarray(weights, float)
    # total weight of points at/above the threshold, divided by the total weight.
    return weights[values >= threshold].sum() / weights.sum()


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
    
