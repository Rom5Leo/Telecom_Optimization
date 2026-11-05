"""Multi-sector radio field: best-server association and interference (Stage 2a).

Extends the single-sector spatial radio model (:mod:`telecomopt.rf`) to several sectors that
transmit simultaneously on the same resource. At every ground point:

    received power   S_j(x) = P * g_j(x; t_j)              (linear, mW; g_j from path_gain_2d_db)
    association      s(x)   = argmax_j S_j(x)              (best server: strongest received power)
    SINR             SINR(x) = S_s / ( sum_{j != s} S_j + N )
    spectral eff.    C(x)   = log2(1 + SINR)                ([E] Eq. 1 and Eq. 2)

Assumptions (decision log D11): all sectors share one resource and transmit simultaneously;
received powers and noise refer to the same bandwidth (per PRB, [E] Table I); the receiver decodes
one serving sector and treats every other sector as interference; association is strongest received
power with no handover hysteresis, coordination or load balancing; shadowing is deferred (D06).

Each sector is described in its own local frame (distance from its mast, azimuth from its
boresight), so the existing single-sector functions are reused unchanged. Functions are written
for N sectors; Stage 2a uses N = 2.
"""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from telecomopt.rf import RFParams, DEFAULT_PARAMS, path_gain_2d_db, to_db


@dataclass(frozen=True)
class Sector:
    """One sector antenna: mast position (m, global frame) and boresight direction.

    ``boresight_deg`` is measured counter-clockwise from the global +x axis (0 = facing +x,
    180 = facing -x). Tilts are not stored here: they are the decision variable and are passed
    to the field functions separately.
    """

    x: float
    y: float
    boresight_deg: float


def to_sector_frame(x, y, sector: Sector):
    """Global ground coordinates -> this sector's local polar coordinates.

    Returns ``(dist, az_deg)`` as float arrays with the shape of ``x``:

    - ``dist``   horizontal distance from the sector's mast (m);
    - ``az_deg`` azimuth of the point relative to the sector's boresight, wrapped to
      [-180, 180) deg (positive = counter-clockwise from boresight).

    These are exactly the ``distance_m`` and ``azimuth_deg`` inputs of
    :func:`telecomopt.rf.path_gain_2d_db`.
    """
    x = np.asarray(x, dtype=float)
    y = np.asarray(y, dtype=float)
    if x.shape != y.shape:
        raise ValueError(f"x and y must share shape, got {x.shape} vs {y.shape}")
    # offset from the mast -> distance; its direction minus the boresight -> azimuth, wrapped into [-180, 180).
    # e.g. sector at (500, 0) facing 180 deg, point (400, 0): offset (-100, 0) -> dist 100 m;
    #      global direction 180 deg, relative 180 - 180 = 0 deg (straight ahead).
    #      np.hypot / np.arctan2 work element-wise; a modulo wrap maps 190 -> -170 and 180 -> -180.

    dx = x - sector.x
    dy = y - sector.y
    dist = np.hypot(dx, dy)
    az_deg = (((np.degrees(np.arctan2(dy,dx)) - sector.boresight_deg) + 180)%360) - 180

    return (dist, az_deg)


def received_power_mw(x, y, sector: Sector, tilt_e_deg: float, tilt_m_deg: float = 0.0,
                      params: RFParams = DEFAULT_PARAMS):
    """Received power (linear, mW) from one sector at each ground point.

        S(x) = 10^((P_dBm + g_dB(x)) / 10),   g_dB = G0 + A - PL   (path_gain_2d_db)

    Points are converted to the sector's local frame first, so a sector at the origin facing +x
    reproduces :func:`telecomopt.rf.path_gain_2d_db` exactly. Loops over points (the rf layer is
    scalar); fine for maps of ~1e4 points.
    """
    # local frame (to_sector_frame), path gain per point (path_gain_2d_db is scalar:
    # loop, and return the same shape as x), then dBm -> mW.
    # e.g. P = 29 dBm and g = -80 dB give -51 dBm = 10^(-51/10) mW ~ 7.9e-6 mW.
    
    dist, az = to_sector_frame(x, y, sector)
    shape = dist.shape
    dist = dist.ravel()
    az = az.ravel()

    tosec = zip(dist, az)
    pg_db = [path_gain_2d_db(d, a, tilt_e_deg, tilt_m_deg, params) for d, a in tosec]
    pg_db = np.array(pg_db).reshape(shape)

    rp_dbm = (pg_db + params.p_dbm)
    rp = 10 ** (rp_dbm / 10)

    return rp




def power_table(x, y, sector: Sector, tilts_e_deg, tilt_m_deg: float = 0.0,
                params: RFParams = DEFAULT_PARAMS):
    """Received power of ONE sector at every point, for every candidate electrical tilt (Stage 2b).

    Parameters
    ----------
    x, y        : ground coordinates (m), same shape; flattened to ``n_points``.
    sector      : the :class:`Sector` whose power is tabulated.
    tilts_e_deg : 1-D sequence of candidate electrical tilts (deg), finite, at least one.
    tilt_m_deg  : mechanical tilt (deg), a single value used for every row.

    Returns an array of shape ``(n_tilts, n_points)`` in mW; row ``k`` is
    ``received_power_mw(x, y, sector, tilts_e_deg[k], tilt_m_deg, params)``.

    Why a table: the power of a sector depends only on **its own** tilt, so a two-sector sweep over
    (t1, t2) needs one table per sector, not one field evaluation per pair. The pair (i, j) is
    ``np.stack([table_1[i], table_2[j]])``, ready for :func:`best_server_sinr`.
    """
    x = np.asarray(x, dtype=float)
    y = np.asarray(y, dtype=float)
    if x.shape != y.shape:
        raise ValueError(f"x and y must share shape, got {x.shape} vs {y.shape}")
    tilts = np.asarray(tilts_e_deg, dtype=float)
    if tilts.ndim != 1 or tilts.size < 1 or not np.all(np.isfinite(tilts)):
        raise ValueError("tilts_e_deg must be a non-empty 1-D sequence of finite values")

    x = x.ravel()
    y = y.ravel()

    list_of_arrays = [received_power_mw(x, y, sector, tilt, tilt_m_deg, params) for tilt in tilts]
    table = np.stack(list_of_arrays)

    return table


def best_server_sinr(powers_mw, noise_mw: float):
    """Best-server association and SINR from per-sector received powers.

    Parameters
    ----------
    powers_mw : array (n_sectors, n_points), received power of every sector at every point (mW).
    noise_mw  : noise power in the same bandwidth (mW), > 0.

    Returns
    -------
    serving : int array (n_points,), index of the strongest sector. Exact ties go to the lowest
              index — bookkeeping only: at a tie every choice gives the same SINR.
    sinr    : float array (n_points,), linear  S_serving / (sum of the other sectors + N).

    With one sector the interference sum is empty, so SINR = SNR (the Stage 1b case).
    """
    powers = np.asarray(powers_mw, dtype=float)
    if powers.ndim != 2 or powers.shape[0] < 1:
        raise ValueError(f"powers_mw must be (n_sectors, n_points), got shape {powers.shape}")
    if not np.all(np.isfinite(powers)) or np.any(powers < 0):
        raise ValueError("received powers must be finite and non-negative")
    if not (np.isfinite(noise_mw) and noise_mw > 0):
        raise ValueError("noise_mw must be finite and > 0")
    # strongest sector per point, its power, and the sum of all the others.
    # e.g. powers [[1, 5], [3, 1], [2, 1]], N = 0.5  ->  serving [1, 0],
    #      SINR = [3 / (1 + 2 + 0.5), 5 / (1 + 1 + 0.5)] = [0.857, 2.0].
    #      "sum of the others" = column total minus the serving power.
    
    serving = np.argmax(powers, axis=0) # returns the first maximum (the lowest-index tie rule);
    signal = powers.max(axis=0)
    total = powers.sum(axis=0)
    interference = total - signal
    sinr = signal / (interference + noise_mw)

    return serving, sinr


def sinr_field(x, y, sectors, tilts_e_deg, tilts_m_deg=None, params: RFParams = DEFAULT_PARAMS):
    """The full multi-sector field at the given points, for one tilt configuration.

    Parameters
    ----------
    x, y        : ground coordinates (m), same shape; flattened to ``n_points``.
    sectors     : sequence of :class:`Sector`.
    tilts_e_deg : electrical tilt per sector (same length as ``sectors``).
    tilts_m_deg : mechanical tilt per sector, default all 0.

    Returns a dict of flat arrays:
      ``powers`` (n_sectors, n_points) mW, ``serving`` (n_points,), ``sinr`` (linear),
      ``se`` = log2(1 + sinr) in bps/Hz.
    Noise is N = 10^(N0_dBm / 10) mW from ``params`` ([E] Table I, per PRB).
    """
    x = np.asarray(x, dtype=float).ravel()
    y = np.asarray(y, dtype=float).ravel()
    n = len(sectors)
    if n < 1:
        raise ValueError("need at least one sector")
    tilts_m_deg = [0.0] * n if tilts_m_deg is None else list(tilts_m_deg)
    if len(tilts_e_deg) != n or len(tilts_m_deg) != n:
        raise ValueError("need exactly one electrical and one mechanical tilt per sector")
    # one received_power_mw row per sector -> powers (n_sectors, n_points); then
    # best_server_sinr with N from params; then SE ([E] Eq. 2). Return the four keys above.
    # e.g. two sectors on 10 points -> powers.shape == (2, 10), serving/sinr/se each (10,).

    rp_ps = zip(sectors, tilts_e_deg, tilts_m_deg)
    powers = np.stack([received_power_mw(x, y, s, te, tm, params) for s,te,tm in rp_ps])

    noise_mw = 10 ** (params.n0_dbm / 10)

    serving, sinr = best_server_sinr(powers, noise_mw)
    se = np.log2(1 + sinr)

    return {"powers": powers, "serving": serving, "sinr": sinr, "se": se}


def se_cube(tables, noise_mw: float):
    """Spectral efficiency of every tilt combination at every point (Stage 2b).

    Parameters
    ----------
    tables   : sequence of arrays, one per sector; ``tables[s]`` has shape ``(n_s, n_points)`` — the
               :func:`power_table` of sector ``s`` (row = candidate tilt, mW). All share ``n_points``.
    noise_mw : noise power in the same bandwidth (mW), > 0; the same value :func:`sinr_field` uses,
               ``10 ** (params.n0_dbm / 10)``.

    Returns an array of shape ``(n_1, ..., n_N, n_points)``: entry ``cube[i, j, ..., :]`` is
    ``log2(1 + SINR)`` at every point when sector 1 uses its tilt number ``i``, sector 2 its tilt number
    ``j``, and so on — best-server association and interference exactly as in :func:`best_server_sinr`.

    Nothing here depends on demand, so one cube serves every demand field (weights enter later).
    Memory grows as ``n_1 * ... * n_N * n_points`` floats: fine for two sectors, a hard limit beyond.
    """
    tables = [np.asarray(t, dtype=float) for t in tables]
    if len(tables) < 1:
        raise ValueError("need at least one power table")
    if any(t.ndim != 2 or t.shape[0] < 1 for t in tables):
        raise ValueError("every table must have shape (n_tilts, n_points) with n_tilts >= 1")
    n_points = tables[0].shape[1]
    if any(t.shape[1] != n_points for t in tables):
        raise ValueError(f"tables must share n_points, got {[t.shape[1] for t in tables]}")
    if not (np.isfinite(noise_mw) and noise_mw > 0):
        raise ValueError("noise_mw must be finite and > 0")
    # allocate the cube -> for every combination of tilt numbers, stack one row per sector,
    # best_server_sinr, SE = log2(1 + SINR) -> store into the cube.

    tilts_counts = [sector_table.shape[0] for sector_table in tables] # Gather the sum of tilts for each sector from tables
    cube_shape = tuple(tilts_counts) + (n_points,) # Build the final shape tuple by adding n_points at the end
    cube = np.empty(cube_shape)

    # Loop over every combination of tilts dynamically

    for idx in np.ndindex(*tilts_counts):
        selected_rows = [tables[s][idx[s]] for s in range(len(tables))] # Creates a list of 1D array of the length of n_points
        power_matrix = np.stack(selected_rows, axis=0) # Stack thes rows into array of shape (n_sectors, n_points)
        
        # Unpack the outputs: discard serving sector (_), keep sinr
        _, sinr = best_server_sinr(power_matrix, noise_mw)

        # Calculate Spectral Efficiency (SE)
        se_result = np.log2(1 + sinr)

        # Store the entire 1D array of points into the cube at coordinate idx
        cube[idx] = se_result

    return cube


def exclusion_mask(x, y, sectors, d_min: float = 35.0):
    """Boolean mask of the points where the radio model is valid (decision log D09).

    True where a point is at least ``d_min`` (horizontal metres) from **every** mast; False inside
    the near-tower exclusion region of any sector, where the empirical path-loss law is not valid.
    Same shape as ``x``.
    """
    x = np.asarray(x, dtype=float)
    y = np.asarray(y, dtype=float)
    if len(sectors) < 1:
        raise ValueError("need at least one sector")
    if not (np.isfinite(d_min) and d_min >= 0):
        raise ValueError("d_min must be finite and >= 0")
    nearest = np.min([to_sector_frame(x, y, s)[0] for s in sectors], axis=0)
    return nearest >= d_min


def field_maps(X, Y, sectors, tilts_e_deg, tilts_m_deg=None, d_min: float = 35.0,
               params: RFParams = DEFAULT_PARAMS):
    """:func:`sinr_field` on a grid, reshaped back to the grid and masked for plotting.

    Points inside the near-tower exclusion region (:func:`exclusion_mask`) are set to NaN in every
    map, so they are neither plotted nor aggregated by mistake.

    Returns a dict:
      ``power_dbm`` (n_sectors, *X.shape) received power of each sector (dBm; row j = sector j),
      ``serving``   X.shape, index of the serving sector — a **float** map, because NaN (masked)
                    cannot be stored in an integer array,
      ``sinr_db``   X.shape, SINR in dB,
      ``se``        X.shape, spectral efficiency log2(1 + SINR) in bps/Hz,
      ``valid``     X.shape, the boolean mask used.
    """
    X = np.asarray(X, dtype=float)
    Y = np.asarray(Y, dtype=float)
    if X.shape != Y.shape:
        raise ValueError(f"X and Y must share shape, got {X.shape} vs {Y.shape}")
    shape = X.shape
    f = sinr_field(X, Y, sectors, tilts_e_deg, tilts_m_deg, params)
    valid = exclusion_mask(X, Y, sectors, d_min)

    def to_map(v):
        return np.where(valid, np.asarray(v, dtype=float).reshape(shape), np.nan)

    power_dbm = np.where(valid, to_db(f["powers"]).reshape((len(sectors),) + shape), np.nan)
    return {"power_dbm": power_dbm, "serving": to_map(f["serving"]),
            "sinr_db": to_map(to_db(f["sinr"])), "se": to_map(f["se"]), "valid": valid}
