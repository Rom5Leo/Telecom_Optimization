"""RF propagation model for antenna-tilt optimization.

Turns an antenna tilt angle and a distance into a received signal strength, then into
SINR and spectral efficiency. This is the physics layer used to build the coverage and
interference coefficients that the QUBO/QAOA layer optimizes over.

All formulas are taken from the Ericsson LTE reference and cross-checked against the
5G/6G Academy worked examples:

    [E]  Athley & Johansson, "Impact of Electrical and Mechanical Antenna Tilt on LTE
         Downlink System Performance", IEEE 2010.
    [A]  Walia, "Antenna Tilt Optimization: Mechanical vs Electrical", 5G/6G Academy, 2026.

Per-function provenance is given in each docstring. The parameter defaults are [E] Table I.
"""

from __future__ import annotations

from dataclasses import dataclass
import numpy as np


@dataclass(frozen=True)
class RFParams:
    """Radio-environment constants.

    Defaults are the Ericsson [E] Table I "Default parameter settings used in the
    simulations" — a realistic LTE macro-cell configuration. Bundling them in one frozen
    object (rather than module globals) keeps the physics functions pure: everything they
    depend on is passed in explicitly.

    Attributes
    ----------
    bs_height : antenna height above ground (m).            [E] Table I: 30 m
    ue_height : user-equipment height (m).                  [E] Table I: 1.5 m
    g0_dbi    : peak antenna gain (dBi).                     [E] Table I: 18 dBi
    hpbw_el   : elevation half-power beamwidth (deg).        [E] Table I: 6.5 deg
    sll_el    : elevation sidelobe level (dB, negative).     [E] Table I: -17 dB
    p_dbm     : transmit power per PRB (dBm).                [E] Table I: 29 dBm
    n0_dbm    : noise power per PRB (dBm).                   [E] Table I: -111 dBm
    pl_intercept : path-loss intercept (dB).                [E] Table I: 134
    pl_slope     : path-loss slope (dB per decade of km).   [E] Table I: 35
    hpbw_az   : azimuth half-power beamwidth (deg).          [E] Table I: 65 deg
    sll_az    : azimuth sidelobe level (dB, negative).       [E] Table I: -25 dB
    sll0      : overall pattern sidelobe floor (dB, neg).    [E] Table I: -30 dB
    """

    bs_height: float = 30.0
    ue_height: float = 1.5
    g0_dbi: float = 18.0
    hpbw_el: float = 6.5
    sll_el: float = -17.0
    p_dbm: float = 29.0
    n0_dbm: float = -111.0
    pl_intercept: float = 134.0
    pl_slope: float = 35.0
    hpbw_az: float = 65.0
    sll_az: float = -25.0
    sll0: float = -30.0


DEFAULT_PARAMS = RFParams()


def to_linear(db_value: float) -> float:
    """Convert a dB / dBm value to linear power.

    dB is logarithmic; powers must be added in linear units. Signal and interference are
    therefore summed in linear space (see :func:`sinr`) before any ratio is taken.
    """
    return 10.0 ** (db_value / 10.0)


def to_db(linear_value: float) -> float:
    """Convert a linear power back to dB."""
    return 10.0 * np.log10(linear_value)


def path_loss_db(distance_m: float, params: RFParams = DEFAULT_PARAMS) -> float:
    """Distance path loss in dB.

    Provenance: [E] Table I, ``134 + 35*log10(R)`` with R in km. The slope of 35 encodes a
    path-loss exponent of 3.5, modelling an obstructed macro environment (signals fade
    faster than the free-space exponent of 2).

    Parameters
    ----------
    distance_m : horizontal distance from antenna to user (m). Clamped at >= 1 m to avoid
                 ``log10(0)`` when a point coincides with the mast.
    """
    d_km = max(distance_m, 1.0) / 1000.0
    return params.pl_intercept + params.pl_slope * np.log10(d_km)


def elevation_angle_deg(distance_m: float, params: RFParams = DEFAULT_PARAMS) -> float:
    """Downward angle (deg) from the antenna to a ground user at horizontal distance.

    Pure geometry: ``arctan((h_bs - h_ue) / d)``. A close user is at a steep angle below
    horizon; a far user is near horizontal. This is the same relationship [A] calls the
    "geometric tilt", ``theta_geo = arctan(h / d)``.
    """
    height = params.bs_height - params.ue_height
    return np.degrees(np.arctan2(height, max(distance_m, 1.0)))


def elevation_gain_db(
    elevation_deg: float, tilt_deg: float, params: RFParams = DEFAULT_PARAMS
) -> float:
    """Antenna elevation-pattern gain (dB) toward a user, given the beam's downtilt.

    Provenance: [E] Equation 4 (the standard 3GPP parabolic vertical pattern; [E] cites TR 36.814 / TR 25.996):

        G_el(alpha) = max( -12 * ((alpha - tilt) / HPBW_el)^2 ,  SLL_el )

    The gain peaks (0 dB relative) when the user's elevation angle equals the tilt — the
    beam points straight at them — and falls off as a downward parabola, floored at the
    sidelobe level. **Tilt enters the whole model only here**: changing ``tilt_deg`` changes
    which users sit in the strong part of the beam. The -12 constant is the 3GPP value that
    makes the gain drop exactly 3 dB at the half-power-beamwidth edge.
    """
    mismatch = (elevation_deg - tilt_deg) / params.hpbw_el
    parabola = -12.0 * mismatch * mismatch
    return max(parabola, params.sll_el)


def path_gain_db(
    distance_m: float, tilt_deg: float, params: RFParams = DEFAULT_PARAMS
) -> float:
    """Total path gain (dB) from an antenna to a user: peak gain + beam gain - path loss.

    Provenance: [E] Section II-A defines path gain as "antenna gain divided by path loss",
    i.e. in dB, ``G0 + G_el - path_loss``. This is the received signal strength (before
    transmit power) at a user at ``distance_m`` from an antenna tilted at ``tilt_deg``.
    """
    el = elevation_angle_deg(distance_m, params)
    return params.g0_dbi + elevation_gain_db(el, tilt_deg, params) - path_loss_db(distance_m, params)


def sinr(
    serving_distance_m: float,
    serving_tilt_deg: float,
    interferers: list[tuple[float, float]],
    params: RFParams = DEFAULT_PARAMS,
) -> float:
    """Signal-to-interference-plus-noise ratio (linear) at a served point.

    Provenance: [E] Equation 1:

        SINR = (P * g_serving) / ( sum_c P * g_interferer_c  +  N0 )

    computed in linear power. ``interferers`` is a list of ``(distance_m, tilt_deg)`` for
    each interfering antenna as seen from the served point.

    Returns a linear ratio; use :func:`to_db` for dB or :func:`spectral_efficiency` for
    throughput.
    """
    signal = to_linear(params.p_dbm + path_gain_db(serving_distance_m, serving_tilt_deg, params))
    interference = sum(
        to_linear(params.p_dbm + path_gain_db(d, t, params)) for d, t in interferers
    )
    noise = to_linear(params.n0_dbm)
    return signal / (interference + noise)


def spectral_efficiency(sinr_linear: float) -> float:
    """Shannon spectral efficiency (bits/s/Hz) from a linear SINR.

    Provenance: [E] Equation 2, ``C = log2(1 + SINR)``, which [E] validate against a full
    dynamic system simulator (their Fig. 6) — the justification for using this closed form
    rather than LTE coding tables.
    """
    return float(np.log2(1.0 + sinr_linear))


def azimuth_gain_db(azimuth_deg: float, params: RFParams = DEFAULT_PARAMS) -> float:
    """Azimuth-pattern gain (dB) toward a user, [E] Equation 3:

        G_az(phi) = max( -12 * (phi / HPBW_az)^2 , SLL_az )

    Same parabola-with-floor as the elevation pattern, but horizontal: peak at boresight
    (phi = 0), -3 dB at +/- HPBW_az/2, floored at the azimuth sidelobe level.
    HPBW_az = 65 deg, SLL_az = -25 dB ([E] Table I).
    """
    # mirror elevation_gain_db, but in azimuth (no tilt term).
    mismatch = azimuth_deg / params.hpbw_az 
    return max(-12 * mismatch**2, params.sll_az)
    


def pattern_gain_db(
    depression_deg: float,
    azimuth_deg: float,
    tilt_e_deg: float = 0.0,
    tilt_m_deg: float = 0.0,
    params: RFParams = DEFAULT_PARAMS,
) -> float:
    """Full 2-D relative antenna pattern (dB), [E] Equation 5, with electrical + mechanical tilt:

        G(alpha, phi) = max{ G_az(phi') + G_el(alpha') , SLL0 }

    Electrical tilt shifts the elevation term only (identical in every azimuth). Mechanical
    tilt physically rotates the panel, so the panel-frame angles (alpha', phi') seen from a
    fixed ground direction change with azimuth -- which is why the two mechanisms differ
    off-boresight ([E] Fig. 2). On boresight (phi = 0) they collapse to the total tilt.

    Parameters
    ----------
    depression_deg : downward angle to the point (positive below horizon = elevation_angle_deg).
    azimuth_deg    : horizontal angle from the sector boresight.
    tilt_e_deg     : electrical downtilt.
    tilt_m_deg     : mechanical downtilt.
    """
    # ground-frame unit vector to the point (elevation positive UP, so a ground user is below: -depression)
    eps = np.radians(-depression_deg)
    phi = np.radians(azimuth_deg)
    ux = np.cos(eps) * np.cos(phi)
    uy = np.cos(eps) * np.sin(phi)
    uz = np.sin(eps)

    # mechanical downtilt = rotate this fixed ground direction into the panel
    # frame about the horizontal y-axis by -tilt_m. With z = radians(tilt_m_deg):
    # ux_p = ;  uz_p = 
    z = np.radians(tilt_m_deg)
    ux_p = np.cos(z)*ux - np.sin(z)*uz    # rotated x-component
    uz_p = np.sin(z)*ux + np.cos(z)*uz   # rotated z-component
    uy_p = uy    # y-axis is the rotation axis, unchanged

    # panel-frame elevation and azimuth of the point
    alpha_p = np.degrees(np.arcsin(np.clip(uz_p, -1.0, 1.0)))
    phi_p = np.degrees(np.arctan2(uy_p, ux_p))

    # elevation term (Eq 4) peaks at alpha_p = -tilt_e (electrical shifts elevation only)
    #  azimuth term (Eq 3) is azimuth_gain_db(phi_p). Combine per Eq 5, floored at SLL0.
    g_el = max(-12 * ((alpha_p + tilt_e_deg) / params.hpbw_el) ** 2, params.sll_el)
    g_az = azimuth_gain_db(phi_p, params)
    return max(g_az + g_el, params.sll0)


def path_gain_2d_db(
    distance_m: float,
    azimuth_deg: float,
    tilt_e_deg: float = 0.0,
    tilt_m_deg: float = 0.0,
    params: RFParams = DEFAULT_PARAMS,
) -> float:
    """2-D path gain (dB): G0 + 2-D pattern - path loss. Generalizes path_gain_db to azimuth."""
    beta = elevation_angle_deg(distance_m, params)
    return params.g0_dbi + pattern_gain_db(beta, azimuth_deg, tilt_e_deg, tilt_m_deg) - path_loss_db(distance_m)
