"""Tests for telecomopt.rf.propagation.

Checks the physics against the reference worked examples so the module is verified against
the sources, not just internally consistent.

Reference:
    [A] 5G/6G Academy, "Antenna Tilt Optimization", Worked Example 1 (h=30 m).
"""

import math

import numpy as np
import pytest

from telecomopt.rf.propagation import (
    RFParams,
    DEFAULT_PARAMS,
    to_linear,
    to_db,
    path_loss_db,
    elevation_angle_deg,
    elevation_gain_db,
    path_gain_db,
    sinr,
    spectral_efficiency,
)


def test_defaults_match_ericsson_table_1():
    """Default params reproduce Ericsson Table I."""
    p = DEFAULT_PARAMS
    assert p.bs_height == 30.0
    assert p.ue_height == 1.5
    assert p.g0_dbi == 18.0
    assert p.hpbw_el == 6.5
    assert p.sll_el == -17.0
    assert p.p_dbm == 29.0
    assert p.n0_dbm == -111.0


def test_to_linear_to_db_roundtrip():
    for db in (-111.0, -17.0, 0.0, 29.0):
        assert to_db(to_linear(db)) == pytest.approx(db)


def test_geometric_tilt_matches_reference_example():
    """[A] Worked Example 1: arctan(30/250) = 6.84 deg."""
    # elevation_angle uses (bs_height - ue_height); the reference uses full height h=30,
    # so compare with ue_height set to 0 to match their arctan(h/d) convention.
    p = RFParams(ue_height=0.0)
    assert elevation_angle_deg(250.0, p) == pytest.approx(6.84, abs=0.05)


def test_elevation_gain_peak_at_beam_center():
    """Gain is 0 dB (relative) when the user's elevation equals the tilt."""
    assert elevation_gain_db(elevation_deg=8.0, tilt_deg=8.0) == pytest.approx(0.0)


def test_elevation_gain_matches_reference_cell_edge():
    """[A] Worked Example 1: at the 250 m cell edge (elev 6.84) with 8 deg tilt,
    A_V = -12*((6.84-8)/6.5)^2 = -0.38 dB."""
    g = elevation_gain_db(elevation_deg=6.84, tilt_deg=8.0)
    assert g == pytest.approx(-0.38, abs=0.02)


def test_elevation_gain_matches_reference_neighbor():
    """[A] Worked Example 1: toward the neighbour at 500 m (elev 3.43) with 8 deg tilt,
    A_V = -12*((3.43-8)/6.5)^2 = -5.93 dB."""
    g = elevation_gain_db(elevation_deg=3.43, tilt_deg=8.0)
    assert g == pytest.approx(-5.93, abs=0.05)


def test_elevation_gain_floored_at_sidelobe():
    """Far from the beam, gain never drops below the sidelobe floor."""
    g = elevation_gain_db(elevation_deg=80.0, tilt_deg=0.0)
    assert g == pytest.approx(DEFAULT_PARAMS.sll_el)


def test_path_loss_increases_with_distance():
    assert path_loss_db(500.0) > path_loss_db(100.0)


def test_sinr_improves_when_interferer_tilts_away():
    """Tilting an interferer's beam away from a served point should raise that point's SINR.

    Served point at 250 m from its own antenna (tilt 8 deg). One interferer 500 m away:
    compare it beaming near the point (low tilt, overshoot) vs steeply down (contained).
    """
    high_interf = sinr(250.0, 8.0, interferers=[(500.0, 3.0)])   # interferer overshoots
    low_interf = sinr(250.0, 8.0, interferers=[(500.0, 14.0)])   # interferer contained
    assert low_interf > high_interf


def test_spectral_efficiency_monotonic_in_sinr():
    assert spectral_efficiency(10.0) > spectral_efficiency(1.0) > spectral_efficiency(0.1)


def test_spectral_efficiency_shannon_value():
    """log2(1 + 1) = 1 bit/s/Hz at SINR = 1 (0 dB)."""
    assert spectral_efficiency(1.0) == pytest.approx(1.0)
