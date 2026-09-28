"""Tests for the 2-D antenna pattern (azimuth + mechanical tilt), telecomopt.rf.

Reference: [E] Athley & Johansson 2010 — Eq. 3 (azimuth), Eq. 5 (2-D gain), Table I,
and Fig. 2 (electrical vs mechanical tilt differ off-boresight).
"""
import numpy as np

from telecomopt.rf import (
    RFParams, DEFAULT_PARAMS, elevation_gain_db, path_gain_db,
    azimuth_gain_db, pattern_gain_db, path_gain_2d_db,
)

P = DEFAULT_PARAMS


def _beta(d, p=P):
    return np.degrees(np.arctan2(p.bs_height - p.ue_height, d))


def test_table1_azimuth_constants():
    assert P.hpbw_az == 65.0
    assert P.sll_az == -25.0
    assert P.sll0 == -30.0


def test_azimuth_pattern_defining_points():
    assert np.isclose(azimuth_gain_db(0.0), 0.0)                       # peak
    for edge in (-P.hpbw_az / 2, P.hpbw_az / 2):
        assert np.isclose(azimuth_gain_db(edge), -3.0)                # -3 dB at +/- HPBW/2
    assert np.isclose(azimuth_gain_db(150.0), P.sll_az)              # far off-beam -> floor
    assert azimuth_gain_db(20.0) == azimuth_gain_db(-20.0)          # symmetric


def test_forward_slice_reduces_to_elevation_model():
    for d in (50.0, 200.0, 500.0, 1000.0):
        b = _beta(d)
        assert np.isclose(pattern_gain_db(b, 0.0, 8.0, 0.0), elevation_gain_db(b, 8.0))
        assert np.isclose(path_gain_2d_db(d, 0.0, 8.0, 0.0), path_gain_db(d, 8.0))


def test_forward_slice_depends_only_on_total_tilt():
    b = _beta(200.0)
    ref = pattern_gain_db(b, 0.0, 8.0, 0.0)
    for ae, am in [(5.0, 3.0), (3.0, 5.0), (0.0, 8.0)]:
        assert np.isclose(pattern_gain_db(b, 0.0, ae, am), ref)


def test_electrical_and_mechanical_differ_off_boresight():
    b = _beta(200.0)
    for az in (30.0, 60.0):
        e_only = pattern_gain_db(b, az, 8.0, 0.0)
        m_only = pattern_gain_db(b, az, 0.0, 8.0)
        assert not np.isclose(e_only, m_only)


def test_pattern_floored_at_sll0():
    assert np.isclose(pattern_gain_db(80.0, 150.0, 0.0, 0.0), P.sll0)


def test_pattern_azimuth_symmetry():
    b = _beta(200.0)
    assert np.isclose(pattern_gain_db(b, 25.0, 8.0, 0.0), pattern_gain_db(b, -25.0, 8.0, 0.0))
