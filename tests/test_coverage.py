"""Tests for telecomopt.coverage (Stage 1b demand/aggregation)."""
import numpy as np

from telecomopt.coverage import (
    sample_wedge, to_xy, uniform_demand, hotspot_demand,
    weighted_mean, weighted_percentile, served_fraction,
    spectral_efficiency_field, coverage_report,
)


def test_sample_wedge_shape_and_area():
    dist, az, area = sample_wedge(35.0, 250.0, 60.0, 50, 41)
    assert dist.shape == az.shape == area.shape == (50 * 41,)
    assert dist.min() >= 35.0 and dist.max() <= 250.0
    assert az.min() >= -60.0 and az.max() <= 60.0
    # polar area elements sum ~ wedge area = (1/2) * (120 deg in rad) * (d_max^2 - d_min^2)
    expected = 0.5 * np.radians(120.0) * (250.0**2 - 35.0**2)
    assert np.isclose(area.sum(), expected, rtol=0.02)


def test_weighted_mean_and_uniform_equals_plain():
    v = np.array([1.0, 2.0, 3.0, 4.0])
    assert np.isclose(weighted_mean(v, np.ones_like(v)), v.mean())
    assert np.isclose(weighted_mean(v, [1, 0, 0, 1]), 2.5)


def test_weighted_percentile_known_cases():
    v = np.array([1.0, 2.0, 3.0, 4.0]); w = np.ones_like(v)
    assert weighted_percentile(v, w, 5.0) == 1.0     # bottom tail
    assert weighted_percentile(v, w, 50.0) == 2.0
    assert weighted_percentile(v, w, 100.0) == 4.0
    # weighting shifts the percentile: pile weight on the high end
    assert weighted_percentile(v, [1, 1, 1, 100], 50.0) == 4.0


def test_served_fraction():
    v = np.array([0.0, 1.0, 2.0, 3.0]); w = np.ones_like(v)
    assert np.isclose(served_fraction(v, w, 1.0), 0.75)   # 3 of 4 >= 1
    assert np.isclose(served_fraction(v, [10, 1, 1, 1], 1.0), 3.0 / 13.0)


def test_hotspot_demand_peaks_at_centre():
    x = np.array([0.0, 100.0]); y = np.array([0.0, 0.0])
    rho = hotspot_demand(x, y, [(0.0, 0.0)], [20.0], [5.0])
    assert np.isclose(rho[0], 5.0)          # at the centre -> peak
    assert rho[1] < 0.01                     # far away -> ~0


def test_coverage_report_keys_and_monotone_threshold():
    dist, az, area = sample_wedge(35.0, 250.0, 60.0, 40, 31)
    dem = uniform_demand(dist, az)
    rep = coverage_report(8.0, dist, az, dem, area, threshold=1.0)
    assert set(rep) == {"mean_se", "p5_se", "served_fraction"}
    assert 0.0 <= rep["served_fraction"] <= 1.0
    # a higher service bar can only lower (or hold) the served fraction
    rep_hi = coverage_report(8.0, dist, az, dem, area, threshold=5.0)
    assert rep_hi["served_fraction"] <= rep["served_fraction"]
    # p5 <= mean for a spread-out field
    assert rep["p5_se"] <= rep["mean_se"]


def test_demand_weighting_changes_the_optimum():
    # edge-heavy demand should prefer a lower (further-reaching) tilt than tower-heavy demand
    dist, az, area = sample_wedge(35.0, 250.0, 60.0, 50, 41)
    x, y = to_xy(dist, az)
    edge = hotspot_demand(x, y, [(230.0, 0.0)], [40.0], [5.0], floor=0.1)
    near = hotspot_demand(x, y, [(60.0, 0.0)], [40.0], [5.0], floor=0.1)
    tilts = np.arange(2.0, 16.0, 1.0)
    best_edge = tilts[np.argmax([coverage_report(t, dist, az, edge, area)["p5_se"] for t in tilts])]
    best_near = tilts[np.argmax([coverage_report(t, dist, az, near, area)["p5_se"] for t in tilts])]
    assert best_edge <= best_near   # edge demand -> shallower tilt reaches further out
