"""Tests for telecomopt.coverage (Stage 1b demand/aggregation)."""
import numpy as np

from telecomopt.coverage import (
    sample_wedge, sample_rectangle, to_xy, uniform_demand, hotspot_demand,
    weighted_mean, weighted_percentile, served_fraction,
    spectral_efficiency_field, coverage_report, non_dominated, sweep_metrics, weighted_cdf,
)
import pytest


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


def _brute_force_front(F):
    """Reference: the definition, one option at a time."""
    n = len(F); keep = np.ones(n, bool)
    for i in range(n):
        for j in range(n):
            if np.all(F[j] >= F[i]) and np.any(F[j] > F[i]):
                keep[i] = False
    return keep


def test_non_dominated_hand_worked():
    a = np.array([1.0, 2.0, 3.0, 2.0])
    b = np.array([3.0, 2.0, 1.0, 1.0])
    # (2, 1) is dominated by (2, 2): equal on a, better on b. The other three trade off.
    assert non_dominated(a, b).tolist() == [True, True, True, False]


def test_non_dominated_ties_are_kept_and_equal_better_dominates():
    assert non_dominated([1.0, 1.0], [2.0, 2.0]).tolist() == [True, True]     # exact duplicates
    assert non_dominated([1.0, 1.0], [2.0, 3.0]).tolist() == [False, True]    # equal a, better b


def test_non_dominated_single_objective_is_argmax_set():
    v = np.array([3.0, 5.0, 1.0, 5.0])
    assert non_dominated(v).tolist() == [False, True, False, True]


def test_non_dominated_matches_brute_force():
    rng = np.random.default_rng(1)
    for k in (2, 3):
        F = rng.integers(0, 6, size=(40, k)).astype(float)       # integers -> many ties
        assert np.array_equal(non_dominated(*F.T), _brute_force_front(F))


def test_non_dominated_keeps_grid_shape():
    t1, t2 = np.meshgrid(np.arange(4.0), np.arange(3.0))
    mask = non_dominated(t1, -t2)                    # maximize t1, minimize t2
    assert mask.shape == (3, 4)
    assert mask.sum() == 1 and mask[0, 3]            # unique best corner: t1 = 3, t2 = 0


@pytest.mark.parametrize("args", [(), ([1.0, 2.0], [1.0]), ([1.0, np.nan], [1.0, 2.0])])
def test_non_dominated_guards(args):
    with pytest.raises(ValueError):
        non_dominated(*args)


# ---- sample_rectangle (Stage 2b) -----------------------------------------------------------

def test_sample_rectangle_hand_worked():
    # [0, 4] x [0, 2] in 2 x 2 cells: dx = 2, dy = 1, centres x = 1, 3 and y = 0.5, 1.5; x-major order.
    x, y, area = sample_rectangle(0.0, 4.0, 0.0, 2.0, 2, 2)
    assert x.tolist() == [1.0, 1.0, 3.0, 3.0]
    assert y.tolist() == [0.5, 1.5, 0.5, 1.5]
    assert area.tolist() == [2.0, 2.0, 2.0, 2.0]


def test_sample_rectangle_shapes_and_total_area():
    x, y, area = sample_rectangle(0.0, 500.0, -250.0, 250.0, 100, 80)
    assert x.shape == y.shape == area.shape == (100 * 80,)
    assert np.isclose(area.sum(), 500.0 * 500.0)


def test_sample_rectangle_map_layout_is_x_major():
    nx, ny = 6, 4
    x, y, _ = sample_rectangle(0.0, 6.0, 0.0, 4.0, nx, ny)
    X, Y = x.reshape(nx, ny), y.reshape(nx, ny)
    assert np.allclose(X, X[:, [0]])                   # x is constant along axis 1 (y)
    assert np.allclose(Y, Y[[0], :])                   # y is constant along axis 0 (x)
    assert np.allclose(X[:, 0], np.arange(nx) + 0.5)
    assert np.allclose(Y[0, :], np.arange(ny) + 0.5)


def test_sample_rectangle_points_are_strictly_inside():
    x, y, _ = sample_rectangle(-100.0, 300.0, 20.0, 70.0, 40, 10)
    assert x.min() > -100.0 and x.max() < 300.0
    assert y.min() > 20.0 and y.max() < 70.0


def test_sample_rectangle_is_mirror_symmetric_about_the_midline():
    x_min, x_max = 0.0, 500.0
    x, y, area = sample_rectangle(x_min, x_max, -250.0, 250.0, 100, 100)

    def as_set(px, py):
        pts = np.round(np.column_stack([px, py]), 9)
        return pts[np.lexsort((pts[:, 1], pts[:, 0]))]

    assert np.allclose(as_set(x_min + x_max - x, y), as_set(x, y))     # x -> 500 - x
    assert np.allclose(as_set(x, -y), as_set(x, y))                    # y -> -y
    assert np.allclose(area, area[0])                                  # equal cells


def test_sample_rectangle_midline_sampled_only_for_odd_nx():
    mid = 250.0
    x_even, _, _ = sample_rectangle(0.0, 500.0, -250.0, 250.0, 100, 4)
    x_odd, _, _ = sample_rectangle(0.0, 500.0, -250.0, 250.0, 101, 4)
    assert not np.any(np.isclose(x_even, mid))
    assert np.any(np.isclose(x_odd, mid))


def test_sample_rectangle_midpoint_rule_integrals():
    # exact for functions linear in each variable ...
    x_min, x_max, y_min, y_max = -100.0, 300.0, -50.0, 150.0
    x, y, a = sample_rectangle(x_min, x_max, y_min, y_max, 37, 23)
    assert np.isclose((a * x).sum(), (x_max**2 - x_min**2) / 2 * (y_max - y_min))
    assert np.isclose((a * x * y).sum(),
                      (x_max**2 - x_min**2) / 2 * (y_max**2 - y_min**2) / 2)
    # ... and for x^2 the error is known: sum = integral - L * dx^2 / 12 (per-cell h^3/12 term).
    nx = 20
    x, y, a = sample_rectangle(0.0, 10.0, 0.0, 1.0, nx, 5)
    dx = 10.0 / nx
    assert np.isclose((a * x**2).sum(), 10.0**3 / 3 - 10.0 * dx**2 / 12)


def test_sample_rectangle_asymmetric_bounds_and_single_cell():
    x, y, a = sample_rectangle(-3.0, 5.0, 10.0, 12.0, 1, 1)
    assert (x.tolist(), y.tolist(), a.tolist()) == ([1.0], [11.0], [16.0])


@pytest.mark.parametrize("args", [
    (5.0, 5.0, 0.0, 1.0, 4, 4),          # empty in x
    (0.0, 1.0, 2.0, 1.0, 4, 4),          # reversed y
    (0.0, np.inf, 0.0, 1.0, 4, 4),       # non-finite
    (0.0, 1.0, 0.0, 1.0, 0, 4),          # no cells
    (0.0, 1.0, 0.0, 1.0, 4, 2.5),        # non-integer count
])
def test_sample_rectangle_guards(args):
    with pytest.raises(ValueError):
        sample_rectangle(*args)


# ---- sweep_metrics (Stage 2b) ---------------------------------------------------------------

def test_sweep_metrics_hand_worked():
    # two combinations, three points, weights 1:1:2
    #   row 0 SE [1, 2, 3]: mean (1 + 2 + 6) / 4 = 2.25, 5th pct = 1 (lowest value carries 25 % of weight)
    #   row 1 SE [4, 4, 10]: mean (4 + 4 + 20) / 4 = 7.0, 5th pct = 4
    cube = np.array([[1.0, 2.0, 3.0], [4.0, 4.0, 10.0]])
    out = sweep_metrics(cube, [1.0, 1.0, 2.0])
    assert set(out) == {"mean_se", "p5_se"}
    assert np.allclose(out["mean_se"], [2.25, 7.0])
    assert np.allclose(out["p5_se"], [1.0, 4.0])


def test_sweep_metrics_matches_the_1b_functions_entry_by_entry():
    rng = np.random.default_rng(5)
    cube = rng.uniform(0.5, 9.0, (3, 4, 60)); w = rng.uniform(0.1, 2.0, 60)
    out = sweep_metrics(cube, w)
    assert out["mean_se"].shape == out["p5_se"].shape == (3, 4)
    for i in range(3):
        for j in range(4):
            assert np.isclose(out["mean_se"][i, j], weighted_mean(cube[i, j], w))
            assert np.isclose(out["p5_se"][i, j], weighted_percentile(cube[i, j], w, 5.0))


def test_sweep_metrics_zero_weight_points_are_ignored():
    rng = np.random.default_rng(6)
    cube = rng.uniform(0.5, 9.0, (2, 2, 30)); w = rng.uniform(0.1, 2.0, 30)
    w[[3, 7, 20]] = 0.0                                   # e.g. points inside the D09 floor
    keep = w > 0
    full = sweep_metrics(cube, w)
    trimmed = sweep_metrics(cube[..., keep], w[keep])
    assert np.allclose(full["mean_se"], trimmed["mean_se"])
    assert np.allclose(full["p5_se"], trimmed["p5_se"])


def test_sweep_metrics_scale_of_weights_does_not_matter():
    rng = np.random.default_rng(7)
    cube = rng.uniform(0.5, 9.0, (2, 3, 25)); w = rng.uniform(0.1, 2.0, 25)
    a, b = sweep_metrics(cube, w), sweep_metrics(cube, 1000.0 * w)
    assert np.allclose(a["mean_se"], b["mean_se"]) and np.allclose(a["p5_se"], b["p5_se"])


def test_sweep_metrics_any_number_of_tilt_axes():
    cube = np.random.default_rng(8).uniform(1.0, 5.0, (2, 3, 2, 10))
    out = sweep_metrics(cube, np.ones(10))
    assert out["mean_se"].shape == (2, 3, 2)
    assert np.isclose(out["mean_se"][1, 2, 0], cube[1, 2, 0].mean())


@pytest.mark.parametrize("cube, w", [
    (np.ones(5), np.ones(5)),                              # no tilt axis
    (np.ones((2, 5)), np.ones(4)),                         # wrong number of weights
    (np.ones((2, 5)), -np.ones(5)),                        # negative weights
    (np.ones((2, 5)), np.zeros(5)),                        # no demand at all
])
def test_sweep_metrics_guards(cube, w):
    with pytest.raises(ValueError):
        sweep_metrics(cube, w)


# --- weighted_cdf -------------------------------------------------------------------------------

def test_weighted_cdf_hand_example():
    v, share = weighted_cdf(np.array([3.0, 1.0, 2.0]), np.array([1.0, 1.0, 2.0]))
    np.testing.assert_allclose(v, [1.0, 2.0, 3.0])
    np.testing.assert_allclose(share, [0.25, 0.75, 1.0])


def test_weighted_cdf_uniform_weights_is_the_plain_ecdf():
    v, share = weighted_cdf(np.array([5.0, 4.0, 3.0, 2.0, 1.0]), np.ones(5))
    np.testing.assert_allclose(v, [1.0, 2.0, 3.0, 4.0, 5.0])
    np.testing.assert_allclose(share, [0.2, 0.4, 0.6, 0.8, 1.0])


def test_weighted_cdf_is_monotone_and_ends_at_one():
    rng = np.random.default_rng(3)
    vals = rng.normal(size=500); w = rng.uniform(0.0, 2.0, 500)
    v, share = weighted_cdf(vals, w)
    assert v.shape == share.shape == (500,)
    assert np.all(np.diff(v) >= 0) and np.all(np.diff(share) >= -1e-15)
    assert np.isclose(share[-1], 1.0)


def test_weighted_cdf_matches_weighted_percentile():
    # the percentile is the first sorted value whose share reaches q / 100 (powers-of-two weights: exact sums)
    vals = np.array([4.0, 0.5, 2.5, 1.0, 3.0, 2.0, 5.0, 0.0])
    w = np.array([0.125, 0.25, 0.0625, 0.125, 0.125, 0.25, 0.03125, 0.03125])
    v, share = weighted_cdf(vals, w)
    for q in (5.0, 25.0, 50.0, 90.0):
        assert v[np.searchsorted(share, q / 100.0)] == weighted_percentile(vals, w, q)


def test_weighted_cdf_guards():
    with pytest.raises(ValueError):
        weighted_cdf(np.ones(3), np.ones(4))
    with pytest.raises(ValueError):
        weighted_cdf(np.ones(3), np.array([1.0, -1.0, 1.0]))
