"""Tests for telecomopt.sweep (Stage 2b: the chained joint tilt sweep)."""
import numpy as np
import pytest

from telecomopt.coverage import weighted_mean, weighted_percentile
from telecomopt.network import Sector, sinr_field, to_sector_frame
from telecomopt.sweep import tilt_sweep

S1 = Sector(0.0, 0.0, 0.0)
S2 = Sector(500.0, 0.0, 180.0)
TILTS = [3.0, 6.0, 9.0]


@pytest.fixture(scope="module")
def small():
    return tilt_sweep([S1, S2], [TILTS, TILTS], nx=20, ny=16)


def test_tilt_sweep_shapes(small):
    n = 20 * 16
    assert small["x"].shape == small["y"].shape == small["weights"].shape == (n,)
    assert small["cube"].shape == (3, 3, n)
    assert small["mean_se"].shape == small["p5_se"].shape == (3, 3)


def test_tilt_sweep_matches_the_direct_field_and_1b_metrics(small):
    # independent route: sinr_field for one tilt pair, then the 1b aggregation on the same weights
    for i, j in [(0, 0), (1, 2), (2, 1)]:
        se = sinr_field(small["x"], small["y"], [S1, S2], [TILTS[i], TILTS[j]])["se"]
        assert np.isclose(small["mean_se"][i, j], weighted_mean(se, small["weights"]), rtol=1e-10)
        assert np.isclose(small["p5_se"][i, j], weighted_percentile(se, small["weights"], 5.0), rtol=1e-10)


def test_tilt_sweep_weights_are_cell_area_outside_the_tower_floor(small):
    cell = (500.0 / 20) * (500.0 / 16)
    near = np.minimum(to_sector_frame(small["x"], small["y"], S1)[0],
                      to_sector_frame(small["x"], small["y"], S2)[0]) < 35.0
    assert near.any()
    assert np.all(small["weights"][near] == 0.0)
    assert np.allclose(small["weights"][~near], cell)


def test_tilt_sweep_objectives_are_symmetric_in_the_two_tilts(small):
    assert np.allclose(small["mean_se"], small["mean_se"].T, rtol=1e-9)
    assert np.allclose(small["p5_se"], small["p5_se"].T, rtol=1e-9)


def test_tilt_sweep_respects_the_region_bounds():
    out = tilt_sweep([S1, S2], [[5.0], [5.0]], x_range=(0.0, 500.0), y_range=(-100.0, 300.0), nx=10, ny=8)
    assert out["y"].min() > -100.0 and out["y"].max() < 300.0
    assert out["cube"].shape == (1, 1, 80)


def test_tilt_sweep_needs_one_tilt_list_per_sector():
    with pytest.raises(ValueError):
        tilt_sweep([S1, S2], [TILTS], nx=4, ny=4)
