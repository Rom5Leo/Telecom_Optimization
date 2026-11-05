"""Tests for telecomopt.network (Stage 2a: best-server association and interference).

Ground truth used: hand-computed geometry, exact reduction to the single-sector model
(telecomopt.rf / telecomopt.coverage), hand-worked SINR arithmetic, and the analytic midpoint
crossing of the facing-sites layout (decision log D11).
"""
import numpy as np
import pytest

from telecomopt.rf import DEFAULT_PARAMS, RFParams, path_gain_2d_db
from telecomopt.coverage import sweep_metrics, sample_wedge, sample_rectangle, to_xy, spectral_efficiency_field
from telecomopt.network import (
    Sector, to_sector_frame, received_power_mw, best_server_sinr, sinr_field,
    exclusion_mask, field_maps, power_table, se_cube,
)

P = DEFAULT_PARAMS
N_MW = 10.0 ** (P.n0_dbm / 10.0)
S1 = Sector(0.0, 0.0, 0.0)        # site 1 at the origin, facing +x
S2 = Sector(500.0, 0.0, 180.0)    # site 2 at ISD = 500 m, facing -x (D11 layout)


# --- to_sector_frame -------------------------------------------------------------------------

def test_sector_frame_known_points_origin_sector():
    x = np.array([100.0, 0.0, 0.0, -100.0])
    y = np.array([0.0, 100.0, -50.0, 0.0])
    dist, az = to_sector_frame(x, y, S1)
    assert np.allclose(dist, [100.0, 100.0, 50.0, 100.0])
    assert np.allclose(az[:3], [0.0, 90.0, -90.0])        # ahead, left (+), right (-)
    assert np.isclose(abs(az[3]), 180.0)                  # directly behind


def test_sector_frame_facing_sector():
    # sector 2 faces -x: (400, 0) is 100 m straight ahead; (500, 100) is to its right (-90 deg)
    dist, az = to_sector_frame(np.array([400.0, 500.0]), np.array([0.0, 100.0]), S2)
    assert np.allclose(dist, [100.0, 100.0])
    assert np.allclose(az, [0.0, -90.0])


def test_sector_frame_wraps_and_keeps_shape():
    rng = np.random.default_rng(0)
    x = rng.uniform(-1000, 1000, (7, 9)); y = rng.uniform(-1000, 1000, (7, 9))
    dist, az = to_sector_frame(x, y, Sector(30.0, -20.0, 170.0))
    assert dist.shape == az.shape == (7, 9)
    assert np.all(az >= -180.0) and np.all(az < 180.0)


def test_sector_frame_shape_mismatch_raises():
    with pytest.raises(ValueError):
        to_sector_frame(np.zeros(3), np.zeros(4), S1)


# --- received_power_mw -----------------------------------------------------------------------

def test_received_power_reduces_to_path_gain_2d():
    # a sector at the origin facing +x: local frame == global polar, so S = 10^((P + g)/10)
    dist = np.array([50.0, 150.0, 240.0]); az = np.array([0.0, 30.0, -55.0])
    x, y = to_xy(dist, az)
    for te, tm in [(8.0, 0.0), (4.0, 3.0)]:
        got = received_power_mw(x, y, S1, te, tm)
        pg = np.array([path_gain_2d_db(d, a, te, tm) for d, a in zip(dist, az)])
        assert np.allclose(got, 10.0 ** ((P.p_dbm + pg) / 10.0), rtol=1e-12)


def test_received_power_uses_custom_params():
    custom = RFParams(p_dbm=40.0, g0_dbi=15.0)
    x, y = np.array([120.0]), np.array([10.0])
    ratio_db = 10 * np.log10(received_power_mw(x, y, S1, 8.0, params=custom)
                             / received_power_mw(x, y, S1, 8.0))
    assert np.isclose(ratio_db[0], (40.0 - 29.0) + (15.0 - 18.0))   # +11 dB power, -3 dB peak gain


def test_received_power_translation_rotation_invariance():
    # (400, 0) seen from sector 2 is the same geometry as (100, 0) seen from sector 1
    a = received_power_mw(np.array([400.0]), np.array([0.0]), S2, 8.0)
    b = received_power_mw(np.array([100.0]), np.array([0.0]), S1, 8.0)
    assert np.allclose(a, b, rtol=1e-12)


def test_received_power_keeps_grid_shape():
    X, Y = np.meshgrid(np.linspace(40, 200, 4), np.linspace(-50, 50, 3))
    assert received_power_mw(X, Y, S1, 8.0).shape == (3, 4)


# --- best_server_sinr ------------------------------------------------------------------------

def test_best_server_hand_worked_three_sectors():
    powers = np.array([[1.0, 5.0], [3.0, 1.0], [2.0, 1.0]])
    serving, sinr = best_server_sinr(powers, 0.5)
    assert serving.tolist() == [1, 0]
    assert np.allclose(sinr, [3.0 / 3.5, 5.0 / 2.5])     # serving / (sum of the others + N)


def test_best_server_single_sector_is_snr():
    powers = np.array([[2.0, 8.0, 0.5]])
    serving, sinr = best_server_sinr(powers, 0.25)
    assert serving.tolist() == [0, 0, 0]
    assert np.allclose(sinr, powers[0] / 0.25)


def test_best_server_equal_powers_is_below_one():
    # D11: S1 = S2 = S  =>  SINR = S / (S + N) < 1 strictly, approaching 1 when S >> N
    s = np.array([1e-3, 1.0, 1e3])
    serving, sinr = best_server_sinr(np.vstack([s, s]), 0.1)
    assert np.allclose(sinr, s / (s + 0.1))
    assert np.all(sinr < 1.0)
    assert serving.tolist() == [0, 0, 0]                  # exact tie -> lowest index


@pytest.mark.parametrize("powers, noise", [
    (np.array([1.0, 2.0]), 0.1),                 # 1-D: not (n_sectors, n_points)
    (np.array([[1.0, -2.0], [1.0, 1.0]]), 0.1),  # negative power
    (np.array([[1.0, np.nan]]), 0.1),            # non-finite power
    (np.array([[1.0, 2.0]]), 0.0),               # noise must be > 0
])
def test_best_server_guards(powers, noise):
    with pytest.raises(ValueError):
        best_server_sinr(powers, noise)


# --- sinr_field ------------------------------------------------------------------------------

def test_sinr_field_single_sector_matches_stage_1b():
    dist, az, _ = sample_wedge(35.0, 250.0, 60.0, 12, 11)
    x, y = to_xy(dist, az)
    f = sinr_field(x, y, [S1], [8.0])
    assert set(f) == {"powers", "serving", "sinr", "se"}
    assert f["powers"].shape == (1, dist.size)
    assert np.allclose(f["se"], spectral_efficiency_field(8.0, dist, az), rtol=1e-10)


def test_sinr_field_midpoint_equal_tilts():
    f = sinr_field(np.array([250.0]), np.array([0.0]), [S1, S2], [8.0, 8.0])
    s1, s2 = f["powers"][:, 0]
    assert np.isclose(s1, s2, rtol=1e-12)                       # symmetric layout
    assert np.isclose(f["sinr"][0], s1 / (s2 + N_MW))
    assert f["sinr"][0] < 1.0 and f["se"][0] < 1.0              # strictly below 0 dB / 1 bps/Hz


def test_sinr_field_second_midpoint_crossing():
    # D11: with t2 = 8, equal power at the midpoint also when t1 = 2*beta - 8 (main lobe, both on
    # boresight at equal distance), beta = arctan((h_bs - h_ue) / 250).
    beta = np.degrees(np.arctan((P.bs_height - P.ue_height) / 250.0))
    f = sinr_field(np.array([250.0]), np.array([0.0]), [S1, S2], [2 * beta - 8.0, 8.0])
    assert np.isclose(f["powers"][0, 0], f["powers"][1, 0], rtol=1e-9)


def test_sinr_field_mirror_symmetry():
    # equal tilts: mirroring x -> 500 - x swaps the serving sector and leaves the SINR unchanged
    x = np.array([120.0, 300.0, 410.0]); y = np.array([40.0, -80.0, 15.0])
    a = sinr_field(x, y, [S1, S2], [8.0, 8.0])
    b = sinr_field(500.0 - x, y, [S1, S2], [8.0, 8.0])
    assert (a["serving"] == 1 - b["serving"]).all()
    assert np.allclose(a["sinr"], b["sinr"], rtol=1e-12)


def test_sinr_field_tilt_count_guard():
    with pytest.raises(ValueError):
        sinr_field(np.array([1.0]), np.array([0.0]), [S1, S2], [8.0])


# --- exclusion_mask / field_maps ---------------------------------------------------------------

def test_exclusion_mask_distances_to_every_mast():
    x = np.array([10.0, 34.9, 35.0, 250.0, 470.0, 490.0])
    y = np.zeros_like(x)
    mask = exclusion_mask(x, y, [S1, S2], d_min=35.0)
    # 10 / 34.9 m from mast 1 -> excluded; exactly 35 m -> kept; 30 and 10 m from mast 2 -> excluded
    assert mask.tolist() == [False, False, True, True, False, False]


def test_exclusion_mask_keeps_shape_and_zero_floor():
    X, Y = np.meshgrid(np.linspace(0, 500, 6), np.linspace(-50, 50, 3))
    assert exclusion_mask(X, Y, [S1, S2]).shape == (3, 6)
    assert exclusion_mask(X, Y, [S1, S2], d_min=0.0).all()
    with pytest.raises(ValueError):
        exclusion_mask(X, Y, [S1, S2], d_min=-1.0)


def _grid():
    return np.meshgrid(np.linspace(0.0, 500.0, 11), np.linspace(-100.0, 100.0, 5))


def test_field_maps_matches_sinr_field_on_valid_points():
    X, Y = _grid()
    m = field_maps(X, Y, [S1, S2], [6.0, 9.0])
    f = sinr_field(X, Y, [S1, S2], [6.0, 9.0])
    v = m["valid"]
    assert m["power_dbm"].shape == (2,) + X.shape
    for key in ("sinr_db", "se", "serving"):
        assert m[key].shape == X.shape
    assert np.allclose(m["power_dbm"][:, v], 10 * np.log10(f["powers"].reshape((2,) + X.shape))[:, v])
    assert np.allclose(m["sinr_db"][v], 10 * np.log10(f["sinr"].reshape(X.shape))[v])
    assert np.allclose(m["se"][v], f["se"].reshape(X.shape)[v])
    assert np.array_equal(m["serving"][v], f["serving"].reshape(X.shape)[v].astype(float))


def test_field_maps_nan_exactly_where_masked():
    X, Y = _grid()                         # includes both mast positions (0, 0) and (500, 0)
    m = field_maps(X, Y, [S1, S2], [8.0, 8.0], d_min=35.0)
    v = m["valid"]
    assert not v.all() and v.any()
    for key in ("sinr_db", "se", "serving"):
        assert np.array_equal(np.isnan(m[key]), ~v)
    assert np.array_equal(np.isnan(m["power_dbm"]), np.broadcast_to(~v, m["power_dbm"].shape))


def test_field_maps_any_number_of_sectors():
    X, Y = _grid()
    m = field_maps(X, Y, [S1], [8.0])
    assert m["power_dbm"].shape == (1,) + X.shape
    assert np.all(m["serving"][m["valid"]] == 0.0)


# --- power_table (Stage 2b: separable tilt sweep) ---------------------------------------------

def _grid_2d():
    """A small 2-D grid of valid points between the two sites (shape (4, 5))."""
    X, Y = np.meshgrid(np.linspace(60.0, 440.0, 4), np.linspace(-120.0, 120.0, 5), indexing="ij")
    return X, Y


def test_power_table_shape_and_rows_equal_direct_calls():
    X, Y = _grid_2d()
    tilts = [2.0, 5.0, 8.0]
    table = power_table(X, Y, S1, tilts)
    assert table.shape == (3, X.size)                         # 2-D input is flattened to n_points
    assert np.all(np.isfinite(table)) and np.all(table > 0.0)
    for k, t in enumerate(tilts):
        assert np.allclose(table[k], received_power_mw(X.ravel(), Y.ravel(), S1, t), rtol=1e-12)


def test_power_table_rows_follow_the_order_of_the_tilts():
    X, Y = _grid_2d()
    forward = power_table(X, Y, S1, [3.0, 6.0, 9.0])
    backward = power_table(X, Y, S1, [9.0, 6.0, 3.0])
    assert np.allclose(forward, backward[::-1])
    assert not np.allclose(forward[0], forward[1])            # different tilts, different power


def test_power_table_rows_reproduce_the_joint_field_powers():
    # separability: the pair (t1, t2) needs only row t1 of sector 1's table and row t2 of sector 2's
    X, Y = _grid_2d()
    t1, t2 = 4.0, 9.0
    joint = sinr_field(X, Y, [S1, S2], [t1, t2])["powers"]
    stacked = np.stack([power_table(X, Y, S1, [t1, 7.0])[0], power_table(X, Y, S2, [3.0, t2])[1]])
    assert np.allclose(stacked, joint, rtol=1e-12)


def test_power_table_mirror_symmetry_of_the_facing_sites():
    # on a mirror-symmetric grid, sector 2's table is sector 1's table reflected x -> 500 - x
    nx, ny = 10, 6
    x, y, _ = sample_rectangle(0.0, 500.0, -250.0, 250.0, nx, ny)
    tilts = [2.0, 5.0, 8.0]
    t1 = power_table(x, y, S1, tilts).reshape(len(tilts), nx, ny)
    t2 = power_table(x, y, S2, tilts).reshape(len(tilts), nx, ny)
    assert np.allclose(t2, t1[:, ::-1, :], rtol=1e-9)


def test_power_table_passes_the_mechanical_tilt_through():
    X, Y = _grid_2d()
    table = power_table(X, Y, S1, [4.0, 8.0], tilt_m_deg=3.0)
    for k, t in enumerate([4.0, 8.0]):
        assert np.allclose(table[k], received_power_mw(X.ravel(), Y.ravel(), S1, t, 3.0), rtol=1e-12)
    assert not np.allclose(table, power_table(X, Y, S1, [4.0, 8.0]))   # mechanical tilt matters


@pytest.mark.parametrize("tilts", [[], [[1.0, 2.0]], [1.0, np.nan], [np.inf]])
def test_power_table_tilt_guards(tilts):
    with pytest.raises(ValueError):
        power_table(np.zeros(3) + 100.0, np.zeros(3), S1, tilts)


def test_power_table_shape_mismatch_raises():
    with pytest.raises(ValueError):
        power_table(np.zeros(3), np.zeros(4), S1, [5.0])


# --- se_cube (Stage 2b: SE of every tilt combination) ----------------------------------------

def test_se_cube_shape():
    rng = np.random.default_rng(0)
    cube = se_cube([rng.uniform(0.1, 5.0, (3, 12)), rng.uniform(0.1, 5.0, (4, 12))], 0.5)
    assert cube.shape == (3, 4, 12)


def test_se_cube_hand_worked_two_sectors():
    # two points, two tilts per sector, noise 0.5. SINR = max / (min + 0.5) at each point:
    #   pair (0,0): powers (1,2) and (5,2)    -> 2/1.5, 5/2.5
    #   pair (0,1): powers (1,0.5) and (5,4)  -> 1/1.0, 5/4.5
    #   pair (1,0): powers (3,2) and (1,2)    -> 3/2.5, 2/1.5
    #   pair (1,1): powers (3,0.5) and (1,4)  -> 3/1.0, 4/1.5
    t1 = np.array([[1.0, 5.0], [3.0, 1.0]])
    t2 = np.array([[2.0, 2.0], [0.5, 4.0]])
    cube = se_cube([t1, t2], 0.5)
    sinr = np.array([[[4 / 3, 2.0], [1.0, 5 / 4.5]],
                     [[1.2, 4 / 3], [3.0, 8 / 3]]])
    assert np.allclose(cube, np.log2(1.0 + sinr))


def test_se_cube_matches_the_direct_field_at_every_pair():
    X, Y = _grid_2d()
    tilts1, tilts2 = [3.0, 6.0, 9.0], [2.0, 8.0]
    cube = se_cube([power_table(X, Y, S1, tilts1), power_table(X, Y, S2, tilts2)], N_MW)
    assert cube.shape == (3, 2, X.size)
    for i, a in enumerate(tilts1):
        for j, b in enumerate(tilts2):
            assert np.allclose(cube[i, j], sinr_field(X, Y, [S1, S2], [a, b])["se"], rtol=1e-12)


def test_se_cube_mirror_symmetry_swaps_the_tilts():
    # equal tilt lists, mirror-symmetric grid: swapping (t1, t2) reflects the SE map in x
    nx, ny = 10, 6
    x, y, _ = sample_rectangle(0.0, 500.0, -250.0, 250.0, nx, ny)
    tilts = [2.0, 5.0, 8.0]
    cube = se_cube([power_table(x, y, S1, tilts), power_table(x, y, S2, tilts)], N_MW)
    maps = cube.reshape(3, 3, nx, ny)
    for i in range(3):
        for j in range(3):
            assert np.allclose(maps[i, j], maps[j, i, ::-1, :], rtol=1e-9)


def test_se_cube_any_number_of_sectors():
    rng = np.random.default_rng(3)
    tabs = [rng.uniform(0.1, 5.0, (n, 6)) for n in (2, 3, 2)]
    cube = se_cube(tabs, 0.5)
    assert cube.shape == (2, 3, 2, 6)
    _, sinr = best_server_sinr(np.stack([tabs[0][1], tabs[1][2], tabs[2][0]]), 0.5)
    assert np.allclose(cube[1, 2, 0], np.log2(1.0 + sinr))


def test_se_cube_one_sector_is_snr():
    p = np.array([[1.0, 2.0, 4.0], [0.5, 8.0, 3.0]])
    cube = se_cube([p], 0.25)
    assert cube.shape == (2, 3)
    assert np.allclose(cube, np.log2(1.0 + p / 0.25))


@pytest.mark.parametrize("tables, noise", [
    ([], 0.5),                                           # no sectors
    ([np.ones(5)], 0.5),                                 # not 2-D
    ([np.ones((2, 5)), np.ones((2, 4))], 0.5),           # different n_points
    ([np.ones((2, 5))], 0.0),                            # no noise
    ([np.ones((2, 5))], np.nan),
])
def test_se_cube_guards(tables, noise):
    with pytest.raises(ValueError):
        se_cube(tables, noise)


# --- the Stage 2b symmetry check: J(t1, t2) = J(t2, t1) --------------------------------------

def test_joint_objectives_are_symmetric_in_the_two_tilts():
    # region, grid, demand (uniform) and the D09 mask are all mirror-symmetric about x = 250
    nx, ny = 20, 20
    x, y, area = sample_rectangle(0.0, 500.0, -250.0, 250.0, nx, ny)
    weights = area * exclusion_mask(x, y, [S1, S2])
    tilts = np.arange(2.0, 11.0, 1.0)
    cube = se_cube([power_table(x, y, S1, tilts), power_table(x, y, S2, tilts)], N_MW)
    out = sweep_metrics(cube, weights)
    for key in ("mean_se", "p5_se"):
        assert np.allclose(out[key], out[key].T, rtol=1e-9, atol=1e-12)
    assert not np.allclose(out["mean_se"], out["mean_se"][0, 0])    # the landscape is not flat
