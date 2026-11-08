"""Tests for telecomopt.plots."""
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pytest

from telecomopt.plots import tilt_heatmap, grid_image, field_map
from telecomopt.coverage import sample_rectangle


def test_tilt_heatmap_puts_sector_1_on_x_and_marks_the_maximum():
    t1 = np.array([0.0, 1.0, 2.0, 3.0])
    t2 = np.array([10.0, 20.0, 30.0])
    values = np.zeros((4, 3)); values[2, 1] = 5.0           # peak at t1 = 2, t2 = 20
    fig, ax = plt.subplots()
    mesh = tilt_heatmap(ax, t1, t2, values, title="J")
    star = [ln for ln in ax.lines if ln.get_marker() == "*"][0]
    assert (star.get_xdata()[0], star.get_ydata()[0]) == (2.0, 20.0)
    assert mesh.get_array().size == values.size
    assert ax.get_title() == "J" and ax.get_xlabel().startswith("$t_1")
    plt.close(fig)


def test_tilt_heatmap_can_skip_the_marker():
    fig, ax = plt.subplots()
    tilt_heatmap(ax, [0.0, 1.0], [0.0, 1.0], np.eye(2), mark_argmax=False)
    assert not [ln for ln in ax.lines if ln.get_marker() == "*"]
    plt.close(fig)


def test_tilt_heatmap_shape_guard():
    fig, ax = plt.subplots()
    with pytest.raises(ValueError):
        tilt_heatmap(ax, [0.0, 1.0], [0.0, 1.0, 2.0], np.zeros((3, 2)))
    plt.close(fig)


# --- grid_image / field_map ---------------------------------------------------------------------

def test_grid_image_rows_are_y_and_columns_are_x():
    x, y, _ = sample_rectangle(0.0, 3.0, 0.0, 2.0, 3, 2)       # x-major: x = .5,.5,1.5,1.5,2.5,2.5
    img = grid_image(10.0 * x + y, 3, 2)
    assert img.shape == (2, 3)
    # image[j, i] = value of x-column i, y-row j  ->  10 * x_i + y_j
    np.testing.assert_allclose(img, [[5.5, 15.5, 25.5], [6.5, 16.5, 26.5]])


def test_grid_image_hand_example_and_float_output():
    img = grid_image(np.arange(6), 3, 2)
    np.testing.assert_array_equal(img, [[0.0, 2.0, 4.0], [1.0, 3.0, 5.0]])
    assert img.dtype == float


def test_grid_image_masks_invalid_cells():
    valid = np.array([True, False, True, True, True, True])
    img = grid_image(np.arange(6), 3, 2, valid)
    assert np.isnan(img[1, 0]) and np.isfinite(img).sum() == 5


def test_grid_image_guards():
    with pytest.raises(ValueError):
        grid_image(np.arange(5), 3, 2)
    with pytest.raises(ValueError):
        grid_image(np.arange(6), 3, 2, np.ones(5, dtype=bool))


def test_field_map_draws_the_grid_image_and_the_masts():
    fig, ax = plt.subplots()
    im = field_map(ax, np.arange(6), 3, 2, (0.0, 3.0, 0.0, 2.0), masts=[(0.0, 1.0), (3.0, 1.0)], cmap="magma")
    np.testing.assert_array_equal(im.get_array(), grid_image(np.arange(6), 3, 2))
    assert list(im.get_extent()) == [0.0, 3.0, 0.0, 2.0]
    assert len([ln for ln in ax.lines if ln.get_marker() == "^"]) == 2
    plt.close(fig)
