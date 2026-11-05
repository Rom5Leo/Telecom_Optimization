"""Tests for telecomopt.plots."""
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pytest

from telecomopt.plots import tilt_heatmap


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
