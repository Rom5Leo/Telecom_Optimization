"""Tests for the observation-frame additions to AntennaNetwork.

Frame selection is pure geometry (translation + a box test); these checks pin down that
moving the frame changes *which* antennas are in view without moving any antenna, and that a
frame subnetwork is a valid standalone network.
"""

import numpy as np
import pytest

from telecomopt.antenna import AntennaNetwork, antenna_tilt_qubo, decode_onehot


def test_frame_defaults_do_not_break_existing_api():
    """Adding frame fields must not change n / k / neighbors / qubo for existing usage."""
    rng = np.random.default_rng(6)
    net = AntennaNetwork(positions=rng.uniform(0, 1000, size=(3, 2)),
                         elec_options=(-10.0, -3.0, 3.0, 10.0))
    assert net.n == 3 and net.k == 4
    q = antenna_tilt_qubo(net)
    assert q.n == net.n * net.k


def test_in_frame_selection():
    net = AntennaNetwork(
        positions=[[0, 0], [400, 0], [900, 0], [0, 900]],
        frame_center=(0.0, 0.0), frame_half_width=500.0,
    )
    # frame is [-500,500]^2 -> only the first two antennas are inside
    assert net.in_frame_mask.tolist() == [True, True, False, False]
    assert net.antennas_in_frame.tolist() == [0, 1]


def test_local_positions_are_frame_relative():
    net = AntennaNetwork(positions=[[1200, 800]], frame_center=(1000.0, 700.0))
    assert net.local_positions[0].tolist() == pytest.approx([200.0, 100.0])


def test_move_frame_changes_view_not_positions():
    positions = np.array([[100.0, 100.0], [1200.0, 1500.0]])
    net = AntennaNetwork(positions=positions, frame_center=(0.0, 0.0), frame_half_width=500.0)
    assert net.antennas_in_frame.tolist() == [0]
    net.move_frame(1200.0, 1500.0)
    assert net.antennas_in_frame.tolist() == [1]
    # positions themselves are untouched
    assert net.positions.tolist() == positions.tolist()


def test_constant_frame_around_network_selects_all():
    """A frame centred on a [0,1000]^2 layout with hw=500 sees every antenna (the constant-frame setup)."""
    rng = np.random.default_rng(6)
    pos = rng.uniform(0, 1000, size=(3, 2))
    net = AntennaNetwork(positions=pos, frame_center=(500.0, 500.0), frame_half_width=500.0,
                         elec_options=(-10.0, -3.0, 3.0, 10.0))
    assert net.antennas_in_frame.tolist() == [0, 1, 2]


def test_subnetwork_in_frame_is_valid_and_maps_back():
    net = AntennaNetwork(
        positions=[[100, 100], [300, 100], [2000, 2000]],
        frame_center=(200.0, 100.0), frame_half_width=500.0,
        elec_options=(-10.0, -3.0, 3.0, 10.0),
    )
    local, ids = net.subnetwork_in_frame()
    assert ids.tolist() == [0, 1]          # the distant antenna is excluded
    assert local.n == 2
    q = antenna_tilt_qubo(local, beta=0.02)
    assert q.n == local.n * local.k
    # a decoded local config maps back onto the original antennas via ids
    best_bits, _ = q.brute_force()
    cfg = decode_onehot(best_bits, local)
    assert len(cfg) == len(ids)
