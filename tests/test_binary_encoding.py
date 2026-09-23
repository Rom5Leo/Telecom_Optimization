"""Tests for the compact (binary) tilt encoding.

Ground truth: the compact QUBO must reproduce exactly the objective the one-hot QUBO has on its
feasible (one-per-antenna) states, over half the qubits, with no one-hot penalty.
"""

from itertools import product

import numpy as np
import pytest

from telecomopt.antenna import (
    AntennaNetwork, antenna_tilt_qubo, antenna_tilt_qubo_compact,
    decode_onehot, decode_compact,
)


def _net(n=4, seed=1):
    rng = np.random.default_rng(seed)
    return AntennaNetwork(positions=rng.uniform(-500, 500, size=(n, 2)),
                          couple_range=1000.0, elec_options=(-10.0, -3.0, 3.0, 10.0))


def _oh_bits(cfg, k):
    b = np.zeros(len(cfg) * k, int)
    for i, c in enumerate(cfg):
        b[i * k + c] = 1
    return b


def _cp_bits(cfg, nbits):
    b = np.zeros(len(cfg) * nbits, int)
    for i, c in enumerate(cfg):
        for m in range(nbits):
            b[i * nbits + m] = (c >> m) & 1
    return b


def test_compact_uses_half_the_qubits():
    net = _net()
    assert antenna_tilt_qubo(net).n == net.n * net.k          # 4 qubits/antenna
    assert antenna_tilt_qubo_compact(net).n == net.n * 2      # 2 qubits/antenna (k=4)


def test_compact_matches_onehot_on_all_configs():
    net = _net(n=4)
    k = net.k
    qoh = antenna_tilt_qubo(net, beta=0.02)
    qc = antenna_tilt_qubo_compact(net, beta=0.02)
    for cfg in product(range(k), repeat=net.n):
        e_oh = qoh.energy(_oh_bits(cfg, k))
        e_cp = qc.energy(_cp_bits(cfg, 2))
        assert e_cp == pytest.approx(e_oh, abs=1e-9)


def test_compact_optimum_equals_onehot_feasible_optimum():
    net = _net(n=4)
    k = net.k
    qoh = antenna_tilt_qubo(net, beta=0.02)
    qc = antenna_tilt_qubo_compact(net, beta=0.02)
    oh_opt = min(product(range(k), repeat=net.n), key=lambda c: qoh.energy(_oh_bits(c, k)))
    cp_best = min(product((0, 1), repeat=qc.n), key=lambda x: qc.energy(x))
    assert tuple(decode_compact(cp_best, net)) == tuple(oh_opt)


def test_decode_compact_roundtrip():
    net = _net(n=5)
    cfg = [0, 3, 2, 1, 3]
    assert decode_compact(_cp_bits(cfg, 2), net) == cfg


def test_compact_requires_power_of_two():
    rng = np.random.default_rng(0)
    net = AntennaNetwork(positions=rng.uniform(-500, 500, size=(3, 2)),
                         elec_options=(-10.0, 0.0, 10.0))  # k=3, not a power of two
    with pytest.raises(ValueError):
        antenna_tilt_qubo_compact(net)


def test_compact_refuses_high_degree():
    rng = np.random.default_rng(0)
    net = AntennaNetwork(positions=rng.uniform(-500, 500, size=(3, 2)),
                         elec_options=tuple(np.linspace(-10, 10, 8)))  # k=8 -> b=3 -> cubic
    with pytest.raises(NotImplementedError):
        antenna_tilt_qubo_compact(net)
