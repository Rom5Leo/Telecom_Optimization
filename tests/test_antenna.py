"""Tests for telecomopt.antenna — the antenna-tilt problem builder."""
import numpy as np
import pytest
from telecomopt.antenna import AntennaNetwork, antenna_tilt_qubo, decode_onehot


def _net():
    rng = np.random.default_rng(6)
    return AntennaNetwork(positions=rng.uniform(0, 1000, size=(3, 2)),
                          elec_options=(-10.0, -3.0, 3.0, 10.0))


def test_network_shapes():
    net = _net()
    assert net.n == 3
    assert net.k == 4
    assert net.n * net.k == 12


def test_qubo_variable_count():
    net = _net()
    q = antenna_tilt_qubo(net)
    assert q.n == net.n * net.k


def test_decode_onehot_roundtrip():
    net = _net()
    # a valid one-hot config -> decode returns the chosen indices
    cfg = [0, 2, 3]
    bits = np.zeros(net.n * net.k, dtype=int)
    for i, k in enumerate(cfg):
        bits[i * net.k + k] = 1
    assert decode_onehot(bits, net) == cfg


def test_optimization_beats_baseline():
    """The brute-force optimum must beat the neutral baseline (there is something to optimize)."""
    from itertools import product
    net = _net()
    neutral = min(range(net.k), key=lambda k: abs(net.elec_options[k]))
    se_base = net.total_spectral_efficiency([neutral] * net.n)
    best = max(product(range(net.k), repeat=net.n), key=net.total_spectral_efficiency)
    se_opt = net.total_spectral_efficiency(list(best))
    assert se_opt > se_base


def test_qubo_optimum_aligns_with_se_optimum():
    """The QUBO's brute-force minimum should decode to a high-throughput config.

    (Not necessarily THE max-SE config, since the QUBO uses a linear interference proxy, but it
    must be clearly better than baseline — the encoding is sound.)
    """
    from itertools import product
    net = _net()
    q = antenna_tilt_qubo(net, beta=0.02)
    best_bits, _ = q.brute_force()
    cfg = decode_onehot(best_bits, net)
    neutral = min(range(net.k), key=lambda k: abs(net.elec_options[k]))
    se_base = net.total_spectral_efficiency([neutral] * net.n)
    assert net.total_spectral_efficiency(cfg) > se_base
