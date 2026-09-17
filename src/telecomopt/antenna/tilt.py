"""Antenna-tilt optimization: turn an RF network into a QUBO.

Builds the coverage/interference coefficients from the RF model (:mod:`telecomopt.rf`) and
assembles them into a one-hot QUBO that a quantum or classical solver can optimize. This is
the problem-specific layer for the antenna-tilt challenge; the heavy lifting (RF physics, QUBO
container, QAOA) lives in the reusable libraries, so this module stays small.

Encoding: one binary per (antenna, tilt), ``v = i*K + k`` meaning antenna ``i`` uses tilt ``k``.
A one-hot penalty enforces exactly one tilt per antenna. Coverage is a per-antenna reward
(linear); interference is a pairwise coupling between neighbours (quadratic ZZ).
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from telecomopt.rf import RFParams, path_gain_db, to_linear, spectral_efficiency
from qcoptlib.qubo import QUBO


@dataclass
class AntennaNetwork:
    """A 2-D antenna network and its tilt option set.

    Attributes
    ----------
    positions   : (n, 2) antenna coordinates (m).
    couple_range: two antennas interfere if within this distance (m).
    baseline_tilt : mechanical baseline downtilt (deg); electrical adds to it.
    elec_options  : the electrical tilt choices (deg), e.g. [-10, -5, 0, 5, 10].
    cell_edge   : distance at which each antenna serves its own users (m).
    params      : RF parameters (Ericsson defaults if omitted).
    """

    positions: np.ndarray
    couple_range: float = 700.0
    baseline_tilt: float = 6.0
    elec_options: tuple[float, ...] = (-10.0, -5.0, 0.0, 5.0, 10.0)
    cell_edge: float = 200.0
    params: RFParams = None  # type: ignore

    def __post_init__(self) -> None:
        self.positions = np.asarray(self.positions, dtype=float)
        if self.params is None:
            self.params = RFParams()

    @property
    def n(self) -> int:
        return len(self.positions)

    @property
    def k(self) -> int:
        return len(self.elec_options)

    def distance(self, i: int, j: int) -> float:
        return float(np.hypot(*(self.positions[i] - self.positions[j])))

    @property
    def neighbors(self) -> list[tuple[int, int]]:
        return [(i, j) for i in range(self.n) for j in range(i + 1, self.n)
                if self.distance(i, j) <= self.couple_range]

    def total_tilt(self, k: int) -> float:
        return self.baseline_tilt + self.elec_options[k]

    # ---- RF-derived quantities ----

    def coverage(self, i: int, k: int) -> float:
        """Interference-free spectral efficiency of antenna i at tilt k (the coverage reward)."""
        sig = to_linear(self.params.p_dbm + path_gain_db(self.cell_edge, self.total_tilt(k), self.params))
        return spectral_efficiency(sig / to_linear(self.params.n0_dbm))

    def interference(self, i: int, j: int, l: int) -> float:
        """Linear interference power antenna j (at tilt l) leaks toward antenna i."""
        return to_linear(self.params.p_dbm + path_gain_db(self.distance(i, j), self.total_tilt(l), self.params))

    def total_spectral_efficiency(self, config) -> float:
        """Full-network throughput for a tilt config (one tilt index per antenna)."""
        total = 0.0
        nb = set(self.neighbors)
        for i in range(self.n):
            sig = to_linear(self.params.p_dbm + path_gain_db(self.cell_edge, self.total_tilt(config[i]), self.params))
            intf = sum(self.interference(i, j, config[j]) for j in range(self.n)
                       if j != i and (min(i, j), max(i, j)) in nb)
            total += spectral_efficiency(sig / (intf + to_linear(self.params.n0_dbm)))
        return total


def antenna_tilt_qubo(net: AntennaNetwork, beta: float = 0.02, penalty: float = 50.0) -> QUBO:
    """Assemble the one-hot QUBO for an :class:`AntennaNetwork`.

    Parameters
    ----------
    beta    : interference weight (the risk-return dial — higher penalizes overlap more).
    penalty : one-hot enforcement strength (must exceed the largest coverage swing).

    Variable ``v = i*K + k`` is 1 iff antenna i uses tilt k. Minimising the QUBO maximises
    coverage minus beta*interference, subject to one tilt per antenna.
    """
    n, k = net.n, net.k
    q = QUBO.zeros(n * k)

    for i in range(n):
        for kk in range(k):
            q.add_linear(i * k + kk, -net.coverage(i, kk))   # reward coverage (negated to minimize)
            q.add_linear(i * k + kk, -penalty)               # one-hot expansion: -P diagonal
        for kk in range(k):
            for ll in range(kk + 1, k):
                q.add_quadratic(i * k + kk, i * k + ll, 2.0 * penalty)  # +2P intra-antenna
        q.add_const(penalty)                                 # +P constant per antenna

    for (i, j) in net.neighbors:
        for kk in range(k):
            for ll in range(k):
                q.add_quadratic(i * k + kk, j * k + ll, beta * net.interference(i, j, ll))

    return q


def decode_onehot(bits, net: AntennaNetwork) -> list[int]:
    """Decode a bit vector to a tilt index per antenna (strongest bit per one-hot block)."""
    bits = np.asarray(bits, dtype=int)
    return [int(np.argmax(bits[i * net.k:(i + 1) * net.k])) for i in range(net.n)]
