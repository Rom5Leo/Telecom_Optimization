"""Antenna-tilt optimization: turn an RF network into a QUBO.

Builds the coverage/interference coefficients from the RF model (:mod:`telecomopt.rf`) and
assembles them into a one-hot QUBO that a quantum or classical solver can optimize. This is
the problem-specific layer for the antenna-tilt challenge; the heavy lifting (RF physics, QUBO
container, QAOA) lives in the reusable libraries, so this module stays small.

Encoding: one binary per (antenna, tilt), ``v = i*K + k`` meaning antenna ``i`` uses tilt ``k``.
A one-hot penalty enforces exactly one tilt per antenna. Coverage is a per-antenna reward
(linear); interference is a pairwise coupling between neighbours (quadratic ZZ).

Observation frame
-----------------
A network can be larger than the region you want to look at or optimize. An
:class:`AntennaNetwork` therefore carries a **square observation frame** — a 2*half-width
window in local coordinates centred at ``frame_center``. Antenna *positions never move*; the
frame just selects which antennas are currently in view (:attr:`antennas_in_frame`). For the
alternative approach we keep the frame **constant** (set once around the network), and
:meth:`move_frame` / :meth:`subnetwork_in_frame` are the hooks for the moving-frame extension
described in ``docs/Ideas/moving_frame_antenna_network.md``.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from telecomopt.rf import RFParams, path_gain_db, to_linear, spectral_efficiency
from qcoptlib.qubo import QUBO


@dataclass
class AntennaNetwork:
    """A 2-D antenna network, its tilt option set, and an observation frame.

    Attributes
    ----------
    positions   : (n, 2) antenna coordinates (m), in global coordinates.
    couple_range: two antennas interfere if within this distance (m).
    baseline_tilt : mechanical baseline downtilt (deg); electrical adds to it.
    elec_options  : the electrical tilt choices (deg), e.g. [-10, -5, 0, 5, 10].
    cell_edge   : distance at which each antenna serves its own users (m).
    params      : RF parameters (Ericsson defaults if omitted).
    frame_center : centre of the square observation frame (m), in global coordinates.
    frame_half_width : half the frame's side length (m); the frame spans
        ``[cx - hw, cx + hw] x [cy - hw, cy + hw]``. Default 500 m -> a 1 km x 1 km zone.
    """

    positions: np.ndarray
    couple_range: float = 700.0
    baseline_tilt: float = 6.0
    elec_options: tuple[float, ...] = (-10.0, -5.0, 0.0, 5.0, 10.0)
    cell_edge: float = 200.0
    params: RFParams = None  # type: ignore
    frame_center: tuple[float, float] = (0.0, 0.0)
    frame_half_width: float = 500.0

    def __post_init__(self) -> None:
        self.positions = np.asarray(self.positions, dtype=float)
        self.frame_center = np.asarray(self.frame_center, dtype=float)
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

    # ---- observation frame ----
    #
    # Translation preserves antenna-to-antenna distances, so moving the frame changes which
    # antennas are *in view*, never their physical positions or any RF calculation. Until a
    # local subnetwork is built (:meth:`subnetwork_in_frame`), ``n``, ``neighbors`` and the
    # QUBO still describe the ENTIRE network.

    @property
    def local_positions(self) -> np.ndarray:
        """Antenna positions relative to the current frame centre."""
        return self.positions - self.frame_center

    @property
    def in_frame_mask(self) -> np.ndarray:
        """Boolean mask: which antennas fall inside the current square frame."""
        return np.all(np.abs(self.local_positions) <= self.frame_half_width, axis=1)

    @property
    def antennas_in_frame(self) -> np.ndarray:
        """Global indices of the antennas inside the current frame."""
        return np.flatnonzero(self.in_frame_mask)

    @property
    def positions_in_frame(self) -> np.ndarray:
        """Local (frame-relative) positions of the antennas inside the current frame."""
        return self.local_positions[self.in_frame_mask]

    def move_frame(self, x: float, y: float) -> None:
        """Recentre the observation frame at global ``(x, y)`` (antennas do not move).

        The constant-frame approach never calls this; it is the entry point for the
        moving-frame extension (sweep a large network zone by zone).
        """
        self.frame_center = np.array([x, y], dtype=float)

    def subnetwork_in_frame(self) -> tuple["AntennaNetwork", np.ndarray]:
        """Build a standalone :class:`AntennaNetwork` from the antennas in the current frame.

        Returns ``(local_net, ids)`` where ``ids`` are the original global indices (keep them
        to map a decoded local solution back to the full network). The subnetwork inherits
        every setting except positions, and is re-centred on the frame so its own frame again
        covers it.

        Modeling limitation: the subnetwork ignores interference from antennas *outside* the
        frame. If boundary interference matters, account for it separately (e.g. as fixed
        external interference, or by including nearby outside antennas in the RF calculation
        without giving them optimization variables).
        """
        ids = self.antennas_in_frame
        local = AntennaNetwork(
            positions=self.positions[ids],
            couple_range=self.couple_range,
            baseline_tilt=self.baseline_tilt,
            elec_options=self.elec_options,
            cell_edge=self.cell_edge,
            params=self.params,
            frame_center=self.frame_center,
            frame_half_width=self.frame_half_width,
        )
        return local, ids

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


# --------------------------------------------------------------------------- #
# Compact (binary) tilt encoding.
#
# One-hot spends K qubits per antenna (one per tilt) and needs a penalty to keep exactly one
# lit. A *binary* encoding instead spends only b = log2(K) qubits per antenna: the tilt index
# is read off as an integer, t_i = sum_m 2^m x_{i,m}, so every bit pattern is a valid tilt and
# NO one-hot penalty is needed. For K = 4 that is 2 qubits/antenna instead of 4 — half the
# circuit width, and a cleaner QAOA landscape (no penalty terms).
#
# The catch, and the reason this is capped at K <= 4 here: a per-antenna cost f(t_i) expands as
# a multilinear polynomial in the b bits, of degree up to b. For b <= 2 that is at most
# quadratic — a genuine QUBO. For b >= 3 (K >= 8) it produces cubic-and-higher terms that a
# QUBO cannot hold directly; you would need *quadratization* (auxiliary qubits) to reduce the
# degree. That is the real cost of finer tilt resolution, and it is left as future work.
# --------------------------------------------------------------------------- #

def _multilinear_terms(values) -> dict[tuple[int, ...], float]:
    """Multilinear expansion of a function of ``b`` bits given its ``2**b`` values.

    ``values[t]`` is the function at the bit pattern whose integer is ``t`` (bit m has weight
    ``2**m``). Returns ``{subset_of_bit_indices: coefficient}`` such that
    ``f(x) = sum_S coeff_S * prod_{m in S} x_m`` exactly (Möbius transform over the subset
    lattice). A subset of size d is a degree-d term.
    """
    from itertools import combinations

    values = np.asarray(values, dtype=float)
    size = len(values)
    b = int(round(np.log2(size)))
    if 2 ** b != size:
        raise ValueError(f"values length {size} is not a power of two")

    def t_of(subset):  # integer index where exactly `subset` bits are 1
        return sum(1 << m for m in subset)

    terms: dict[tuple[int, ...], float] = {}
    for d in range(b + 1):
        for S in combinations(range(b), d):
            # c_S = sum_{U subset of S} (-1)^{|S|-|U|} f(t(U))
            c = 0.0
            for du in range(d + 1):
                for U in combinations(S, du):
                    c += ((-1) ** (d - du)) * values[t_of(U)]
            if abs(c) > 1e-12:
                terms[S] = c
    return terms


def antenna_tilt_qubo_compact(net: AntennaNetwork, beta: float = 0.02) -> QUBO:
    """Assemble the antenna-tilt QUBO with a **binary** tilt encoding (b = log2(K) qubits/antenna).

    Reproduces exactly the objective the one-hot :func:`antenna_tilt_qubo` has on its feasible
    (one-per-antenna) states — per-antenna coverage reward plus the linear interference proxy —
    but over ``net.n * log2(net.k)`` binary variables and with no one-hot penalty. Requires
    ``net.k`` to be a power of two and (so the result stays a genuine QUBO) at most 4.

    Variable layout: ``v = i*b + m`` is bit ``m`` of antenna ``i``; its tilt index is
    ``t_i = sum_m 2**m x_{i,m}``.
    """
    n, k = net.n, net.k
    b = int(round(np.log2(k)))
    if 2 ** b != k:
        raise ValueError(f"compact encoding needs k a power of two, got k={k}")
    if b > 2:
        raise NotImplementedError(
            f"k={k} needs b={b} bits/antenna, whose cost expansion is degree {b} (cubic+). "
            "A QUBO is degree 2 — reduce it with quadratization (auxiliary qubits). Left as future work."
        )

    # Per-antenna cost F_m(t): coverage reward + interference this antenna leaks as an interferer
    # (matching the one-hot builder, where pair (i,j) contributes beta*interference(i,j,t_j)).
    q = QUBO.zeros(n * b)
    for m in range(n):
        F = np.array([
            -net.coverage(m, t) + sum(beta * net.interference(i, m, t)
                                      for (i, j) in net.neighbors if j == m)
            for t in range(k)
        ])
        for S, coeff in _multilinear_terms(F).items():
            if len(S) == 0:
                q.add_const(coeff)
            elif len(S) == 1:
                q.add_linear(m * b + S[0], coeff)
            else:  # len == 2 (guaranteed by the b <= 2 check)
                q.add_quadratic(m * b + S[0], m * b + S[1], coeff)
    return q


def decode_compact(bits, net: AntennaNetwork) -> list[int]:
    """Decode a binary-encoded bit vector to a tilt index per antenna (``t_i = sum_m 2**m x_{i,m}``)."""
    bits = np.asarray(bits, dtype=int)
    b = int(round(np.log2(net.k)))
    return [int(sum((1 << m) * bits[i * b + m] for m in range(b))) for i in range(net.n)]
