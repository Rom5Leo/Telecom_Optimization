# RF Model — Equations & Sources (cross-referenced to the notebook)

This sheet maps every calculation in `antenna_tilt_2d.ipynb` to the exact equation, table, or figure
in the three reference papers. Use it to answer "where does this come from?" in the pitch Q&A.

---

## Source papers
- **[E]** Ericsson — Athley & Johansson, *Impact of Electrical and Mechanical Antenna Tilt on LTE
  Downlink System Performance*, IEEE 2010. **(primary source)**
- **[BL]** Bell Labs — Eckhardt, Klein & Gruber, *Vertical Antenna Tilt Optimization for LTE Base
  Stations*, IEEE 2011.
- **[A]** 5G/6G Academy — Walia, *Antenna Tilt Optimization: Mechanical vs Electrical*, 2026.

---

## 1. Parameters (notebook Part 1) — from [E] Table I

Every constant is copied verbatim from Ericsson Table I ("Default parameter settings"):

| Code constant | Value | [E] Table I |
|---|---|---|
| BS_HEIGHT | 30 m | Base station height = 30 m |
| UE_HEIGHT | 1.5 m | Mobile height = 1.5 m |
| G0_dBi | 18 dBi | Antenna gain G₀ = 18 dBi |
| HPBW_EL | 6.5° | Elevation HPBW = 6.5° |
| SLL_EL | −17 dB | Elevation SLL = −17 dB |
| P_dBm | 29 dBm | eNB power per PRB P = 29 dBm |
| N0_dBm | −111 dBm | Noise power per PRB N₀ = −111 dBm |
| (path-loss) | 134 + 35·log₁₀(R) | Path loss = 134 + 35·log₁₀(R), R in km |
| (layout) | — | Intersite distance 500 m; 19 sites; 3 sectors/site |

---

## 2. Path loss (notebook Part 5, function 1) — [E] Table I

    path_loss_dB(d) = 134 + 35·log10(d_km)

Exponent 3.5 (the 35) models an obstructed macro environment. Contrast with free-space path loss
(FSPL = 20·log10(d) + 20·log10(f) + const, exponent 2), which is the form in the hand-sketch photo —
[E] deliberately uses the higher exponent for realism.

---

## 3. Antenna elevation pattern (notebook Part 5, function 3) — [E] Equation 4

The exact equation from the paper:

    G_el(α) = max[ −12·((α + α_e)/HPBW_el)² ,  SLL_el ]

- α = elevation angle to the point
- α_e = electrical downtilt (our "tilt")
- HPBW_el = 6.5°, SLL_el = −17 dB (Table I)
- The −12 constant is the 3GPP value making gain drop 3 dB at the half-power beamwidth edge.

[A] confirms this is the 3GPP TR 38.901 model: "A_V(θ) = −min(12·((θ − θ_tilt)/θ_3dB)², SLA_V)",
"used in all major planning tools (Atoll, ASSET, Planet)."

The full 2-D gain (we use only the elevation cut) is [E] Equation 5:

    G(α,φ) = max{ G_az(φ) + G_el(α), SLL0 } + G0

---

## 4. Path gain (notebook Part 5, function 4) — [E] Section II-A

[E] defines path gain as "antenna gain divided by path loss" → in dB:

    path_gain_dB(d, tilt) = G0 + G_el(elevation(d), tilt) − path_loss_dB(d)

---

## 5. SINR (notebook Part 6) — [E] Equation 1

The exact equation:

    SINR_n = ( P · g_{1,n} ) / ( Σ_{c=2}^{M} P · g_{c,n} + N0 )

- P = transmit power per PRB
- g_{1,n} = path gain from the SERVING antenna to user n
- Σ g_{c,n} = summed path gain from all INTERFERING antennas
- N0 = noise per PRB
- Computed in LINEAR units (powers add linearly, not in dB) — hence to_linear() in the code.

---

## 6. Spectral efficiency (notebook Part 6) — [E] Equation 2

    C_n = log2(1 + SINR_n)

[E]: "Motivated by Shannon's capacity formula, we approximate the spectral efficiency..."
[E] validated this simple model against a full dynamic system simulator (their Fig. 6) — the reason we
can justifiably use the closed form rather than LTE coding tables.

---

## 7. Utility / objective (notebook Part 6) — [BL] Equation 1  (fuller form, upgrade target)

Bell Labs' utility, weighting cell-edge users 10×:

    U = (1/P) · Σ_m [ w_avg · s_{m,avg} + w_edge · s_{m,edge} ]
    with w_avg = 1,  w_edge = 10

- s_{m,avg} = average spectral efficiency over sector m
- s_{m,edge} = 5% quantile (cell-edge) spectral efficiency
Our notebook currently sums per-antenna spectral efficiency at a single cell-edge point (a
simplification); the faithful version samples many user points to get s_avg and the 5% quantile, then
applies this weighting.

---

## 8. Geometric tilt / cell radius (notebook Part 2 baseline) — [A]

    θ_geo = arctan(h / d)          (geometric tilt to reach distance d)
    cell_radius = h / tan(θ)       (coverage radius for a given tilt)

[A] worked example: h = 30 m, cell edge 250 m → arctan(30/250) = 6.84°, + ~1.5° margin ≈ 8° total.
Basis for our BASELINE_TILT ≈ 6° and the ±10° electrical range.

[A] reference table (h = 30 m): 4°→429 m, 6°→286 m, 8°→213 m, 10°→170 m radius.

---

## 9. Neighbor/cluster graph (notebook Part 4) — [BL] Section II-B, Fig. 1 + [A]

- [BL]: "the definition of neighbors is based on the mutual interference of the sectors"; optimize
  cluster = center sector + its interfering neighbors.
- [A]: "Optimize in clusters of 7–19 sites simultaneously"; "optimizing one site in isolation" is a
  listed mistake because "fixing one site's interference creates a new problem at the neighbor."
Our distance-threshold (700 m vs the 500 m intersite distance) is a simplification of this clustering.

---

## 10. Classical baseline & why-quantum (notebook Part 7) — [BL] Section II-B, Fig. 2 + [E]

- Baseline: [BL] "a heuristic variant of the gradient ascent method", cluster-by-cluster, greedy.
  Their Fig. 2 = utility vs iteration climbing to convergence (our convergence plot mirrors this).
- Why quantum: classical local search "can miss better global configurations on complex networks"
  (the coupled-sectors argument). QAOA searches jointly.

---

## 11. Reported gains to quote (business impact) — [BL] Section IV + [A]

- [BL]: average spectral efficiency +~10%; cell-edge (5% quantile) +~100%; 46/57 sectors improved.
- [A] operator data — Vodafone Germany (12,000 sites): −15% interference, +8% avg throughput,
  +22% cell-edge throughput, ROI < 3 months. T-Mobile: +30% coverage from mMIMO tilt re-optimization.
- AT&T brief: ~$500M/yr operational savings + $600M/yr revenue uplift; CEO cites it as a quantum
  frontier "following AI."

---

## 12. Figures our visualization mirrors (notebook Part 8)

| Our panel | Reference figure |
|---|---|
| Network map (antennas + interference links) | [BL] Fig. 1 (sector map), Fig. 3/5 (best-server plots) |
| Convergence (throughput vs step) | [E] Fig. 2, [BL] Fig. 2 (utility vs iteration) |
| Tilt per antenna (before/after) | [BL] Fig. 4 (optimized per-sector tilts) |
| Per-antenna SINR (before/after) | [E] Fig. 6–7, [BL] Fig. 7 (per-sector spectral-eff gains) |

---

## 13. QAOA formulation (notebook Part 9) — challenge brief reading list

The brief itself cites the method: Farhi, Goldstone & Gutmann "A Quantum Approximate Optimization
Algorithm"; Lucas "Ising formulations of many NP problems"; Glover et al. "QUBO tutorial." The
quantum-classical split ("classical handles RF simulation... quantum takes the tilt configuration
across interacting sectors") is stated verbatim in the brief's antenna technical description.
