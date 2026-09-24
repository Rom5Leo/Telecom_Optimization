# Domain Primer — Antenna Tilt Optimization (how AT&T's cellular network actually works)

Read this alongside the three reference papers ([E] Ericsson 2010, [BL] Bell Labs 2011,
[A] 5G/6G Academy 2026 — in `docs/references/`) and the 3GPP antenna model (TR 38.901). This
primer gives the working physical intuition our pipeline is built on: what the antennas do, how a
signal becomes coverage, and — the point Leo raised — the *full* problem space, not just
interference.

> Repo status: **in build.** This is the investigation phase (understand before modelling).

---

## 1. What the machine is

A cellular operator like AT&T covers the country with **macro base stations** (cell towers). A
typical tower is a **3-sector site**: three antenna panels at ~120° apart, each a *sector* serving
a wedge of the surrounding area. In the reference scenario there are 19 sites × 3 sectors on a
hexagonal grid, with an **inter-site distance (ISD) of 500 m** and antennas at **30 m** height
([E] Table I).

Each sector antenna is a **vertical panel array**: many radiating elements stacked vertically. That
stacking is what creates a *narrow* beam in the vertical (elevation) plane — the beam you tilt. The
key control knob is **downtilt**: how many degrees below the horizon the main beam points.

- **Too little downtilt** → the beam overshoots, energy spills far past the intended cell and lands
  in *neighbouring* cells as interference (and causes pilot pollution / poor cell isolation).
- **Too much downtilt** → the beam points into the ground close to the tower, the cell shrinks,
  and users at the edge lose signal (coverage holes, dropped calls, handover failures).

There is a genuine optimum in between — and because sectors interfere with each other, one sector's
best tilt depends on its neighbours'. That coupling is the whole problem.

## 2. How a signal becomes coverage (the physics chain)

For a user at horizontal distance `d` from an antenna, the received signal strength is built from
three pieces (all in dB), exactly as `telecomopt.rf` implements them from [E]:

1. **Path loss** — how much the signal weakens with distance:
   `PL(d) = 134 + 35·log₁₀(d_km)` ([E] Table I). The slope 35 encodes a **path-loss exponent 3.5**
   — an obstructed macro environment (signals fade faster than the free-space exponent of 2).
2. **Antenna elevation gain** — the beam's shape. The user sits at an elevation angle
   `α = arctan((h_bs − h_ue)/d)` below the antenna. The gain toward that angle is a Gaussian main
   beam with a sidelobe floor ([E] Eq. 4, the 3GPP TR 38.901 pattern):
   `G_el(α) = max( −12·((α + α_e)/HPBW_el)² , SLL_el )`, with **HPBW_el = 6.5°**,
   **SLL_el = −17 dB**, and `α_e` the electrical downtilt. Gain peaks when the beam points *at* the
   user (`α = −α_e`) and falls off as a downward parabola. **Tilt enters the entire model only
   here** — it decides which users sit in the strong part of the beam.
3. **Peak gain** `G₀ = 18 dBi`. Total **path gain** = `G₀ + G_el(α) − PL(d)` ([E] §II-A).

Multiply by transmit power `P = 29 dBm/PRB` to get received signal power. Noise floor is
`N₀ = −111 dBm/PRB`. There is also **lognormal shadow fading** (σ = 8 dB) — random attenuation from
buildings/terrain — which is why coverage is a *distribution*, not a single number.

## 3. Tilt mechanics — electrical vs mechanical

- **Mechanical tilt** — physically rotate the panel. The *effective* pattern seen from the ground
  changes with azimuth (the beam tilts down in front, less to the sides).
- **Electrical tilt (RET)** — a phase taper across the array elements steers the beam down
  electronically, **the same in every azimuth direction**. Remotely adjustable (Remote Electrical
  Tilt), so operators tune it on live networks. Limited to ≲10° before grating lobes appear ([E]).
- **Combined** — total tilt `α_tilt = α_e + α_m`; the ratio `r = α_e/α_tilt` is a design choice.

**What [E] found (the paper's core result):** total tilt strongly affects *both* coverage and
capacity; the electrical/mechanical *split* matters only for capacity (≤0.5 dB effect on coverage).
Pure electrical is best for cell-edge and mean throughput; peak throughput likes an even split.
Cell-edge performance is the most tilt-sensitive metric, and the optimal cell-edge tilt is about
*half* the peak-rate tilt. Practical baseline (from [A]): geometric tilt `θ_geo = arctan(h/d)`, so
h = 30 m to a 250 m edge → ~6.8°, plus margin → ~8° total; cell radius ≈ `h/tan(θ)`.

## 4. From signal to the numbers we optimize

- **SINR** ([E] Eq. 1): `SINR = P·g_serving / (Σ_c P·g_interferer,c + N₀)` — computed in **linear**
  power (powers add linearly, not in dB). The serving antenna is the one with the strongest path
  gain; every *other* antenna in range is an interferer.
- **Spectral efficiency** ([E] Eq. 2): `C = log₂(1 + SINR)` bps/Hz — Shannon, validated in [E]
  against a full dynamic simulator. This is our per-user throughput proxy.
- **Coverage vs capacity — two regimes**:
  - *Coverage* is **noise-limited** (cell edge): defined in [E] as the **5-percentile** path gain.
    It answers "does the weakest 5% of users have signal?"
  - *Capacity* is **interference-limited**: the **mean** spectral efficiency (cell throughput) and
    the **95-percentile** (peak rate).
  Tilt trades these against each other — the fundamental coverage/capacity tension.
- **Cell-edge weighting** ([BL] Eq. 1): the fuller objective weights edge users 10× the average
  (`w_avg=1, w_edge=10`), sampling many user points and using the average and the 5% quantile —
  our upgrade target beyond a single cell-edge sample.

## 5. The full problem space — *not just interference*

Interference is the headline, but AT&T's tilt problem is really a bundle of coupled objectives:

- **Overshoot & pilot pollution** — under-tilted cells throw energy into neighbours; users see many
  similar-strength cells, none dominant, so control channels degrade.
- **Coverage holes & cell shrinkage** — over-tilting leaves gaps between cells; edge users drop.
- **Handover / mobility failures** — the brief names this explicitly: too much downtilt → more
  handover failures near cell edges; ping-pong handovers waste signalling.
- **Cell-edge vs peak users** — the 5% and 95% of the throughput distribution want *different*
  tilts; you choose whom to favour.
- **Capacity & load balancing** — hot sectors (stadiums, downtown) may want tighter beams to pack
  capacity; tilt can shift load between cells.
- **Energy efficiency** — better SINR at lower power; tilt is one lever operators use to cut energy.
- **Dynamics** — traffic moves (rush hour, day/night, events), so the *optimal* tilt drifts over
  time. This is why the industry uses **SON / self-optimization** and, recently, **reinforcement
  learning** for Remote Electrical Tilt (see `docs/references/`). Our static optimization is the
  snapshot such systems solve repeatedly.
- **The coupling (the hard core)** — because changing one sector's tilt changes its neighbours'
  SINR, you cannot tune sectors independently; the network-wide search space is `K^N` (K tilt
  options, N sectors) and classical local search can miss the global optimum. This is what makes it
  a genuine combinatorial optimization problem — and the candidate for QAOA.

## 6. Why quantum, honestly

The value case ([A]/brief): AT&T estimates ~\$500M/yr operational savings + ~\$600M/yr revenue
uplift; operator trials report −15% interference, +8% average throughput, +22% cell-edge throughput
(Vodafone), ROI < 3 months. The *algorithmic* case: the problem is coupled combinatorial
optimization with a clean QUBO encoding — a fair QAOA target. The honest framing (ours throughout):
classical handles the RF simulation and validation; the quantum component takes the coupled tilt
choice; every quantum result is benchmarked against a classical baseline and brute force on small
instances. Recent literature confirms the direction — there is 2025 work on *quantum approaches to
large-scale wireless coverage optimization* (see references).

## 7. Interview vocabulary
- **Downtilt / RET** — beam angle below horizon; remotely-adjustable electrical tilt.
- **SINR** — signal / (interference + noise); the quality of a link.
- **Coverage (5%) vs capacity (mean/95%)** — cell-edge reliability vs throughput.
- **Cell isolation / overshoot** — how well a cell's energy stays in its own footprint.
- **HPBW / SLL** — half-power beamwidth / sidelobe level (the beam's shape).
- **Path-loss exponent** — how fast signal fades with distance (3.5 here).
- **Coupling** — a sector's SINR depends on neighbours' tilts → combinatorial, not separable.
- **CCO / SON** — coverage-capacity optimization / self-organizing networks.

## 8. Reading list (in order of value)
1. **[E] Ericsson 2010** — the RF model and the coverage/capacity/tilt findings (our physics).
2. `docs/references/RF_model_references.md` — every code equation mapped to its source.
3. **[BL] Bell Labs 2011** — the optimization objective (cell-edge weighting), clustering, baseline.
4. **[A] 5G/6G Academy 2026** — practitioner framing, mechanical-vs-electrical, ROI.
5. 3GPP TR 38.901 — the standardized antenna pattern (our `G_el`).
6. The CCO / RET-RL / quantum-coverage papers in `docs/references/README.md` — where the field is now.
