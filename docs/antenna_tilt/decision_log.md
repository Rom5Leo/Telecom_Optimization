# Decision Log — Antenna Tilt Optimization

The audit trail for the antenna-tilt project: decisions (D##), open questions (OQ##), lessons
(L##). Seeded during the investigation phase; grows as the project develops.

> Status: **in build.** Investigation done (papers read, physics in `domain_primer.md`); build
> starts one antenna at a time.

## D00 — Scope & architecture
- Classical-first RF simulation + QUBO/QAOA optimization, quantum benchmarked honestly against a
  classical baseline and brute force. Physics in `telecomopt.rf`, problem model in
  `telecomopt.antenna`, QUBO/QAOA in `qcoptlib`. Notebooks thin; reasoning in `docs/`.
- The active AT&T challenge is antenna tilt; dispatch and routing follow in their own repos.

## D01 — RF model = Ericsson [E], cited equation-by-equation
- Choice: adopt [E]'s model verbatim — path loss `134 + 35·log₁₀(d_km)`, Gaussian elevation
  pattern `G_el` (Eq. 4, HPBW 6.5°, SLL −17 dB), path gain `G₀+G_el−PL`, SINR (Eq. 1), spectral
  efficiency `log₂(1+SINR)` (Eq. 2). Parameters from [E] Table I (already in `RFParams`).
- Reason: it is the primary source, validated in [E] against measured Kathrein patterns (~1 dB) and
  a dynamic simulator; every constant is citable. Mapping: `docs/references/RF_model_references.md`.
- Consequence: relative performance only (all constants are [E]'s defaults) — we report gains vs a
  baseline, not absolute dBm, exactly as [E] does.

## D02 — Objective & metrics
- Optimize a network-wide tilt configuration for a weighted objective; report the KPIs the brief
  names: SINR / spectral efficiency (coverage), throughput (capacity), and — later — handover proxy.
- Metric definitions from [E]: **coverage = 5-percentile** path gain (cell edge, noise-limited);
  **capacity = mean** spectral efficiency; **peak = 95-percentile**. Cell-edge is the most
  tilt-sensitive metric and the honest headline.
- Baseline = every antenna at its neutral/geometric tilt; ground truth = brute force on small
  instances (`Kⁿ`). Report the QAOA gap honestly (the alternative-approach showcase already does).

## D03 — Build it from scratch: field → users → coupling → users+coupling → many
Refined so users enter right after the single-antenna field, so every later metric is meaningful.
Two reusable concepts are named where they first appear: the **demand field** (1b) and
**best-server assignment** (2a).
- **Stage 1a — one antenna, the RF field.** Geometry (elevation angle), path loss, elevation gain,
  path gain, SNR, spectral efficiency as functions of position. "What does one antenna produce in
  space?" Verify each piece against [E].
- **Stage 1b — one antenna + users.** Introduce a **demand field ρ(x,y)** (user/traffic density);
  define the aggregate metrics carried everywhere after — coverage (served fraction / 5th-percentile
  edge) and mean throughput (capacity), all ρ-weighted. Where "coverage" becomes a number about
  people (OQ5).
- **Stage 2a — two antennas, fields interact.** **Best-server assignment (MAX over antennas)** +
  **interference (SUM of the rest)** → real SINR; the coupling and the overlap/handover region
  appear (OQ4).
- **Stage 2b — two antennas + users.** The demand-weighted joint objective; the mountain-vs-populated
  overlap trade-off (OQ5) becomes concrete; the joint tilt trade-off.
- **Stage 3 — N antennas + users.** Interference graph → QUBO (one-hot / compact) → classical
  baseline vs QAOA. Extra physics to verify as we scale (OQ2).
- Rationale: each stage is a focused step that earns the next; the physics is clearest small, and
  introducing users early sets up OQ4/OQ5 exactly where they bite.
- Refinement (from the 1a build): Stage 1a closes the **antenna model** fully before demand — beyond
  the boresight chain it also covers electrical-vs-mechanical tilt (D07) and the full 2-D pattern
  ([E] Eq. 5, azimuth), ending with a **2-D spatial radio model** over a ground plane. Stage 1b then
  adds the demand field ρ(x,y), service thresholds, and aggregation *on that 2-D model* — the 2-D map
  is chosen over a 1-D observation line because the demand field is itself 2-D.

## D04 — Encoding (from qcoptlib)
- One-hot per (antenna, tilt) with a one-hot penalty (`qcoptlib.qubo.onehot_penalty`), or the
  compact binary encoding (`antenna_tilt_qubo_compact`, half the qubits) — both already built and
  tested. Decision of which to feature per instance is deferred to Stage 3.

## D05 — Path loss uses horizontal distance (modeling convention); slant kept distinct
- Choice: feed the **horizontal** distance `d` into `PL(d) = 134 + 35·log₁₀(d_km)`, and let the
  elevation angle `β = arctan(Δh/d)` describe the geometry separately. Slant distance
  `√(d² + Δh²)` is kept as a distinct quantity, not substituted into `PL`.
- Reason: this is the **convention of the [E] macro model**, an empirical formula written in terms
  of horizontal (2D) cell-plane distance. Adopting it is a modeling choice, stated as such — the
  presence of a separate elevation term does not by itself establish which distance the propagation
  model requires; we follow the source's convention.
- Consequence: horizontal vs slant differ by `ΔPL = 35·log₁₀(√(d²+Δh²)/d)`, which for Δh = 28.5 m is
  ≈ **2.14 dB at 50 m** and ≈ **0.025 dB at 500 m** — material only in the near field the model is
  not meant to extrapolate into. Revisit if a near-tower regime ever matters.

## D06 — Uncertainty & statistical rigor (report distributions, not point values)
- Choice: treat every headline number as an estimate with a stated uncertainty, and keep **four
  distinct sources** separate (Monte-Carlo is a *sampling method*, not itself a Bayesian analysis):
  1. **Environmental variability** — the physical randomness the model represents (lognormal shadow
     fading, σ = 8 dB). Coverage/capacity are *distributions*; report their **percentiles** (the
     spread of outcomes over the population/realizations). This is why [E] uses the 5th/95th percentile.
  2. **Finite Monte-Carlo error** — numerical uncertainty in an *estimate* from only N realizations.
     For the mean, `SE[μ̂] ≈ s/√N`. A percentile interval of outcomes and a confidence interval for the
     mean answer **different questions**; do not conflate them. Caveat: one realization can be a whole
     spatially-correlated shadowing field — correlated points within it are *not* independent repeats,
     so N is the number of independent fields, not user points. σ = 8 dB is an input, not the error bar
     on the final KPI; it must be propagated through the model.
  3. **Parameter / model uncertainty** — limited knowledge of the modeled system (path-loss slope, the
     approximate pattern, the horizontal-vs-slant choice of D05). Treat these *first* as explicit
     alternative assumptions and sensitivity sweeps; assigning them Bayesian priors is possible but
     needs justification (with measurements D: `p(θ|D) ∝ p(D|θ)p(θ)`; without D it is only a *prior*
     predictive analysis).
  4. **Solver randomness** — optimizer seeds now, quantum measurement shots later. Report
     quantum-vs-classical as a distribution over seeds — but only after **defining "better"**
     (mean objective? reliability? time-to-target? under what compute budget?); seeds alone don't
     settle it, and a Bayesian comparison is one option, not automatically preferable.
- Variance-reduction to adopt from the start: **paired comparison under common conditions** — for each
  environmental realization i, evaluate both tilts and take `ΔK_i = K_i(t_A) − K_i(t_B)`. Holding the
  shadowing field fixed across the two settings separates the effect of tilt from the luck of the draw
  and sharpens the comparison.
- Explicitly *not* done: formal error propagation on the fixed [E] constants — results are reported
  **relative to a baseline** (D01), so absolute-value uncertainty is moot, and this is a simulation,
  not a measurement, so there is no instrumental error to propagate.
- Reason: it is both physically honest (the fading term makes KPIs random by construction) and a
  portfolio differentiator (most such projects report bare single numbers). Also feeds robustness:
  is the optimal tilt fragile to a perturbed demand map or a different fading draw? (ties to OQ5, and
  to why operators re-tune tilt on live traffic).

## D07 — Tilt: electrical vs mechanical, and where the single-tilt model is valid
- Forward (boresight) slice: mechanical downtilt `α_m` makes the elevation `α = α_m − β`, so [E] Eq. 4's
  mismatch `α + α_e = α_m + α_e − β` depends only on **total** tilt `t = α_e + α_m`. Electrical and
  mechanical are therefore interchangeable **there** (§5 confirms it: equal-sum splits give identical
  curves).
- Not interchangeable in general: mechanical tilt rotates the whole 3-D pattern (azimuth coupled),
  electrical shifts only the elevation term in antenna coordinates; off-boresight they diverge ([E]
  Eq. 5 + Fig. 2 coordinate transform). [E]: split ≤0.5 dB on coverage, matters for capacity (OQ3).
- Decision: `t` denotes electrical tilt with `α_m = 0` through §1–§4; §5 states the forward-slice
  equivalence and §6 adds the 2-D pattern so the distinction is visible. The notebook must not present
  the two mechanisms as interchangeable in general.

## D08 — Full 2-D pattern (Eq. 3 + Eq. 5) closes 1a with a spatial radio model
- Added to `telecomopt.rf`: `azimuth_gain_db` (Eq. 3, `max(-12(φ/HPBW_az)², SLL_az)`),
  `pattern_gain_db` (Eq. 5, `max{G_az(φ') + G_el(α'), SLL0}`), and `path_gain_2d_db`
  (`G0 + pattern − PL`). New `RFParams` fields from [E] Table I: `hpbw_az=65`, `sll_az=−25`,
  `sll0=−30`. Tests in `tests/test_propagation_2d.py`.
- Mechanical tilt = a coordinate rotation: the fixed ground direction (unit vector) is rotated into the
  panel frame about the horizontal y-axis by `−α_m`; electrical tilt shifts only the elevation term.
  This reproduces [E]'s stated behaviour: identical to electrical on boresight (both → total tilt),
  divergent off-boresight (verified ≈0.98 dB at az=40°, up to ~12 dB near the azimuth edge).
- Verification (tests, all green): azimuth defining points; forward slice reduces exactly to the 1-D
  `path_gain_db`; boresight total-tilt equivalence; off-boresight electrical≠mechanical; SLL0 floor.
- Consequence: 1a now ends with a 2-D **spatial radio model** (`path_gain_2d_db` over an (x,y) grid),
  the substrate 1b builds demand/thresholds/aggregation on (D03 refinement).

---

# Open Questions

## OQ8 — Constants as a config layer (FAHM-style)
- Now: all [E] Table I constants live in `RFParams` — a frozen, typed dataclass with inline provenance,
  passed explicitly to pure functions. That is already a single source of truth, and arguably safer for
  fixed physics constants than a loose config file (typed, refactor-safe, no hidden globals).
- Question: add a declarative config layer (e.g. `configs/*.toml` + `RFParams.from_toml`) as in FAHM?
  Its value appears when **scenarios multiply** — named parameter sets (dense-urban vs rural),
  reproducible experiment configs, tuning without code edits — which is Stage 3, not now (YAGNI today).
- Leaning: keep `RFParams` as the typed interface; add the TOML scenario layer when the first alternate
  scenario is needed (or now, if we want the FAHM pattern established early for portfolio consistency).

## OQ1 — Growing-frame idea ("integral derivation" / local-to-global assembly)
- **Idea:** solve small sub-zones with a few antennas each, then let an
  observation **frame grow**, adding more antennas and re-optimizing, assembling the global solution
  from local ones — like building an integral from local pieces.
- **Why it's promising:** it directly attacks the `Kⁿ` blow-up (QAOA qubit budget) by keeping each
  solve small; it matches the constant/moving **frame** already in `telecomopt.antenna`
  (`frame_center`, `subnetwork_in_frame`, `move_frame`) and the idea doc in `docs/Ideas/`.
- **The hard part to test:** interference is coupled *across* frame boundaries, so a locally optimal
  zone can be globally suboptimal (the boundary-interference limitation already flagged in the frame
  code). Questions: how much overlap between adjacent frames is needed? Fixed external interference
  vs including boundary antennas as fixed (non-optimized) neighbours? Does re-optimizing on frame
  growth converge, and to how close to the global optimum (measure vs brute force on medium N)?
- **Related methods to check:** domain decomposition, large-neighbourhood / rolling-horizon search,
  and clustering approaches in [BL]/[A] ("optimize in clusters of 7–19 sites"). Our growing frame is
  a principled, quantum-budget-aware version of that clustering.
- **Stakes:** if it holds, it's the scalability story that makes QAOA credible beyond toy sizes —
  the project's potential novelty. Test empirically once Stage 3 exists.

## OQ2 — Which objective, and how much physics to add
- Single cell-edge sample (current showcase) vs [BL]'s sampled-user objective with 5% quantile and
  10× edge weighting. Candidate extra physics to verify as we scale: genuine *coupled* interference
  in the QUBO (the current proxy is separable — see the alternative-approach §6b finding), shadow
  fading realizations, a handover/overshoot proxy, azimuth pattern (not just elevation).
- Stakes: model fidelity (the 86% ceiling in the showcase came from the separable proxy).

## OQ3 — Electrical / mechanical split
- [E]: the split barely affects coverage (≤0.5 dB) but matters for capacity (pure electrical best
  for edge/mean; even split for peak). Decide whether to model only total tilt (simpler) or expose
  the split as a second variable. Default: total tilt first; revisit if capacity metrics need it.

## OQ4 — Coverage is best-server (MAX), not per-antenna sum
- Per point, coverage comes from the **strongest** serving antenna (MAX over antennas); the others
  are interference (SUM in the SINR denominator). Network coverage = union of best-server cells.
- Our current QUBO models coverage as a **per-antenna reward** — a simplification of the best-server
  MAX, and the reason §6b found the interference proxy *separable* (~86% ceiling). Modelling true
  best-server coverage is a Stage-2/3 fidelity upgrade (it is non-quadratic, so it needs care:
  sampled users + max, or an auxiliary encoding). Stakes: closes the model-fidelity gap.

## OQ5 — Traffic / user-density weighting (the mountain-overlap case)
- Weight the objective by a demand map ρ(x,y): overlap or a hole over an empty mountain costs ≈0;
  over a populated block it costs a lot. Standard practice ("traffic-driven tilt optimization",
  "density-based CCO" — see `docs/references/external_literature.md`).
- Implementation: sample SINR/throughput over user points, multiply each by its weight → weighted
  coverage/interference coefficients in the QUBO. Pairs with OQ1 (grow the frame toward high-demand
  zones first). Stakes: makes the objective match what actually matters; strong portfolio angle.

## OQ6 — Real tower geometry as the layout (data source)
- Tower **locations** are open (OpenCelliD, FCC ASR, HIFLD); **tilt/azimuth/power/traffic are
  proprietary**. Plan: use real AT&T tower coordinates for one city as a realistic *layout*, then
  *simulate* the RF and tilt on top — "real geometry, simulated physics", stated as such. Refs in
  `external_literature.md` (mind OpenCelliD's CC-BY-SA license). Stakes: realism + a data-prep step.

## OQ7 — 2D now, 3D as an extension
- The model is 2D (ground plane, UE at 1.5 m), matching the papers and tractable. But tilt is a
  vertical control, so 3D changes the answer (upper floors, UAV corridors). Start 2D; log 3D as a
  "further work" extension (user height distribution / UAV placement — see `external_literature.md`).

---

# Lessons
*(carry over the FAHM habits: measure first, run the trivial baseline, verify against the source,
base-rate every claim, check a recipe's assumptions against the data.)*

## L01 — The path-gain maximum is not the beam-pointing distance
- The beam points at `d_beam = Δh/tan t`, but path gain `g = G0 + G_el − PL` peaks *nearer* the tower
  (≈153 m vs `d_beam`≈203 m at t=8°): at `d_beam` the elevation gain has zero slope while path loss is
  still rising, so `dg/dd|_{d_beam} = −35/(d_beam·ln10) < 0`. Named `d_beam`, not `d*` — a star implies
  an optimum it is not.
- Reading corollaries: the sharp corners in the path-gain / pattern curves are the model floor's `max()`
  slope discontinuity (≈101 m at t=8°), not a physical beam edge; and the near-tower rise is the
  empirical path-loss law extrapolated toward `d→0` where it is unphysical — the global argmax there is
  an artefact, not a prediction.
