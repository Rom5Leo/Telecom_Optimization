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

---

# Open Questions

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
*(to be filled as we build — carry over the FAHM habits: measure first, run the trivial baseline,
verify against the source, base-rate every claim, check a recipe's assumptions against the data.)*
