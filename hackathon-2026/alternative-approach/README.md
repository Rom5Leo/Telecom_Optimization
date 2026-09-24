# Alternative approach — direct-SINR antenna-tilt (QAOA showcase)

This author's take on the QUBIT × AT&T antenna-tilt challenge: build the objective **directly from
SINR / spectral efficiency** (rather than an overlap-area surrogate), map it to a one-hot QUBO, and
solve it with QAOA against a brute-force classical baseline. The notebook —
`antenna_tilt_hackathon_showcase.ipynb` — is a thin walk-through; the RF physics lives in
`telecomopt.rf`, the problem model in `telecomopt.antenna`, and the QUBO/QAOA machinery in
[`qcoptlib`](https://github.com/Rom5Leo/Quantum_Optimization).

> **Status: in build (hackathon PoC kept as a basis).** It is an honest algorithmic proof of
> concept, not a production system — the point is the reasoning and the working pipeline.

## What it does, section by section
1. **The physics.** The Ericsson LTE model (`telecomopt.rf`) turns a tilt and a distance into
   SINR and spectral efficiency; a single-point sweep shows there is a genuine optimal tilt
   (the coverage-vs-interference trade-off). Cited equation-by-equation in `docs/references/`.
2. **Encoding.** Antennas in a 2-D world with a **constant observation frame** (a fixed square
   zone that selects which antennas are optimized — the fixed-frame use of the moving-frame idea in
   `docs/Ideas/`). One binary per (antenna, tilt); coverage as a linear reward, interference as a
   pairwise term, a one-hot penalty per antenna.
3. **Classical ground truth.** Brute force over tilt configs (`Kⁿ`, small) gives the true
   SINR optimum and the neutral baseline — the honest benchmark.
4. **QAOA.** The same QUBO solved on Qiskit/Aer via `qcoptlib`, read out best-of-samples, with
   COBYLA-native convergence and results cached to disk (`cached_qaoa`).
5. **How good is it, really? (§6b).** The gap to optimum is decomposed into two independent parts:
   **model fidelity** (the linear interference proxy is separable across antennas, so its own
   optimum reaches only ~86% of the true coupled-SINR optimum) and **QAOA optimization** (what the
   solver leaves on the table). Separating them is the honest reading of "% of optimum".
6. **One-hot vs compact encoding (§8).** A binary tilt encoding (`log₂K` qubits/antenna, no
   penalty) reproduces the *same* objective at **half the qubits** (24 → 12 for 4 tilts) with far
   fewer couplings — a much friendlier QAOA landscape. Includes a structural comparison and a
   multi-seed reliability benchmark. Finer tilt resolution (8+ levels) would need quadratization —
   flagged as future work.

## Run it
```
poetry install
poetry run jupyter lab      # open antenna_tilt_hackathon_showcase.ipynb
```
First run populates `qaoa_cache/`; re-runs load instantly, so edits to plots/markdown don't
recompute the (slow, ~24-qubit) QAOA solves.

## Relationship to the team solution
The submitted `team-solution/` (git submodule, credit Michael Morami) optimizes an exactly-quadratic
**coverage-minus-overlap** surrogate and reports SINR post-hoc; this alternative optimizes
**spectral efficiency from SINR directly** with a per-cell serving assumption to stay quadratic.
Same RF physics, complementary framings — see the parent `hackathon-2026/` notes.

## References
Ericsson (Athley & Johansson, 2010), Bell Labs (Eckhardt et al., 2011), 5G/6G Academy (2026) —
`../../docs/references/` (PDFs + equation mapping).
