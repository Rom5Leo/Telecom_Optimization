# telecom-optimization (`telecomopt`)

Real-world **telecom network optimization** problems — the AT&T / QUBIT hackathon challenges and
their continuations — solved **classical-first**, with quantum methods (QAOA and friends) used as an
optional, honestly-benchmarked component via the
[`quantum-optimization`](https://github.com/Rom5Leo/quantum-optimization) library (`qcoptlib`).

> **Status: in build.** The antenna-tilt problem is being rebuilt from scratch, one antenna at a
> time, with every physics step verified against the reference papers. APIs and metrics may still
> shift, and nothing here is a production system.

The guiding principle: solve the problem with the best available tool. Classical methods do the
heavy lifting; quantum is applied where it is a credible fit and benchmarked honestly against a
classical baseline and brute force.

## Problems (staged)

| Problem | Status | Approach |
|---|---|---|
| Antenna tilt optimization | hackathon PoC + from-scratch rebuild in progress | RF/SINR spatial model → demand-weighted coverage/capacity → QUBO → QAOA + classical baseline |
| Field-technician dispatch (VRP) | planned | classical VRP (OR-Tools) + QAOA on the assignment core |
| Network traffic routing | planned | multi-commodity flow, classical + QAOA path selection |

### Antenna tilt — where it stands
Built stage by stage (single sector → + users → coupling → network), documented as it goes:
- **Stage 1a — single-sector spatial radio model (complete, verified against [E]):** geometry, path
  loss, the 2-D antenna pattern (elevation + azimuth, electrical + mechanical tilt), path gain, SNR,
  and spectral efficiency — a ground-plane map of one sector.
- **Stage 1b — demand + aggregation (in progress):** a demand field ρ(x,y), service thresholds, and
  ρ-weighted metrics (served fraction, 5th-percentile cell-edge, mean spectral efficiency) that turn
  the field into the numbers we optimize.
- **Stage 2–3 — interference, the network, and the QUBO/QAOA solve** against an honest classical
  benchmark, plus a *moving-frame* decomposition for scalability.

The full reasoning trail — every modelling decision, open question and lesson — lives in
`docs/antenna_tilt/` (`decision_log.md`, `domain_primer.md`), with the physics cited
equation-by-equation in `docs/references/`.

## What's inside

| Package | What it does |
|---|---|
| `telecomopt.rf` | Ericsson LTE propagation model — path loss, the 2-D antenna pattern (elevation + azimuth, electrical + mechanical tilt), path gain, SINR, spectral efficiency. Pure functions over an `RFParams` dataclass of [E] Table I constants; cited equation-by-equation. |
| `telecomopt.antenna` | The problem model — antenna layout and the constant **moving frame** (`frame_center`, `subnetwork_in_frame`, `move_frame`) that selects which antennas a local solve optimizes. |

## `hackathon-2026/`
The QUBIT × AT&T hackathon material for the antenna-tilt challenge — the proof of concept this
rebuild grew out of:
- `alternative-approach/` — this author's direct-SINR QAOA showcase (kept as a basis)
- `team-solution/` — the submitted team solution (git submodule → teammate's repo)
- `pitch/` — the slide deck and speaker notes

## Install & test

    poetry install
    poetry run pytest -v        # RF physics, 2-D antenna pattern, antenna model

## References
RF model grounded in: Ericsson (Athley & Johansson, 2010 — the primary source; its antenna pattern
is the 3GPP TR 36.814 model), Bell Labs (Eckhardt et al., 2011), and the 5G/6G Academy tilt guide.
PDFs plus an equation-to-code mapping in `docs/references/`.
