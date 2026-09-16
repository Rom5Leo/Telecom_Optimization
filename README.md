# telecom-optimization (`telecomopt`)

Real-world **telecom network optimization** problems — the AT&T / QUBIT hackathon challenges
and their continuations — solved **classical-first**, with quantum methods used as an optional
component via the [`quantum-optimization`](https://github.com/Rom5Leo/quantum-optimization)
library.

The guiding principle: solve the problem with the best available tool. Classical methods do
the heavy lifting; quantum (QAOA and friends) is applied where it is a credible fit and
benchmarked honestly against the classical baseline.

## Problems (staged)

| Problem | Status | Approach |
|---|---|---|
| Antenna tilt optimization | hackathon PoC → in progress | RF/SINR model → QUBO → QAOA + classical baseline |
| Field-technician dispatch (VRP) | planned | classical VRP (OR-Tools) + QAOA on the assignment core |
| Network traffic routing | planned | multi-commodity flow, classical + QAOA path selection |

## What's inside

| Package | What it does |
|---|---|
| `telecomopt.rf` | Ericsson LTE propagation model — path loss, antenna elevation pattern, SINR, spectral efficiency. Domain physics, cited to the reference papers. |
| *(more per problem, added as each is built)* | |

## `hackathon-2026/`
The QUBIT × AT&T hackathon material for the antenna-tilt challenge:
- `alternative-approach/` — this author's SINR-based approach (the hackathon effort, kept as a basis)
- `team-solution/` — the submitted team solution (git submodule → teammate's repo)
- `pitch/` — the slide deck and speaker notes

## Install

    poetry install
    poetry run pytest -v        # 11 tests (RF model)

## References
RF model grounded in: Ericsson (Athley & Johansson, 2010), Bell Labs (Eckhardt et al., 2011),
and the 5G/6G Academy tilt guide. See `docs/references/`.
