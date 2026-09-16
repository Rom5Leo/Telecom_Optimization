# QUBIT × AT&T Hackathon 2026 — Antenna Tilt Optimization

Team solution and analysis for the antenna down-tilt challenge, plus an alternative
modelling approach and a comparison of the two.

## Result

QAOA-optimized antenna tilts on a real RF model: coverage 46% → 69%, handover risk
cut ~6× (1.74% → 0.28%), with a one-hot vs binary encoding comparison showing a 50%
qubit saving.

## Contents

| Path | What |
|---|---|
| `team-solution/` | The submitted approach — overlap-area QUBO + encoding comparison. Credit: Michael Morami ([repo](https://github.com/MichaelMorami/Classiq-Hackathon-2026)) |
| `alternative-approach/` | A direct-SINR objective formulation (this author's) |
| `comparison.md` | Overlap-surrogate vs direct-SINR; one-hot vs binary encoding — the trade-offs |
| `pitch/` | Slide deck and speaker notes |

## The two approaches in one line

- **Team (overlap surrogate):** optimize an exactly-quadratic coverage-minus-overlap
  objective; report SINR post-hoc. Clean QUBO, avoids the non-quadratic best-server term.
- **Alternative (direct SINR):** optimize spectral efficiency built from SINR directly,
  with a per-cell serving assumption keeping it quadratic. Closer to the reference metric.

Both use the same RF physics (Ericsson LTE model) and agree qualitatively.
