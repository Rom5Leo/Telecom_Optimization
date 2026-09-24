# References — Antenna-Tilt Optimization

Source papers for the antenna-tilt problem (QUBIT × AT&T Hackathon 2026, Challenge 1), kept here
so every equation in `telecomopt.rf` / `telecomopt.antenna` traces to a citable source. See
`RF_model_references.md` for the equation-by-equation mapping (code line → paper table/equation).

> Repo status: **in build.** These references anchor the RF physics and the optimization framing;
> the deep per-paper review notes are being written stage by stage (see `docs/decision_log.md`).

## The three papers

| Cite | File | Paper | Role |
|---|---|---|---|
| **[E]** | `E_Ericsson2010_tilt_LTE_downlink.pdf` | Athley & Johansson, *Impact of Electrical and Mechanical Antenna Tilt on LTE Downlink System Performance*, IEEE 2010 | **Primary source.** Table I parameter defaults; Eq. 4 elevation pattern; Eq. 1 SINR; Eq. 2 spectral efficiency. Everything in `rf/propagation.py`. |
| **[BL]** | `BL_BellLabs2011_vertical_tilt_optimization.pdf` | Eckhardt, Klein & Gruber, *Vertical Antenna Tilt Optimization for LTE Base Stations*, IEEE 2011 | The **optimization** paper: cell-edge-weighted utility (Eq. 1, edge ×10), neighbor/cluster graph, greedy gradient-ascent baseline, reported gains (+~10% avg, +~100% cell-edge). Upgrade target for our objective. |
| **[A]** | `A_5G6GAcademy2026_mechanical_vs_electrical.pdf` | Walia, *Antenna Tilt Optimization: Mechanical vs Electrical, Coverage vs Interference*, 5G/6G Academy 2026 | Practitioner framing: geometric tilt & cell-radius table, mechanical-vs-electrical, cluster-optimization pitfalls, operator ROI figures (Vodafone, T-Mobile). |

## The problem (from the brief)
Choose a discrete tilt per sector so the network-wide configuration maximizes a weighted objective
(coverage / SINR / throughput, energy) while minimizing interference, overshoot and outage.
Sectors are **strongly coupled** — one sector's tilt changes its neighbors' SINR — so it is a
large coupled combinatorial problem. KPIs: SINR improvement, user throughput, drop/handover-failure
reduction. Hybrid framing: classical handles RF simulation + validation; the quantum component
chooses the coupled tilt configuration.

## Related method references (from the brief, for the QUBO/QAOA layer)
Farhi–Goldstone–Gutmann (QAOA); Lucas, *Ising formulations of many NP problems*; Glover–Kochenberger–Du,
*A tutorial on formulating and using QUBO models*. (Already reflected in `qcoptlib`.)

## Further reading — the field beyond the three papers
A running, annotated bibliography (CCO & traffic-weighting, SON/RET-via-learning, the quantum angle,
tower-location datasets, and 3D/UAV coverage) lives in **`external_literature.md`**, each entry with
a one-line "why-for-us". Add to it as stages need new sources.

## How to find more articles (method, reusable for every problem)
1. **Mine the papers you have.** Follow [E]/[BL]/[A]'s reference lists backward (foundational work)
   and use Google Scholar's "Cited by" forward (who built on them) — this is the fastest way to the
   canonical chain.
2. **Search by the problem's real names,** not the hackathon phrasing: "antenna tilt optimization",
   "coverage capacity optimization (CCO)", "remote electrical tilt (RET)", "self-organizing
   networks (SON)", "cell shaping". Add a method ("reinforcement learning", "QUBO", "QAOA") or an
   era ("5G", "6G", "2024..2026") to narrow.
3. **Primary venues:** IEEE Xplore (VTC, GLOBECOM, ICC, TWC), arXiv (cs.NI, quant-ph), 3GPP TRs
   (38.901 for the antenna/channel model). Vendor research (Ericsson, Nokia/Bell Labs) is gold for
   realistic models and numbers.
4. **Standards & data:** 3GPP technical reports for models/parameters; operator whitepapers for the
   business figures (the ROI numbers in [A]).
5. **Access:** you have IEEE via Tel Aviv University; for paywalled items check arXiv/ResearchGate
   for the author's preprint, and Semantic Scholar / Connected Papers to map a topic's graph.
6. **Log what you keep** here with a one-line reason (why it matters to *our* model) — like [E]'s
   equation map — so the reference earns its place.

## The other two AT&T challenges (future repos, same rhythm)
- **Field-technician dispatch** — VRP/TSP-with-time-windows; refs: QAOA, Lucas, Glover QUBO, D-Wave / Qiskit / PennyLane optimization docs.
- **Network traffic routing** — multi-commodity flow; refs: RFC 2702 (MPLS TE), Fortz–Thorup (OSPF-weight TE), Kar–Kodialam–Lakshman (minimum-interference routing).
