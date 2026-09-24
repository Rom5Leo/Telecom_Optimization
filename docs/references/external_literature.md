# External literature — running bibliography (antenna tilt & beyond)

Papers, standards and datasets found by literature search, kept so we can pull them in when a stage
needs them. Each entry has a one-line **why-for-us**. The three *owned* papers (the physics + the
hackathon framing) are in this folder as PDFs; see `README.md` and `RF_model_references.md`.

## RF physics & the standardized model
- **[E] Athley & Johansson**, *Impact of Electrical and Mechanical Antenna Tilt on LTE Downlink
  System Performance*, IEEE 2010 — **owned** (`E_Ericsson2010_tilt_LTE_downlink.pdf`). Our RF model.
- **[BL] Eckhardt, Klein & Gruber**, *Vertical Antenna Tilt Optimization for LTE Base Stations*,
  IEEE 2011 — **owned**. The optimization objective (cell-edge weighting), clustering, baseline.
- **[A] Walia**, *Antenna Tilt Optimization: Mechanical vs Electrical*, 5G/6G Academy 2026 —
  **owned**. Practitioner framing + operator ROI.
- **3GPP TR 38.901** (channel model) — the standardized antenna radiation pattern behind our `G_el`.
  https://arxiv.org/html/2507.19266v1 (Rel-19 overview referencing 38.901).

## Coverage-Capacity Optimization (CCO) & traffic/user-density weighting
> Answers Leo's "permit overlap where nobody is" — weight the objective by demand ρ(x,y).
- *On the potential of traffic-driven tilt optimization in LTE-A networks*, IEEE —
  https://ieeexplore.ieee.org/document/6666644/ — the direct precedent for demand-weighted tilt.
- *Dynamic Coverage Optimization for 5G Ultra-dense Networks Based on User Densities*, Wireless Pers.
  Comm. 2022 — https://link.springer.com/article/10.1007/s11277-022-09969-4 — density-driven CCO.
- *A Mathematical Model for User Traffic in Coverage and Capacity Optimization*, IEEE —
  https://ieeexplore.ieee.org/document/5956220/ — how to put a traffic map into the objective.
- Dandanov et al., *Dynamic Self-Optimization of the Antenna Tilt for Best Trade-off Between Coverage
  and Capacity*, WPC 2017 — https://link.springer.com/article/10.1007/s11277-016-3849-9.
- *Optimizing Coverage and Capacity in Cellular Networks using Machine Learning*, arXiv:2010.13710 —
  https://arxiv.org/pdf/2010.13710 — CCO with ML (Nokia/Bell Labs).
- *Online Antenna Tilt-Based Capacity and Coverage Optimization* (Vodafone Chair) —
  https://www.vodafone-chair.org/pbls/legacy/s-berger/Online_Antenna_Tilt-Based_Capacity_and_Coverage_Optimization.pdf

## Self-Organizing Networks / Remote Electrical Tilt via learning
> How operators actually re-tune tilt on live traffic — our static solve is the snapshot they repeat.
- Vannella et al., *Remote Electrical Tilt Optimization via Safe Reinforcement Learning*,
  arXiv:2010.05842 — https://arxiv.org/abs/2010.05842.
- *Off-policy Learning for Remote Electrical Tilt Optimization*, arXiv:2005.10577 —
  https://arxiv.org/html/2005.10577.
- *Radio Network Optimization Through Antenna Tilt Adjustment Using Metaheuristics and GIS*,
  JNSM 2026 — https://link.springer.com/article/10.1007/s10922-026-10051-8.

## Quantum approaches (project positioning / novelty)
- *Large-scale wireless coverage optimization: A quantum approach*, 2025 —
  https://www.sciencedirect.com/science/article/pii/S2405959525001006 — closest prior art; read early.
- *Quantum Computing for Large-scale Network Optimization: Opportunities and Challenges*,
  arXiv:2509.07773 — https://arxiv.org/pdf/2509.07773.
- *Heuristic Quantum Optimization for 6G Wireless Communications*, IEEE Network —
  https://par.nsf.gov/servlets/purl/10299374.
- *QAOA for multi-objective routing in large-scale 6G networks*, Comput. Netw. 2025 —
  https://www.sciencedirect.com/science/article/pii/S1389128625003123 (relevant to the routing repo).

## Datasets — real tower geometry (locations only; configs are proprietary)
> Use real coordinates as a layout, simulate the RF/tilt on top. State the license/caveat.
- **OpenCelliD** — largest open crowdsourced cell DB, downloadable: https://www.opencellid.org/ ,
  downloads https://opencellid.org/downloads.php (CC-BY-SA; cell locations, MCC/MNC → carrier).
- **FCC Antenna Structure Registration (ASR)** — US regulatory tower registry:
  https://wireless2.fcc.gov/UlsApp/AsrSearch/asrRegistrationSearch.jsp (structures/heights, not tilt).
- HIFLD/ArcGIS "Cellular Towers in the US" layer —
  https://hub.arcgis.com/datasets/15dabb4108254481b591018be2598f3c_0/about.

## 3D / UAV coverage (the "why not 3D?" extension)
> Tilt is a vertical control, so 3D genuinely changes the answer (upper floors, drones).
- *On the Optimal 3D Placement of a UAV Base Station for Maximal Coverage*, arXiv:2008.09262 —
  https://arxiv.org/pdf/2008.09262.
- *Optimal 3D Placement of UAV-BS Subject to User Priorities and Distributions*, Electronics 2022 —
  https://www.mdpi.com/2079-9292/11/7/1036 (also a user-priority/weighting example).
