
Workflow Overview
=================

Conceptual Pipeline
-------------------

The ML Doping Workflow implements a fully automated, multi-stage surrogate
pipeline for the exploration of doped crystalline materials.

Pipeline Structure
------------------

Enumeration → Ranking → Relaxation → Filtering → Band Gap → Formation Energy → Database

Stages
------

1. Symmetry-reduced dopant enumeration
2. ML-based energy ranking (M3GNet)
3. Full cell relaxation (M3GNet + FIRE)
4. Energy-window filtering
5. Band gap prediction (ALIGNN)
6. Formation energy evaluation using ML chemical potentials
7. Database assembly
