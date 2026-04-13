Workflow Overview
=================

Conceptual Pipeline
-------------------

The ML Doping Workflow implements a fully automated, multi-stage surrogate
pipeline for the exploration of doped crystalline materials.

It combines symmetry-aware structure generation with machine-learned
interatomic potentials to efficiently screen large configurational spaces.


Pipeline Structure
------------------

Reference Construction → Enumeration → Screening → Relaxation → Filtering → Band Gap → Formation Energy → Database


Stages
------

0. Reference construction and relaxation
   - Relax host structure (unit cell and supercell)
   - Relax reference phases (metal or oxide mode)
   - Build thermodynamic reference dataset

1. Symmetry-reduced dopant enumeration
   - Generate substitutional doped configurations
   - Identify symmetry-unique arrangements on the cation sublattice

2. ML-based energy screening
   - Evaluate single-point energies using a selected ML backend
   - Supports: M3GNet, UMA, MACE, GRACE
   - Exact enumeration or stochastic sampling

3. Structure relaxation
   - Relax candidate structures using ML forces
   - Uses ASE optimizers (e.g. BFGS, FIRE, LBFGS)
   - CPU or GPU execution

4. Energy-based filtering
   - Select low-energy candidates
   - Window-based or top-N selection strategies

5. Band gap prediction
   - Predict electronic band gaps using ALIGNN

6. Formation energy evaluation
   - Compute formation energies using reference structures
   - Supports metal and oxide reference schemes

7. Database assembly
   - Aggregate results across all stages
   - Export a unified CSV database


Design Principles
-----------------

- **Modular**: Each stage can be executed independently
- **Backend-agnostic**: Multiple ML potentials are supported
- **Reproducible**: Fully controlled via ``input.toml``
- **Scalable**: Supports multiprocessing and GPU execution
- **Extensible**: New models and stages can be added easily