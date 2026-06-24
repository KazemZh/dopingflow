Workflow Overview
=================

Conceptual Pipeline
-------------------

The ML Doping Workflow implements a fully automated, multi-stage surrogate
pipeline for the exploration of doped crystalline materials.

It combines symmetry-aware structure generation with machine-learned
interatomic potentials to efficiently screen large configurational spaces.

The workflow is designed to first identify promising bulk candidates and
then optionally extend the analysis to surface structures.


Pipeline Structure
------------------

Reference Construction → Enumeration → Screening → Relaxation → Filtering → Band Gap → Formation Energy → Database → 1D Alloy Hull → Phase Diagram

Optional Post-Processing:

Database → Surface Generation → Surface Relaxation

Sequential Workflow
-------------------

For gradual doping studies, ``dopingflow`` also provides a sequential workflow:

::

   dopingflow sequential-run -c input.toml

In this mode, the workflow processes the composition list step by step. For example,
a sequence such as 2.5%, 5%, 7.5%, and 10% Sb is treated as a gradual doping path.
At each step, the workflow scans possible dopant positions, relaxes the selected
candidates, and copies the lowest-energy relaxed structure as the starting point
for the next composition.

The sequential workflow supports two modes:

- ``full``: run generation, scan, relaxation, filtering, optional bandgap,
  formation energy, and database collection for each composition.
- ``recompute_energies``: reuse existing relaxed sequential structures and
  recompute formation/mixing energies after changing reference structures.

After ``sequential-run`` completes, run ``dopingflow alloy-hull -c input.toml``
to construct the global hull from the merged sequential database.

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

8. Restricted one-dimensional alloy hull

   - Construct the lower convex envelope along one fixed substitutional alloy line
   - Compute energy above the restricted 1D hull per cation
   - Identify stable alloy vertices and tie-line decompositions

9. Full phase diagram analysis

   - Compute energy above hull for each candidate using pymatgen
   - Compare against all included competing phases in the multicomponent chemical system

10. Surface generation (optional)

   - Select candidates from the database
   - Generate slab structures for chosen Miller indices
   - Enumerate surface terminations
   - Optionally fix atoms in the slab

11. Surface relaxation (optional)

   - Relax slab structures using ML interatomic potentials
   - Apply atom constraints (e.g. fixed bottom layers)
   - Use the same backend abstraction as bulk relaxation

Design Principles
-----------------

- **Modular**: Each stage can be executed independently
- **Backend-agnostic**: Multiple ML potentials are supported
- **Reproducible**: Fully controlled via ``input.toml``
- **Scalable**: Supports multiprocessing and GPU execution
- **Extensible**: New models and stages can be added easily


Notes
-----

- The core workflow (Stages 0–9) focuses on bulk screening, database generation,
  and thermodynamic analysis.
- Surface generation is intentionally decoupled from the main pipeline and is executed separately.
- This design allows users to:
  - inspect and validate bulk candidates before surface modeling
  - control the number of generated slabs
  - avoid combinatorial explosion of surface structures

Typical Usage
-------------

A typical workflow consists of:

1. Running the full bulk pipeline:

   ::

      dopingflow run-all -c input.toml

2. Inspecting the resulting database:

   ::

      results_database.csv

3. Generating and optionally relaxing surfaces:

   ::

      dopingflow surface -c input.toml
