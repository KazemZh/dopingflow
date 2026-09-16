Workflow Overview
=================

Conceptual Pipeline
-------------------

The ML Doping Workflow implements a fully automated, multi-stage surrogate
pipeline for the exploration of doped crystalline materials.

It combines symmetry-aware structure generation with machine-learned
interatomic potentials to efficiently screen large configurational spaces.

The workflow is designed to first identify promising bulk candidates and
then optionally extend the analysis to oxygen vacancies and surface structures.


Pipeline Structure
------------------

Reference Construction → Optional Correction Fit → Enumeration → Screening → Relaxation → Filtering → Band Gap → Formation Energy → Database → 1D Alloy Hull → Phase Diagram → Vacancies

Optional Post-Processing:

Vacancy Results → Optional M0/M1-corrected Vacancy Thermodynamics

Vacancy Results → Vacancy-resolved Raw/Corrected Phase Diagram

Database → Surface Generation → Surface Relaxation

The vacancy M0/M1 option reuses the already fitted backend-specific correction
model. It does not refit a separate vacancy-specific model. The correction is
applied to the relaxed parent and selected vacancy minima before different
vacancy counts are compared. Global/chemistry-specific experimental oxygen
calibration is mutually exclusive with this option to prevent double counting.

The vacancy-resolved energy-above-hull analysis is intentionally a
post-processing use of the existing phase-diagram stage. The normal ``run-all``
order creates the standard phase diagram before vacancy generation. Therefore,
a new workflow normally runs ``vacancies`` first and then reruns
``phase-diagram`` with ``include_vacancy_minima = true``. No vacancy relaxation
is repeated by this post-processing step.

Sequential Workflow
-------------------

For gradual doping studies, ``dopingflow`` also provides a sequential workflow:

::

   dopingflow sequential-run -c input.toml

When energy correction is enabled, ``sequential-run`` performs the correction
fit/reuse once as a preflight before the composition loop. The model is not
refitted per composition; it uses the existing references and the same
backend/model/task required for candidate relaxation.

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

0b. Backend-specific energy correction (optional)

   - Load curated Kingsbury or explicitly unit-tagged custom measurements
   - Match a separate calibration manifest to same-backend calculated structures
   - Fit an uncertainty-weighted, identifiable correction model
   - Save exact calibration records, covariance, fit report, and backend provenance

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
   - Retain the legacy raw reaction energy and, when enabled, add a separately
     corrected balanced-reaction energy and coefficient uncertainty

7. Database assembly

   - Aggregate results across all stages
   - Export a unified CSV database

8. Restricted one-dimensional alloy hull

   - Construct the lower convex envelope along one fixed substitutional alloy line
   - Compute energy above the restricted 1D hull per cation
   - Identify stable alloy vertices and tie-line decompositions

9. Full phase diagram analysis

   - Build a separate pymatgen hull for each exact candidate chemical system
   - Include compatible reference phases and lower-dimensional candidates
   - Require one elemental terminal reference for every element
   - Write per-system CSV files plus a combined result table
   - When correction is enabled, build independent complete raw and corrected
     entry sets and reconstruct both hulls
   - Optionally, after the vacancy stage has completed, include the lowest-energy
     relaxed structure at every vacancy count and write
     ``vacancy_energy_above_hull.csv``

10. Oxygen vacancies (optional run-all extension)

   - Analyze actual selected-parent dopant counts and reachable formal charges
   - Expand each parent with a user-selected supercell
   - Search each count by symmetry enumeration or Metropolis Monte Carlo
   - In Monte Carlo mode, swap vacancies/anions and any number of cation species
   - Run isothermally or hold and linearly cool from an annealing temperature
   - Screen and relax top-k arrangements with one shared ML calculator, then rerank
   - Keep vacancy results separate from normal thermodynamic databases
   - Optionally aggregate exact integer compositions and converged count minima
   - Verify an O2 reference and solve exact oxygen-grand-potential stability windows
   - Optionally apply the fitted M0/M1 correction as
     ``C(defect)-C(parent)`` with correlated coefficient uncertainty
   - Write compact minima, interval, selected-condition, pressure, and free-energy tables

The preferred vacancy count depends on ``delta_mu_O``. Raw total energy, energy
per atom, and energy per vacancy cannot rank structures with different oxygen
contents. Vacancy formation thermodynamics therefore use an oxygen reservoir.
When ``apply_fitted_energy_correction = true``, the active cross-count energy is
corrected while the raw reaction energy remains in the output. See
:doc:`methods/vacancy_energy_correction`.

The optional vacancy-resolved energy-above-hull post-processing asks a different
question: for each oxygen-deficient composition, how far is the selected vacancy
minimum from the closed-system decomposition hull? It uses the existing phase-
diagram entry set and correction model. A decrease in energy above hull with
vacancy count means movement closer to that composition's hull, not necessarily
favorable oxygen removal. It is also not yet an oxygen-open competing-phase
grand-potential hull.

11. Surface generation (optional)

   - Select candidates from the database
   - Generate slab structures for chosen Miller indices
   - Enumerate surface terminations
   - Optionally fix atoms in the slab

12. Surface relaxation (optional)

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

- The standard bulk stages use static relaxed energies unless a method explicitly
  adds another thermodynamic term.
- The vacancy T-pO2 analysis changes the oxygen-gas reservoir and is not a full
  finite-temperature competing-phase diagram.
- M0/M1-corrected vacancy thermodynamics must not be combined with the global or
  chemistry-specific experimental oxygen calibration.
- The vacancy-resolved closed-system hull is only as complete as the competing
  phases supplied to the phase-diagram calculation.
- Surface generation is intentionally decoupled from the main pipeline and is executed separately.

Typical Usage
-------------

A typical workflow consists of:

1. Fitting/reusing the optional correction and running the bulk/vacancy stages:

   ::

      [energy_correction]
      enabled = true
      model_family = "m0"   # or m1 / auto

      [vacancies]
      apply_fitted_energy_correction = true
      oxygen_reference_mode = "reference_file"

   ::

      dopingflow corrections-fit -c input.toml
      dopingflow run-all -c input.toml --from generate --until vacancies

2. Enabling the vacancy-resolved phase diagram if desired:

   ::

      [phase_diagram]
      include_vacancy_minima = true
      vacancy_results_directory = "vacancy-selected"

3. Rebuilding the phase diagram without repeating vacancy relaxation:

   ::

      dopingflow phase-diagram -c input.toml

4. Inspecting the vacancy correction controls and raw/corrected phase stability
   in the Streamlit ``Vacancy M0/M1 Energy Correction`` and ``Phase Diagram`` pages.

5. Generating and optionally relaxing surfaces:

   ::

      dopingflow surface -c input.toml
