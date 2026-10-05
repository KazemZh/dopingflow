# Changelog

- Reference construction now relaxes configured metal and oxide phase-diagram
  entries in either formation-reference mode and incrementally reuses entries
  whose source structures and relaxation settings are unchanged.
- Selecting the phase-diagram source in the GUI Results Explorer now provides
  two dopant selectors and a direct composition-minimum energy-above-hull plot
  with compatible boundary curves, stable markers, and the host reference.

All notable changes to this project will be documented in this file.

The format loosely follows semantic versioning.

---

## [Unreleased]
### Added
- Surface Pourbaix results now support interactive thermodynamic replotting from selected already-calculated chemistry/coverage groups, so users can include or exclude specific protonation/adsorbate coverages without rerunning structure calculations.
- Surface Pourbaix now detects exposed O/cation sites from the outermost atomic layer/group by default, using a normal-direction layer-gap criterion inside the safety depth window; the former full height-window behavior remains available only as an explicit legacy mode.
- Surface Pourbaix single-family coverage lists now accept `0%` as the clean state instead of raising a validation error; negative values and values above 100% remain invalid.
- Surface Pourbaix can now run different selected surfaces concurrently in isolated CPU worker processes (`parallel_surfaces`, `surface_workers`), while preserving checkpoint reuse and deterministic aggregation; single-CUDA and executing-GPAW runs automatically fall back to one effective worker for safety.
- Surface Pourbaix coverage now always uses the complete eligible site set (legacy 8-site caps are ignored), and lattice-O protonation has an independent side selector with `both` as the default while O*/OH*/H2O* adsorption keeps its own side choice.
- Unified every GUI structure visualization behind `gui/view_structure.py`, with the enhanced shared element palette, customizable atom colors, hover site labels, consistent sphere/stick styling, unit-cell display, coordinate axes, and true slab-normal surface arrows.
- Surface Pourbaix now validates relaxed surface chemistry before thermodynamic competition: surface-bound reconstructions are reclassified, while detached O/OH/H2O/O2-like and fragmented states are excluded by default and reported explicitly in the GUI/CSV outputs.
- New **Surface Pourbaix** stage between surface segregation and leaching. It enumerates protonated, O*, OH*, H2O*, and mixed O/OH intact-surface states, ML-relaxes coverage/site arrangements, and builds CHE stability maps versus potential and pH.
- The Surface Pourbaix default pH window is **-1 to 3** with configurable SHE/RHE potential scale, potential range, grid spacing, and temperature.
- Optional provenance-checked GPAW refinement targets the clean slab plus ML-stable/near-stable states rather than blindly sending every arrangement to DFT.
- A Streamlit Surface Pourbaix page, example input, restart-safe state checkpoints, stable-domain tables, and a dedicated `leaching_surface_states.csv` hand-off are included.
- Leaching `source_mode = "auto"` now prefers electrochemically stable Surface Pourbaix states before falling back to clean/refined surface tables.
- New **Surface Segregation Monte Carlo** stage for user-selected natural surface
  terminations. Fixed-composition host↔dopant swaps are sampled with a user-selected
  MLFF at finite temperature, with burn-in and production sampling.
- Site-resolved dopant occupancy, surface/subsurface/bulk populations normalized by
  available site count, occupancy-derived effective segregation free energies,
  site PMFs, convergence traces, swap statistics, and representative MC structures
  are written for every selected surface.
- Dedicated Streamlit **Surface Segregation MC** page with exact surface selection,
  calculator/temperature/trajectory controls, live progress, and occupancy/free-energy
  heat maps.
- Surface generation now keeps exactly **one natural structure per termination**:
  dopants are never swapped between host sites. Each termination is labeled from
  the dopants' inherited positions (surface/subsurface/bulk-like) on the slab cut.
- Refinement selection is simplified to global or per-orientation termination
  selection, with independent termination counts per Miller orientation and
  optional exact manual include/exclude overrides.
- The Surface Screening GUI can review the automatic shortlist and manually include
  or exclude exact screened surfaces before refinement. Manual choices use stable
  surface IDs and can be reapplied without rerunning the screen.
- All successfully refined surfaces are kept by default; the mandatory GUI-level
  final global top-k reduction has been removed.
- Surface refinement shortlist selection now supports both **global top-N** and
  **top-N per Miller orientation**. The per-orientation mode keeps every requested
  facet represented when one orientation dominates the global surface-energy ranking.
- The previous surface segregation-energy calculation/reporting path has been removed.
- Surface target selection now mirrors Electronic Conductivity/Oxidation States:
  `source_root`, independent vacancy-free/O-vacancy toggles, and optional exact/glob
  `target_include` selectors drive both GUI preview and backend execution. Vacancy-free
  and O-vacancy source structures are ranked independently.
- Dedicated Streamlit **Surface Screening** page with candidate/facet/termination
  configuration, natural dopant-depth labeling, independent GRACE/MACE-style
  screen/refine calculators, separate-environment execution, surface-energy plots,
  3D slab browsing, and raw result tables. The duplicate Surface editor in the main
  Input Builder now links to this page.
- Staged surface screening and refinement with separate fast and higher-fidelity
  MLFF calculators, configurable Miller orientations and terminations,
  representative surface/subsurface/bulk co-dopant placements, calculator-specific
  bulk references, top-k handoff, and explicit exclusion of non-proportional slabs
  from the simple surface-energy ranking.
- New surface-scan, surface-refine, and surface CLI commands, plus a GRACE-to-MACE
  example configuration using MACE MH-1 / matpes_r2scan refinement.
- A staged Monte Carlo vacancy workflow for incompatible ML environments:
  ``dopingflow vacancies-mc-search`` runs only the ``mc_*`` search calculator,
  while ``dopingflow vacancies-finalize`` runs only the ordinary final vacancy
  calculator. This supports GRACE occupation search followed by MACE single
  points, relaxation, reranking, and thermodynamics without installing both
  backends in the same Python environment.
- Explicit ``vacancy_counts = [1, 2, ...]`` for research-design vacancy counts.
  The explicit list controls the searched counts while formal-charge scenarios
  remain available as interpretation metadata.
- Independent Monte Carlo search-calculator controls:
  ``mc_backend``, ``mc_model``, ``mc_task``, ``mc_device``, and ``mc_gpu_id``.
  Search and final-calculator provenance are stored separately.
- Staged parent controls: ``parent_include`` limits the study to named
  compositions or exact ``composition/candidate`` IDs; ``parent_pick`` can keep
  all selected parents or only the lowest-energy selected parent per
  composition; ``output_directory`` mirrors the complete GRACE/MACE handoff
  tree away from the source parents.
- A dedicated Streamlit **Staged Vacancy Monte Carlo: GRACE → MACE** page. It
  exposes parent selection/output routing, explicit vacancy counts, supercell,
  GRACE search settings, coupled cation/vacancy move weights, annealing/stopping
  controls, MACE finalization settings, and separate environment run buttons.
- A dedicated staged vacancy configuration reference and synchronized staged
  example using a ``2 x 2 x 2`` search cell, explicit one/two-vacancy counts,
  GRACE-1L-OMAT search, and MACE-MH finalization.
- Optional user-controlled fitted M0/M1 correction of oxygen-vacancy
  thermodynamics. The vacancy reaction correction is evaluated as
  `C(defect)-C(parent)` with correlated covariance propagation, raw values are
  retained, and the feature is guarded against simultaneous
  global/chemistry-specific experimental oxygen calibration to avoid double
  counting.
- A dedicated Streamlit **Vacancy M0/M1 Energy Correction** page for enabling
  the correction, reviewing the configured M0/M1/auto family, and explicitly
  accepting legacy vacancy-energy provenance when older data require it.
- Optional vacancy-resolved raw/corrected phase diagrams: the lowest-energy
  relaxed structure at each oxygen-vacancy count can be added to the existing
  phase-diagram entry set, with a compact `vacancy_energy_above_hull.csv` output
  for plotting energy above hull versus vacancy count.
- A dedicated Streamlit **Phase Diagram** page with exact-system selection,
  raw/corrected hull switching, a two-dopant composition map, concentration
  curves, decomposition tables, and vacancy-count energy-above-hull plots.
- A second vacancy search method, ``search_method = "monte-carlo"``, that
  performs generic vacancy-anion and multi-species cation swaps on a configurable
  supercell, archives low-energy occupations, and feeds the established top-k
  relaxation and reranking stages.
- Optional Monte Carlo annealing with a high-temperature hold, linear cooling
  ramp, and final target temperature; the default remains constant-temperature
  sampling when ``mc_annealing = false``.
- Finite-temperature vacancy formation free energies with optional ideal or
  exact-orbit configurational entropy. The explicit partition-function mode uses
  exact symmetry degeneracies and writes `vacancy_formation_free_energy.csv/json`
  while keeping direct delta-mu intervals static-lattice.
- Optional backend-specific, uncertainty-weighted energy-correction fitting
  from the curated Kingsbury dataset or explicit custom measurements, with
  phase-matched calibration manifests and reproducibility artifacts.
- Conservative M0/M1 correction families with workflow-scoped oxide-cation
  terms, forced-family modes, and automatic leave-one-out selection that falls
  back to M0 unless M1 clears independent coverage, improvement, conditioning,
  family non-worsening, and optional one-standard-error gates.
- Phase-resolved calibration expansion across the complete host-and-dopant
  scope, with strict curated phase/`likely_mpid` selection, immutable OPTIMADE
  structure caches, same-backend calibration relaxation and chemical-system
  hulls, and hashed expansion, candidate-model, and selection artifacts.
- Separately retained raw/corrected formation energies and independently rebuilt
  raw/corrected multicomponent phase diagrams.
- A complete one-process oxygen-vacancy workflow through
  ``dopingflow vacancies -c input.toml`` plus the staged Monte Carlo search and
  finalize entry points above. All paths share the same flat ``[vacancies]``
  configuration table.
- Actual-count formal-charge analysis, mixed-valence population scenarios,
  symmetry-reduced exact/sampled enumeration, fixed-count ML ranking, top-k
  relaxation, compatible parent references, resumable metadata, and separate
  vacancy CSV/JSON databases.
- Optional composition-level vacancy thermodynamics with verified/explicit O2
  references, exact lower-envelope intervals, selected-condition best counts,
  plotting-ready compact tables, GUI controls, and a matplotlib example.
- Global and chemistry-specific vacancy oxygen-reference calibration from real
  refs-build ordinary binary oxides, matching same-backend bulk metals, and
  experimental 298 K formation enthalpies. Calibration reports retain every
  included/excluded oxide, individual per-O value, fit spread, and residual.
- Optional ideal occupied/vacant oxygen-site configurational entropy for the
  explicitly temperature-dependent vacancy T-pO2 map, while direct delta-mu
  intervals remain static-lattice quantities.
- Explicit Level-1 static-lattice outputs and an ideal-gas temperature-oxygen-
  pressure mapping. A user O2 standard-state table is supported; without one,
  pressure results are labeled qualitative and approximate.
- Built-in continuous NIST O2 Shomate standard-state corrections for arbitrary
  temperatures from 100 to 6000 K, with extrapolation prohibited and the source
  and zero-point-energy convention recorded in metadata.
- Built-in 298 K enthalpy-origin handling for NIST O2 gas thermochemistry when
  the oxygen reference itself was fitted to experimental 298 K formation
  enthalpies.
- Accuracy-aware GUI and example T-pO2 plots identify NIST Shomate, user-table,
  and approximate no-thermal-correction mappings; new GUI inputs default to the
  NIST mode.
- The general Results Explorer vacancy plots retain their original
  ``delta_mu_O`` axes and behavior; the T-pO2 map compares inclusion versus
  omission of the configured O2 ``delta_mu_O_standard(T)`` term.
- Vacancy controls, execution presets, result discovery, and three-way structure
  comparison in the Streamlit GUI.
- Dynamic discovery of the installed MACE ``mace_mp`` foundation-model aliases,
  including MACE-MH-0 and MACE-MH-1.
- MACE custom-checkpoint paths and optional multi-head selection through each
  stage's existing ``model`` and ``task`` parameters.
- Shared GUI controls and backend tests for MACE aliases, checkpoints, and heads.
- Multi-reference oxide endpoint and co-doping provenance columns.
- One phase-diagram CSV per exact candidate chemical system.
- Phase-diagram source selection in the GUI Results Explorer.
- Tests for flat formation configuration, metadata flattening, per-system phase
  diagrams, staged vacancy parent selection, split-environment entry points, and
  staged GUI command construction.

### Changed
- The staged Monte Carlo path now supports a dedicated output root while still
  reading source scan/relax structures from the original parent tree. GRACE and
  MACE therefore share the same persisted handoff without modifying source
  calculations.
- The current staged cross-backend selection is explicitly documented as
  ``GRACE archive -> GRACE top-k -> MACE single-point/relax selected``. MACE does
  not yet rescore the complete GRACE archive before the relaxation top-k is
  chosen; cross-backend ranking agreement should be validated before a large
  production campaign.
- Documentation and GUI guidance now distinguish the package-level small Monte
  Carlo defaults from the larger project-specific ``2 x 2 x 2`` production
  starting schedule. ``mc_max_steps`` is the total trajectory length, and the
  no-improvement patience counter is active from step 1.
- Phase-diagram GUI metadata matching now tolerates calculations copied from
  another filesystem by matching the final `composition/candidate` path when
  absolute path prefixes differ. Vacancy phase rows no longer break the normal
  composition plot and are represented separately in the vacancy-hull view.
- Vacancy oxygen-reference selection now exposes ``global`` and
  ``chemistry-specific`` calibrated modes alongside the existing
  ``reference_file``, ``same_calculator``, ``explicit``, and ``none`` modes.
- Calibrated vacancy references are backend/model/task-specific and never
  hard-code a literature O2 correction or invent a missing calibration oxide.
- Blank or omitted MACE ``task`` values now resolve to the compatible
  ``omat_pbe`` head for ``model = "mh-1"`` instead of requesting MACE's
  nonexistent ``default`` head.
- GUI-generated defaults match runtime defaults for worker counts, random seeds,
  relative energies, phase-diagram stability, and ordinary vacancy calculators;
  the dedicated staged GUI supplies production-oriented staged values without
  changing the library defaults.
- The MACE optional dependency now requires ``mace-torch>=0.3.14`` and the GUI
  no longer advertises unsupported ``small-mpa-0``/``large-mpa-0`` aliases.
- Relative-energy settings remain directly inside `[formation]` as
  `relative_enabled` and `endpoint_x`.
- Formation-stage oxide tie-line values are preserved during collection and
  sequential database merging.
- Sequential formation/collection rebuilds per composition and merges only the
  current invocation's steps, preventing stale project-level databases and
  obsolete historical steps from leaking into recomputed results.
- Correction provenance verifies convergence, identical relaxation signatures,
  and stored structure hashes for host, elemental, gas, and compound references
  before fitting or application.
- `[phase_diagram].skip_if_done` and `stable_threshold_eV_per_atom` are honored by
  the implementation.
- Formation, database, phase-diagram, vacancy, GUI, and example documentation is
  synchronized with the current staged workflow and multi-reference output
  schema.

## [0.3.0] - 2026-02-16
### Added
- Streamlit GUI for interactive workflow configuration and execution
- Results Explorer with Plotly-based visualization
- Improved Input Builder with structured parameter sections
- GUI-based run controls (All / Range / Single stage execution)

### Changed
- Improved layout and parameter descriptions in GUI
- Removed unnecessary `poscar_in` exposure from Scan GUI
- Enhanced formation energy explanations with equations

---

## [0.2.0] - 2026-02-10
### Added
- ALIGNN bandgap prediction stage
- Formation energy normalization options
- Improved CLI run-all command controls

---

## [0.1.0] - 2026-02-01
### Initial release
- CLI-based modular workflow
- Reference energy computation
- Structure generation and dopant enumeration
- M3GNet scanning and relaxation
- Filtering and candidate collection
