# Changelog

All notable changes to this project will be documented in this file.

The format loosely follows semantic versioning.

---

## [Unreleased]
### Added
- A complete one-command oxygen-vacancy workflow, exposed only as
  ``dopingflow vacancies -c input.toml`` and configured in one flat
  ``[vacancies]`` table
- Actual-count formal-charge analysis, mixed-valence population scenarios,
  symmetry-reduced exact/sampled enumeration, fixed-count ML ranking, top-k
  relaxation, compatible parent references, resumable metadata, and separate
  vacancy CSV/JSON databases
- Vacancy controls, execution presets, result discovery, and three-way structure
  comparison in the Streamlit GUI
- Dynamic discovery of the installed MACE ``mace_mp`` foundation-model aliases,
  including MACE-MH-0 and MACE-MH-1
- MACE custom-checkpoint paths and optional multi-head selection through each
  stage's existing ``model`` and ``task`` parameters
- Shared GUI controls and backend tests for MACE aliases, checkpoints, and heads
- Multi-reference oxide endpoint and co-doping provenance columns
- One phase-diagram CSV per exact candidate chemical system
- Phase-diagram source selection in the GUI Results Explorer
- Tests for flat formation configuration, metadata flattening, and per-system phase diagrams

### Changed
- Blank or omitted MACE ``task`` values now resolve to the compatible
  ``omat_pbe`` head for ``model = "mh-1"`` instead of requesting MACE's
  nonexistent ``default`` head
- GUI-generated defaults now match runtime defaults for worker counts, random
  seeds, relative energies, phase-diagram stability, and vacancy calculators;
  the input reference also labels default and conditionally required fields
- The MACE optional dependency now requires ``mace-torch>=0.3.14`` and the GUI
  no longer advertises unsupported ``small-mpa-0``/``large-mpa-0`` aliases
- Relative-energy settings now remain directly inside `[formation]` as
  `relative_enabled` and `endpoint_x`
- Formation-stage oxide tie-line values are preserved during collection and
  sequential database merging
- `[phase_diagram].skip_if_done` and
  `stable_threshold_eV_per_atom` are now honored by the implementation
- Formation, database, phase-diagram, GUI, and example documentation now
  describe the multi-reference output schema

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
