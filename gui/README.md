# dopingflow GUI (Streamlit)

This folder contains the Streamlit-based graphical user interface for **dopingflow**.
The GUI is optional; the CLI remains the primary interface for scripted and HPC
workflows.

The GUI supports interactive `input.toml` editing, workflow execution, log
inspection, structure viewing, phase-diagram analysis, vacancy thermodynamics,
and the staged GRACE → MACE Monte Carlo vacancy workflow.

---

## Installation

Install the GUI extra from the project root:

```bash
pip install -e ".[gui]"
```

Install other extras only in the environments that need them. In particular,
the staged vacancy workflow is designed for **separate GRACE and MACE
environments**:

```bash
conda create -n dopingflow-grace python=3.11 pip -y
conda activate dopingflow-grace
pip install -e ".[grace,gui]"

conda create -n dopingflow-mace python=3.11 pip -y
conda activate dopingflow-mace
pip install -e ".[mace,gui]"
```

Both environments may point to the same editable checkout. If experimental
formation-energy correction or oxygen-reference calibration uses the curated
dataset, install the `corrections` extra in the environment that performs that
analysis.

---

## Launching the GUI

From the repository/project root:

```bash
streamlit run gui/app.py
```

Streamlit also discovers the dedicated pages under `gui/pages/`.

---

## Main app pages

### Input Builder

The main Input Builder edits the normal project-wide `input.toml` sections,
including:

- structure and doping setup;
- references, scan, relax, filter, bandgap, formation, and phase diagram;
- the ordinary one-process `[vacancies]` controls;
- vacancy oxygen-reference and thermodynamic settings;
- optional fitted solid-energy correction controls.

The ordinary vacancy panel remains useful for symmetry enumeration and for
Monte Carlo when every requested backend is available in one Python
environment.

### Run

Runs ordinary workflow stages and monitors logs. Large HPC campaigns are still
better launched from the CLI or scheduler.

### Results Explorer

Loads the main results database, phase-diagram outputs, and vacancy-analysis
files. It distinguishes closed-system energy above hull from oxygen-open vacancy
thermodynamics and can display the compact vacancy free-energy / T-pO2 outputs.

### Structure Viewer

Provides interactive inspection of generated, relaxed, and vacancy-containing
structures.

---

## Staged Vacancy MC page

`gui/pages/Vacancy_MC_Staged.py` is the dedicated interface for the current
large-supercell GRACE-search/MACE-finalization workflow.

It exposes the staged controls that are intentionally not overloaded into the
ordinary Input Builder:

```toml
[vacancies]
parent_source = "directory"
parent_directory = "vacancy-selected"
parent_include = ["Ti_2.5Sb_2.5", "Ti_2.5Sb_5"]
parent_pick = "lowest_energy"
output_directory = "vacancy-mc-grace-mace"

search_method = "monte-carlo"
vacancy_counts = [1, 2]
supercell = [2, 2, 2]

mc_backend = "grace"
mc_model = "GRACE-1L-OMAT"
mc_task = ""
mc_device = "cuda"
mc_gpu_id = 0

mc_annealing = true
mc_initial_temperature_K = 1500.0
mc_annealing_hold_steps = 5000
mc_annealing_steps = 50000
mc_temperature_K = 600.0
mc_run_mode = "combined"
mc_max_steps = 200000
mc_patience = 100000
mc_improvement_tolerance_eV = 1.0e-5
mc_energy_window_eV = 1.0
mc_cation_move_weight = 0.5
mc_vacancy_move_weight = 0.5
sample_seed = 42
sample_max_saved = 100

backend = "mace"
model = "mh-1"
task = "matpes_r2scan"
device = "cuda"
gpu_id = 0
topk_per_vacancy_count = 20
optimizer = "bfgs"
fmax = 0.05
max_steps = 300
```

The page also builds separate `conda run` commands for the two environments and
can execute them directly:

```bash
conda run -n dopingflow-grace dopingflow vacancies-mc-search -c input.toml --verbose
conda run -n dopingflow-mace  dopingflow vacancies-finalize  -c input.toml --verbose
```

The same `input.toml` and `output_directory` are used for both stages, so the
MACE finalize process continues from the GRACE archive and
`selected_candidates.txt` files written by Stage 1.

### Parent selection semantics

`parent_include` accepts composition labels independent of common element order
and decimal/`p` notation. For example, `Ti_2.5Sb_5`, `Ti2p5_Sb5`, and
`Sb5_Ti2p5` select the same composition. A selector containing `/`, such as
`Sb5_Ti2p5/candidate_003`, is treated as an exact parent ID.

`parent_pick = "lowest_energy"` keeps the first selected parent for each
composition. DopingFlow filtering writes `selected_candidates.txt` in ascending
relaxed-energy order, so this corresponds to the lowest-energy filtered parent.
Use `parent_pick = "all"` to keep every selected parent.

`output_directory` mirrors selected parent IDs into a dedicated staged result
tree while leaving the source parent calculations untouched.

### Monte Carlo schedule note

`mc_max_steps` is the total trajectory length; the hot hold and cooling ramp are
part of this number. In `combined`/`converged` mode the no-improvement patience
counter is active from step 1. If the full annealing schedule must be reached,
set

```text
mc_patience > mc_annealing_hold_steps + mc_annealing_steps
```

The page uses 200,000 maximum steps and 100,000 patience as a
production-oriented **starting point** for the present 2×2×2 / 960-atom coupled
occupation study. These values are not a universal convergence guarantee.

### Current GRACE → MACE selection limitation

The current staged path is:

```text
GRACE archive
    → GRACE ranking
    → GRACE top-k
    → MACE single point on GRACE-selected candidates
    → MACE relaxation
    → MACE relaxed-energy reranking
```

MACE does not yet rescore every archived GRACE candidate before the relaxation
top-k is chosen. The GUI therefore displays this limitation explicitly. Validate
GRACE/MACE ranking agreement on a smaller representative archive before a very
large production campaign.

---

## Phase Diagram page

`gui/pages/Phase_Diagram.py` is the dedicated phase-stability interface. It can
read the ordinary phase-diagram outputs and `vacancy_energy_above_hull.csv`,
switch between raw/corrected hulls when available, and plot vacancy-resolved
closed-system energy above hull.

A lower vacancy-containing energy above hull means the oxygen-deficient
composition is closer to its **closed-system decomposition hull**. This is not
the same quantity as a vacancy formation free energy and is not an oxygen-open
grand-potential hull.

---

## Vacancy energy-correction page

`gui/pages/Vacancy_Energy_Correction.py` controls whether an already fitted,
backend-compatible solid-energy correction is reused inside vacancy
thermodynamics. It preserves raw values and validates provenance.

When a fitted solid-energy correction is active, keep the oxygen reservoir on a
raw same-backend mode such as `reference_file` or `same_calculator`. Do not
combine it with the experimental `global` or `chemistry-specific`
oxygen-reference calibration path.

---

## Surface Screening page

`gui/pages/Surface_Screening.py` is the dedicated interface for the staged
surface workflow. Surface controls are intentionally no longer duplicated in
the main Input Builder; its Surface expander now links to this page and preserves
the existing `[surface]` section unchanged.

The page provides:

- bulk-candidate selection from `results_database.csv`;
- explicit low-index facets or automatic symmetrically distinct Miller indices;
- termination and slab-size/vacuum controls;
- representative surface/subsurface/bulk co-dopant placement scans;
- fixed middle/bottom slab regions during relaxation;
- independent fast-screen and higher-fidelity refinement calculators;
- direct execution in the current environment or separate named Conda environments;
- screening/refinement TOML preview and selected-parent preview;
- surface-energy and segregation-energy plots;
- per-surface structure browsing with the 3D structure viewer;
- raw screen, shortlist, refinement, and final-shortlist tables.

A common split-environment setup is:

```bash
conda activate dopingflow-grace
streamlit run gui/app.py
```

Then select **Current environment** on the Surface Screening page to run
`surface-scan`. For refinement, either launch the GUI from the MACE environment
or select **Named Conda environments** and provide the MACE environment name.

The segregation plot uses the same orientation, termination, composition, and
calculator for all compared variants:

```text
E_seg = E_variant - E_all-bulk-like
```

Negative values indicate that the requested surface/subsurface dopant placement
is favored relative to the generated all-bulk-like placement.

---


## Development and tests

Relevant files are:

```text
gui/
├── app.py
├── gui_config.py
├── vacancy_staged.py
├── phase_diagram_plots.py
├── vacancy_thermo_plots.py
├── pages/
│   ├── Surface_Screening.py
│   ├── Vacancy_MC_Staged.py
│   ├── Vacancy_Energy_Correction.py
│   └── Phase_Diagram.py
├── io_project.py
└── view_structure.py
```

Focused tests cover the staged vacancy and surface workflows. The surface CI also
compiles the main app and dedicated Surface Screening page, while the documentation
workflows keep the Sphinx guides synchronized.

---

## Notes

- The GUI reads/writes the same `input.toml` as the CLI.
- Large workflows should normally be launched from CLI/HPC after validating the
  configuration in the GUI.
- The staged GUI does not require GRACE and MACE to be importable in the same
  Python process; it uses separate environment commands.
- Phase-diagram conclusions are only as complete as the competing phases
  supplied to the calculation.

© 2026 Kazem Zhour

