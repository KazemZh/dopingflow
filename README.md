<p align="center">
  <img src="logo.png" width="500">
</p>

# dopingflow

**High-throughput ML-driven doping workflow for materials screening.**

`dopingflow` is a modular CLI pipeline for automated generation,
screening, relaxation, and evaluation of doped crystal structures using
machine-learning interatomic potentials and graph neural networks.

Designed for **reproducible, scalable materials discovery workflows**.

------------------------------------------------------------------------

## 📘 Documentation

The full documentation is available in multiple formats:

- 🌐 **Online HTML (auto-deployed via GitHub Actions):**  
  https://kazemzh.github.io/dopingflow/

- 📄 **User Guide (PDF):**  
  [Download dopingflow User Guide](dopingflow-user-guide.pdf)

------------------------------------------------------------------------

## Installation

### Clone repository

``` bash
git clone https://github.com/KazemZh/dopingflow.git
cd ml-doping-workflow
```

### Create environment

``` bash
conda create -n dopingflow python=3.11
conda activate dopingflow
```

### Install Packages

> ⚠️ Choose only one backend between **M3GNet** and **UMA**. They require incompatible versions of `numpy` and `ase`. Do not install both in the same environment!

#### MACE backend:

``` bash
pip install -e ".[mace]"
```

#### GRACE backend:

``` bash
pip install -e ".[grace]"
```

#### M3GNet backend:

``` bash
pip install -e ".[m3gnet]"
```

#### UMA backend:

``` bash
pip install -e ".[uma]"
```

Requires Hugging Face access (see setup below).

##### UMA Backend Setup

The UMA backend is provided through FAIR-Chem and requires access to the pretrained UMA models hosted on Hugging Face.

1. Request access to the UMA model repository  
   https://huggingface.co/facebook/UMA

2. Log in to Hugging Face

After access is granted, authenticate in your UMA environment:

```bash
hf auth login
```

#### ALIGNN Backend:

``` bash
pip install -e ".[alignn]"
```

#### GUI:

``` bash
pip install -e ".[gui]"
```

#### Material Project API:

``` bash
pip install -e ".[mp]"
```

#### Energy-correction and automatic oxygen-calibration support (optional):

``` bash
pip install -e ".[corrections]"
```

The same optional extra supplies the curated experimental 298 K formation-
enthalpy dataset used by the vacancy ``global`` and ``chemistry-specific``
oxygen-reference calibration modes and by the backend-specific M0/M1 energy-
correction fit. A custom experimental CSV can be used instead when a project
must remain independent of the curated dataset.

Configuration, scientific assumptions, and compatibility guidance are covered
in the full user documentation.

#### Development tools:

``` bash
pip install -e ".[dev]"
```

------------------------------------------------------------------------

## Required Environment Variables

### ALIGNN model directory (required for bandgap step)

``` bash
export ALIGNN_MODEL_DIR=/path/to/alignn/model
```

### Materials Project API (optional)

``` bash
export MP_API_KEY=your_api_key
```

------------------------------------------------------------------------

## Workflow Commands

Each stage can be run individually:

``` bash
dopingflow refs-build -c input.toml
dopingflow corrections-fit -c input.toml
dopingflow generate -c input.toml
dopingflow scan -c input.toml
dopingflow relax -c input.toml
dopingflow filter -c input.toml
dopingflow bandgap -c input.toml
dopingflow formation -c input.toml
dopingflow collect -c input.toml
dopingflow alloy-hull -c input.toml
dopingflow phase-diagram -c input.toml
dopingflow vacancies -c input.toml
dopingflow surface -c input.toml
```

Or run the complete pipeline:

``` bash
dopingflow run-all -c input.toml
```

The unified `vacancies` command determines a charge-based oxygen-vacancy range
(or uses an explicit `vacancy_counts` list), searches configurations by symmetry
enumeration or Monte Carlo, and relaxes/reranks the top-k at each fixed vacancy
count. The final/reference calculator may be M3GNet, UMA, MACE, or GRACE; Monte
Carlo can optionally use a separate fast calculator for occupation-search
energies. It uses one flat `[vacancies]` section. To append it to the normal
pipeline, run `dopingflow run-all -c input.toml --until vacancies`. Results are
written separately to `<structure.outdir>/vacancies_database.csv`. Comparing
different vacancy counts thermodynamically requires an oxygen chemical potential;
raw ML total-energy differences alone are not vacancy formation energies.

The symmetry search remains the default. A Metropolis Monte Carlo alternative can
jointly redistribute vacancies and any number of cation/dopant species on a chosen
supercell, archive low-energy unique occupations, then use the normal top-k
relaxation and relaxed-energy reranking pipeline. For example, GRACE can screen a
large occupation space while MACE remains the final thermodynamic calculator:

```toml
[vacancies]
search_method = "monte-carlo"       # default: "enumeration"
supercell = [2, 1, 1]
vacancy_counts = [1, 2]              # optional explicit research-design counts

mc_backend = "grace"
mc_model = "GRACE-1L-OMAT"
mc_task = ""
mc_device = "cuda"
mc_gpu_id = 0

mc_initial_temperature_K = 1500.0
mc_annealing = true
mc_annealing_hold_steps = 500
mc_annealing_steps = 2000
mc_temperature_K = 600.0             # target temperature
mc_run_mode = "combined"
mc_max_steps = 10000
mc_patience = 2000
mc_improvement_tolerance_eV = 1.0e-5
mc_energy_window_eV = 0.5
mc_cation_move_weight = 0.5
mc_vacancy_move_weight = 0.5
sample_seed = 42
sample_max_saved = 100
topk_per_vacancy_count = 15

# Final/reference calculator used for top-k relaxation and thermodynamics
backend = "mace"
model = "mh-1"
task = "matpes_r2scan"
device = "cuda"
gpu_id = 0
```

If the `mc_*` calculator keys are omitted, Monte Carlo inherits the final vacancy
calculator exactly as before. When they are supplied, the MC archive and top-k
selection use the dedicated search energies, while the selected structures are
relaxed and reranked with the ordinary `backend`/`model`/`task` calculator.
Search-energy provenance is recorded separately in `01_scan/meta.json`,
`ranking_scan.csv`, `vacancy_results.*`, and `monte_carlo_summary.json`; a
cross-backend search-to-relax energy difference is not reported as though both
energies came from the same model. Changing the explicit vacancy counts or MC
calculator participates in the vacancy fingerprint, so incompatible completed
searches are not silently reused.

With `mc_annealing = true`, the schedule holds `mc_initial_temperature_K` for
`mc_annealing_hold_steps`, cools linearly to `mc_temperature_K` over
`mc_annealing_steps`, and then continues at the target temperature. Set both step
counts to zero for an immediate transition. With `mc_annealing = false` (the
default), the entire search is performed at the constant `mc_temperature_K` and
all annealing parameters are ignored.

Optional `[vacancies].static_thermodynamic_analysis = true` adds Level-1
static-lattice composition minima, exact oxygen-grand-potential intervals,
preferred counts, and a temperature–oxygen-pressure map.

### Optional M0/M1 correction for oxygen-vacancy thermodynamics

The vacancy analysis can reuse the **same fitted backend-specific correction model**
used by corrected bulk formation energies and the corrected phase diagram. A practical
Kingsbury/automatic M0–M1 setup is:

```toml
[energy_correction]
enabled = true
experimental_source = "kingsbury"
model_family = "auto"
correction_terms = ["oxide"]
m1_elements = "workflow"
calibration_selection = "phase_resolved"
auto_fetch_phase_structures = true
reuse_fitted = true

[vacancies]
static_thermodynamic_analysis = true
apply_fitted_energy_correction = true
allow_legacy_energy_correction_provenance = false
oxygen_reference_mode = "reference_file"
oxygen_reference_file = "reference_structures/reference_energies.json"
```

Run the relevant stages in order:

```bash
dopingflow refs-build -c input.toml
dopingflow corrections-fit -c input.toml
dopingflow vacancies -c input.toml
```

The vacancy correction is applied as
`ΔE_corrected = ΔE_raw + C(defect) - C(parent)`; the vacancy stage does **not**
fit a second correction model. If `model_family = "auto"`, it uses the M0 or M1
family selected by `corrections-fit`.

For reproducibility, the correction fit and corrected vacancy energies must have
compatible provenance: backend, model, task, optimizer, force tolerance (`fmax`),
maximum relaxation steps, positive convergence, and available structure/energy
metadata are checked. `allow_legacy_energy_correction_provenance = true` accepts
missing historical metadata fields only; it does not waive a known mismatch.

When M0/M1 correction is active, use a **raw same-backend oxygen reservoir** such
as `reference_file` or `same_calculator`. Do not combine it with the vacancy
`global` or `chemistry-specific` oxygen-reference calibration, because both paths
use experimental formation-enthalpy information and would double count the
oxygen-related calibration.

Before corrected thermodynamic tables replace the active cross-count values,
dopingflow retains raw snapshots and also writes raw/corrected columns side by
side, including the selected correction family and `correction_fit_id`.

### Alternative calibrated oxygen reference (without M0/M1 vacancy correction)

Legacy raw-reference modes remain available, while two calibrated modes improve
the absolute oxygen reference without hard-coding a universal O2 correction:

```toml
[vacancies]
static_thermodynamic_analysis = true
oxygen_reference_mode = "global"            # or "chemistry-specific"
oxygen_reference_file = "reference_structures/reference_energies.json"
oxygen_calibration_experimental_source = "kingsbury"
oxygen_calibration_min_references = 2
solid_configurational_entropy = "none"       # optional: "ideal" or "configurational"
oxygen_standard_state_mode = "nist_shomate"
```

``global`` fits one backend/model/task-specific oxygen reference from every
eligible real ordinary binary oxide already calculated by `refs-build` and
having a matching experimental 298 K formation enthalpy. ``chemistry-specific``
performs the same fit separately for each vacancy chemistry, using only oxides
of the host and actually present dopant cations. Missing stoichiometries are
never invented. The fitted per-O values, included/excluded references, spread,
and formation-enthalpy residuals are written to
`oxygen_calibration_report.json`.

For calibrated references, the T-pO2 map adds the NIST O2 gas enthalpy/entropy
correction with a 298 K enthalpy origin and the ideal-gas pressure term. This is
separate from the zero-temperature/backend oxygen calibration.
``solid_configurational_entropy = "ideal"`` adds the ideal binary occupied/vacant
oxygen-site mixing entropy. ``"configurational"`` instead evaluates a canonical
partition function over the exact symmetry-distinct vacancy configurations and
their orbit degeneracies. If every exact configuration was relaxed, the relaxed
spectrum is used; otherwise the complete exact single-point spectrum provides
the configurational correction to the relaxed minimum. Sampled enumeration is
rejected for this mode because exact degeneracies are unavailable. Both entropy
treatments enter only finite-temperature outputs; direct delta-mu intervals remain
static-lattice quantities. Monte Carlo supports ``"none"`` and ``"ideal"`` only;
the explicit ``"configurational"`` mode requires ``search_method = "enumeration"``
with ``enumeration_mode = "exact"``. ``vacancy_formation_free_energy.csv/json`` reports
DeltaG_vac(T,pO2) for every vacancy count. Solid vibrational, zero-point, magnetic,
electronic and anharmonic terms remain outside this screening level.

``oxygen_standard_state_mode = "nist_shomate"`` evaluates continuous NIST O2
enthalpy/entropy corrections from 100 to 6000 K; ``user_table`` remains available
for alternative conventions and ``none`` remains a qualitative, approximate
pressure-only mode. Plot titles and metadata report which convention was used.
The Results Explorer preserves the original direct ``delta_mu_O`` plots. Only
the T-pO2 map compares including versus omitting ``delta_mu_O_standard(T)``;
the omitted-correction view is explicitly labeled approximate.

Existing multi-composition trees can be processed directly with
`parent_source = "directory"` and a flat `parent_directory` path under
`[vacancies]`. UMA model/task choices and the full supported GRACE model list are
available in the GUI. The GUI oxygen-reference selector also exposes the new
``global`` and ``chemistry-specific`` modes; advanced calibration data-source
fields can still be edited directly in `input.toml`.

For gradual composition-by-composition doping, use sequential-run. This reuses the lowest-energy relaxed structure from each composition as the base for the next composition:

``` bash
dopingflow sequential-run -c input.toml
```

------------------------------------------------------------------------

## Logging

Logs are written to:

    logs/dopingflow.log

Use `--verbose` for detailed output.


------------------------------------------------------------------------


## Graphical User Interface (Streamlit)

`dopingflow` provides an optional Streamlit-based graphical user interface for interactive workflow configuration, execution, and results analysis.

The GUI allows you to:

- Build and edit `input.toml`
- Run workflow stages interactively
- Visualize generated structures
- Explore `results_database.csv` and per-system phase-diagram CSVs with Plotly
- Configure optional formation-energy corrections
- Configure, run, explore, and compare parent/generated/relaxed vacancy structures
- Enable/disable fitted M0/M1 correction for vacancy thermodynamics and inspect raw vs corrected vacancy free energies
- Select raw, global-calibrated, or chemistry-specific oxygen references for vacancy thermodynamics
- Select no, ideal, or explicit partition-function vacancy configurational entropy
- Inspect raw/corrected phase diagrams and vacancy-resolved energy above hull

Relative-energy controls remain inside the existing `[formation]` section:

```toml
[formation]
relative_enabled = true
endpoint_x = "auto"
```

### Launch the GUI

From the project root directory:

```bash
streamlit run gui/app.py
```

Streamlit also discovers dedicated **Vacancy M0/M1 Energy Correction** and
**Phase Diagram** pages under `gui/pages/`.

After launching, a local browser window will open automatically.

------------------------------------------------------------------------

##  Project Structure

```
.
├── CHANGELOG.md
├── docs
│   ├── make.bat
│   ├── Makefile
│   └── source
│       ├── api
│       │   ├── dopingflow.rst
│       │   └── modules.rst
│       ├── examples
│       │   ├── enumerate_screening.rst
│       │   ├── explicit_batch.rst
│       │   ├── explicit_single_oxides.rst
│       │   ├── explicit_single.rst
│       │   ├── smoke_test.rst
│       │   ├── sequential_workflow.rst
│       │   └── vacancies.rst
│       ├── index.rst
│       ├── input_file.rst
│       ├── input_file_phase_diagram.rst
│       ├── input_file_vacancy_correction.rst
│       ├── installation_and_usage.rst
│       ├── methods
│       │   ├── bandgap.rst
│       │   ├── database.rst
│       │   ├── energy_corrections.rst
│       │   ├── filtering.rst
│       │   ├── formation_energy.rst
│       │   ├── generation.rst
│       │   ├── oxygen_calibration.rst
│       │   ├── phase_diagram.rst
│       │   ├── references.rst
│       │   ├── relaxation.rst
│       │   ├── sequential.rst
│       │   ├── scanning.rst
│       │   ├── surfaces.rst
│       │   ├── vacancy_energy_above_hull.rst
│       │   ├── vacancy_energy_correction.rst
│       │   └── vacancies.rst
│       ├── required_inputs.rst
│       ├── _static
│       │   ├── .gitkeep
│       │   └── logo.png
│       ├── _templates
│       └── workflow_overview.rst
├── dopingflow-user-guide.pdf
├── examples
│   ├── enumerate_screening
│   ├── explicit_batch
│   ├── explicit_single_composition
│   ├── explicit_single_composition_oxide_reference
│   ├── smoke_test
│   ├── surface_creation
│   └── vacancies
│       ├── input.toml
│       ├── README.md
│       ├── plot_static_vacancy_thermodynamics.py
│       └── plot_vacancy_analysis.py
├── .github
│   └── workflows
│       ├── docs.yml
│       └── vacancy-calibration-tests.yml
├── .gitignore
├── gui
│   ├── app.py
│   ├── gui_config.py
│   ├── pages
│   │   ├── Phase_Diagram.py
│   │   └── Vacancy_Energy_Correction.py
│   ├── phase_diagram_plots.py
│   ├── README.md
│   └── view_structure.py
├── LICENSE
├── logo.png
├── pyproject.toml
├── README.md
├── src
│   └── dopingflow
│       ├── bandgap.py
│       ├── cli.py
│       ├── collect.py
│       ├── filtering.py
│       ├── formation.py
│       ├── generate.py
│       ├── hardware.py
│       ├── __init__.py
│       ├── logging.py
│       ├── ml_backends.py
│       ├── ml_relaxation.py
│       ├── oxygen_calibration.py
│       ├── phase_diagram.py
│       ├── phase_diagram_convergence_extensions.py
│       ├── refs.py
│       ├── relax.py
│       ├── scan.py
│       ├── sequential.py
│       ├── surface.py
│       ├── vacancies.py
│       ├── vacancy_analysis.py
│       ├── vacancy_configurational_thermodynamics.py
│       ├── vacancy_energy_correction_extensions.py
│       ├── vacancy_mc_extensions.py
│       ├── vacancy_monte_carlo.py
│       ├── vacancy_static_thermodynamics.py
│       └── utils
│           ├── io.py
│           ├── parallel.py
│           ├── pymatgen_helpers.py
│           └── symmetry.py
└── tests
    ├── test_cli_help.py
    ├── test_cli.py
    ├── test_generate_minimal.py
    ├── test_vacancy_energy_correction.py
    ├── test_vacancy_mc_extensions.py
    ├── test_vacancy_phase_diagram_extension.py
    └── test_imports.py


```

------------------------------------------------------------------------

## License

Proprietary and confidential.

© 2026 Kazem Zhour\
RWTH Aachen University

Unauthorized use, modification, or distribution is prohibited.

------------------------------------------------------------------------

## Author

Kazem Zhour\
RWTH Aachen University
