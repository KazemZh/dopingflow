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
oxygen-reference calibration modes. A custom experimental CSV can be used
instead when a project must remain independent of the curated dataset.

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

The unified `vacancies` command determines a continuous charge-based oxygen-
vacancy range, symmetry-reduces configurations, screens all configurations and
relaxes the top-k at each fixed vacancy count with one shared M3GNet, UMA, MACE,
or GRACE calculator. It uses one flat `[vacancies]` section. To append it to
the normal pipeline, run `dopingflow run-all -c input.toml --until vacancies`.
Results are written separately to `<structure.outdir>/vacancies_database.csv`.
Comparing different vacancy counts thermodynamically requires an oxygen chemical
potential; raw ML total-energy differences alone are not vacancy formation energies.

Optional `[vacancies].static_thermodynamic_analysis = true` adds Level-1
static-lattice composition minima, exact oxygen-grand-potential intervals,
preferred counts, and a temperature–oxygen-pressure map. Legacy raw-reference
modes remain available, while two calibrated modes improve the absolute oxygen
reference without hard-coding a universal O2 correction:

```toml
[vacancies]
static_thermodynamic_analysis = true
oxygen_reference_mode = "global"            # or "chemistry-specific"
oxygen_reference_file = "reference_structures/reference_energies.json"
oxygen_calibration_experimental_source = "kingsbury"
oxygen_calibration_min_references = 2
solid_configurational_entropy = "none"       # optional: "ideal"
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
``solid_configurational_entropy = "ideal"`` optionally adds the ideal binary
occupied/vacant oxygen-site mixing entropy to T-dependent pressure maps only;
T-independent delta-mu stability intervals remain static-lattice quantities.
All other solid vibrational, zero-point, magnetic, electronic and anharmonic
terms remain outside this screening level.

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
- Select raw, global-calibrated, or chemistry-specific oxygen references for vacancy thermodynamics

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
│       ├── installation_and_usage.rst
│       ├── methods
│       │   ├── bandgap.rst
│       │   ├── database.rst
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
│       └── plot_vacancy_analysis.py
├── .github
│   └── workflows
│       └── docs.yml
├── .gitignore
├── gui
│   ├── app.py
│   ├── gui_config.py
│   ├── io_project.py
│   ├── README.md
│   └── view_structure.py
├── input.toml
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
│       ├── refs.py
│       ├── relax.py
│       ├── scan.py
│       ├── sequential.py
│       ├── surface.py
│       ├── vacancies.py
│       ├── vacancy_analysis.py
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