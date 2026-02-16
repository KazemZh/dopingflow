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

## Documentation

-   Full User Guide (PDF):\
    👉 **[Download dopingflow User Guide](dopingflow-user-guide.pdf)**

-   Detailed documentation (Sphinx source):\
    `docs/`

------------------------------------------------------------------------

## Installation

### Clone repository

``` bash
git clone https://github.com/KazemZh/ml-doping-workflow.git
cd ml-doping-workflow
```

### Create environment

``` bash
conda create -n dopingflow python=3.11
conda activate dopingflow
```

### Install

Full ML stack:

``` bash
pip install -e ".[m3gnet,alignn,mp,gui]"
```

Core package:

``` bash
pip install -e .
```

Development tools:

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
dopingflow generate -c input.toml
dopingflow scan -c input.toml
dopingflow relax -c input.toml
dopingflow filter -c input.toml
dopingflow bandgap -c input.toml
dopingflow formation -c input.toml
dopingflow collect -c input.toml
```

Or run the complete pipeline:

``` bash
dopingflow run-all -c input.toml
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
- Explore `results_database.csv` with Plotly

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
│   ├── build
│   ├── make.bat
│   ├── Makefile
│   └── source
│       ├── api
│       │   ├── dopingflow.rst
│       │   └── modules.rst
│       ├── conf.py
│       ├── examples
│       │   ├── enumerate_screening.rst
│       │   ├── explicit_batch.rst
│       │   ├── explicit_single.rst
│       │   └── smoke_test.rst
│       ├── index.rst
│       ├── input_file.rst
│       ├── installation_and_usage.rst
│       ├── methods
│       │   ├── bandgap.rst
│       │   ├── database.rst
│       │   ├── filtering.rst
│       │   ├── formation_energy.rst
│       │   ├── generation.rst
│       │   ├── references.rst
│       │   ├── relaxation.rst
│       │   └── scanning.rst
│       ├── required_inputs.rst
│       ├── _static
│       │   ├── .gitkeep
│       │   └── logo.png
│       ├── _templates
│       └── workflow_overview.rst
├── dopingflow-user-guide.pdf
├── examples
│   ├── enumerate_screening
│   │   ├── input.toml
│   │   └── README.md
│   ├── explicit_batch
│   │   ├── input.toml
│   │   └── README.md
│   ├── explicit_single_composition
│   │   ├── input.toml
│   │   └── README.md
│   └── smoke_test
│       ├── input.toml
│       └── README.md
├── .github
│   └── workflows
│       └── docs.yml
├── .gitignore
├── gui
│   ├── app.py
│   ├── gui_config.py
│   ├── io_project.py
│   ├── README.md
│   ├── requirements-gui.txt
│   └── view_structure.py
├── input.toml
├── LICENSE
├── logo.png
├── pyproject.toml
├── README.md
├── results_database.csv
├── src
│   ├── dopingflow
│   │   ├── bandgap.py
│   │   ├── cli.py
│   │   ├── collect.py
│   │   ├── filtering.py
│   │   ├── formation.py
│   │   ├── generate.py
│   │   ├── __init__.py
│   │   ├── logging.py
│   │   ├── refs.py
│   │   ├── relax.py
│   │   ├── scan.py
│   │   └── utils
│   │       ├── io.py
│   │       ├── parallel.py
│   │       └── pymatgen_helpers.py
│   └── dopingflow.egg-info
│       ├── dependency_links.txt
│       ├── entry_points.txt
│       ├── PKG-INFO
│       ├── requires.txt
│       ├── SOURCES.txt
│       └── top_level.txt
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