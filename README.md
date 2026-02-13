
# dopingflow

**ML-driven doping workflow pipeline for high-throughput materials screening.**

`dopingflow` is a modular CLI-based workflow that automates:

- Structure generation  
- Symmetry-unique dopant enumeration  
- M3GNet screening  
- Relaxation  
- Filtering  
- ALIGNN bandgap prediction  
- Formation energy calculation  
- Final database collection  

Designed for reproducible, scalable materials screening.

---


# 📁 Project Structure

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
│       │   ├── minimal_example.rst
│       │   ├── production_run.rst
│       │   └── sb_codoping.rst
│       ├── index.rst
│       ├── input_file.rst
│       ├── methods
│       │   ├── bandgap.rst
│       │   ├── database.rst
│       │   ├── filtering.rst
│       │   ├── formation_energy.rst
│       │   ├── generation.rst
│       │   ├── relaxation.rst
│       │   └── scanning.rst
│       ├── _static
│       │   └── .gitkeep
│       ├── _templates
│       └── workflow_overview.rst
├── .github
│   └── workflows
│       └── docs.yml
├── .gitignore
├── input.toml
├── LICENSE
├── pyproject.toml
├── README.md
├── requirements.txt
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

---

# 🚀 Installation

## 1️⃣ Clone repository

```bash
git clone https://github.com/KazemZh/ml-doping-workflow.git
cd ml-doping-workflow
```

## 2️⃣ Create environment

```bash
conda create -n dopingflow python=3.11
conda activate dopingflow
```

## 3️⃣ Install package (editable mode)

Core only:

```bash
pip install -e .
```

With dev tools (pytest + ruff):

```bash
pip install -e ".[dev]"
```

With full ML stack:

```bash
pip install -e ".[m3gnet,alignn,mp,dev]"
```

---

# 📦 Optional Dependencies

| Extra      | Provides |
|------------|----------|
| `m3gnet`   | Structure relaxation & energy evaluation |
| `alignn`   | Bandgap prediction |
| `mp`       | Materials Project references |
| `dev`      | pytest + ruff |

---

# ⚙ Environment Variables

### ALIGNN model (required for Step 05)

```bash
export ALIGNN_MODEL_DIR=/path/to/your/alignn/model
```

### Materials Project (optional)

```bash
export MP_API_KEY=your_api_key
```

---

# 🔬 Workflow Steps

| Step | Command |
|------|--------|
| 00 | `dopingflow refs-build` |
| 01 | `dopingflow generate` |
| 02 | `dopingflow scan` |
| 03 | `dopingflow relax` |
| 04 | `dopingflow filter` |
| 05 | `dopingflow bandgap` |
| 06 | `dopingflow formation` |
| 07 | `dopingflow collect` |

---

# 🧪 Example Usage

```bash
dopingflow refs-build -c input.toml --verbose
dopingflow generate -c input.toml --verbose
dopingflow scan -c input.toml --verbose
dopingflow relax -c input.toml --verbose
dopingflow filter -c input.toml --verbose
dopingflow bandgap -c input.toml --verbose
dopingflow formation -c input.toml --verbose
dopingflow collect -c input.toml --verbose
```

---

# 🧾 Configuration

All behavior is controlled via:

```
input.toml
```

Main sections:

```
[structure]
[doping]
[generate]
[scan]
[relax]
[filter]
[bandgap]
[references]
```

---

# 🧪 Testing

Run:

```bash
pytest
```

---

# 🧹 Linting

```bash
ruff check .
```

---

# 📝 Logging

All runs write logs to:

```
logs/dopingflow.log
```

Use `--verbose` for detailed logs.

---

# 📜 License

This project is proprietary and confidential.

All rights reserved © 2026 Kazem Zhour.

No part of this repository may be used, copied, modified, or distributed
without explicit written permission from the author.

---

# 👤 Author

Kazem Zhour  
RWTH Aachen University  