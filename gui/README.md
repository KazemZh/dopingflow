# dopingflow GUI (Streamlit)

This folder contains the Streamlit-based graphical user interface for **dopingflow**.

The GUI provides an interactive way to:

- Build and edit `input.toml`
- Run workflow stages
- Monitor logs
- Visualize generated structures
- Explore `results_database.csv` interactively with Plotly

The GUI is optional. The CLI remains the primary interface for scripted and HPC workflows.

---

## Installation

The GUI dependencies are defined as an optional extra in the main `pyproject.toml`.

From the project root:

```bash
pip install -e ".[gui]"
```

If you also need ML models:

```bash
pip install -e ".[m3gnet,alignn,mp,gui]"
```

---

## Launching the GUI

From the project root directory:

```bash
streamlit run gui/app.py
```

A local browser window will open automatically (usually at http://localhost:8501).

---

## GUI Pages Overview

### 1️⃣ Input Builder

Interactive editor for `input.toml`.

- Structure definition
- Doping setup (explicit or enumerate mode)
- Scan, Relax, Filter, Bandgap, Formation settings
- Live TOML preview
- Save directly to `input.toml`

---

### 2️⃣ Run

Graphical interface for:

```bash
dopingflow run-all
```

Supports:

- Full workflow execution
- Stage range execution
- Single-stage execution
- Optional overrides
- Log monitoring

---

### 3️⃣ Results Explorer

Reads `results_database.csv` and allows:

- Column selection
- Interactive Plotly plotting
- Data filtering
- Scatter / line / bar plots

Ideal for rapid exploration of screening results without writing analysis scripts.

---

### 4️⃣ Structure Viewer

Visual inspection of generated structures using `py3Dmol`.

Useful for:

- Checking dopant placement
- Inspecting relaxed geometries
- Quick sanity checks

---

## ⚠️ Notes

- The GUI assumes it is launched from the project root.
- It uses the same `input.toml` as the CLI.
- Large workflows are better executed from CLI or HPC systems.
- The GUI is intended for development, testing, and interactive analysis.

---

## Development

GUI source files:

```
gui/
├── app.py
├── gui_config.py
├── io_project.py
└── view_structure.py
```

The layout and logic are defined in `app.py`.

---

© 2026 Kazem Zhour
