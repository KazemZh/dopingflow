# dopingflow GUI (Streamlit)

This folder contains the Streamlit-based graphical user interface for **dopingflow**.

The GUI provides an interactive way to:

- Build and edit `input.toml`
- Run workflow stages
- Monitor logs
- Visualize generated structures
- Explore the main database and per-system phase-diagram CSVs with Plotly
- Inspect raw/corrected phase-diagram stability in composition space
- Plot raw/corrected energy above hull against oxygen-vacancy count
- Configure and run the unified vacancy workflow and explore its separate database
- Choose Enumeration or Monte Carlo vacancy search with a configurable supercell
- Run Monte Carlo isothermally or enable a high-temperature hold and cooling ramp
- Compare relaxed parent, generated vacancy, and relaxed vacancy structures
- Configure calculator-verified, global-calibrated, or chemistry-specific oxygen references for vacancy thermodynamics
- Configure none, ideal-mixing, or exact-orbit partition-function vacancy entropy
- Select either the normal structure output or an existing directory containing many composition subdirectories

The GUI is optional. The CLI remains the primary interface for scripted and HPC workflows.

---

## Installation

The GUI dependencies are defined as an optional extra in the main `pyproject.toml`.

From the project root:

```bash
pip install -e ".[gui]"
```

If the vacancy oxygen calibration or experimental energy correction should use
the curated experimental dataset, also install:

```bash
pip install -e ".[corrections]"
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
Streamlit also discovers `gui/pages/Phase_Diagram.py`, so a dedicated **Phase
Diagram** page appears in the app navigation.

---

## GUI Pages Overview

### 1️⃣ Input Builder

Interactive editor for `input.toml`.

- Structure definition
- Doping setup (explicit or enumerate mode)
- Scan, Relax, Filter, Bandgap, Formation, optional Energy correction, and one
  flat Vacancies section
- Vacancy oxygen-reference selector including `global` and `chemistry-specific`
  calibration in addition to the existing raw/reference modes
- Conditional vacancy controls: enumeration and Monte Carlo parameters are shown
  only for the selected search, and annealing parameters appear only when enabled
- Live TOML preview
- Save directly to `input.toml`

The Energy-correction panel exposes the experimental source, optional custom
CSV and matminer cache path, model family, M1 scope, manifest/phase-resolved
selection, OPTIMADE endpoint, support/CV thresholds, conditioning, fit-quality
warning threshold, phase-mismatch override, provenance compatibility, and exact
fit reuse.

For oxide references the Input Builder uses the explicit oxygen convention:
`oxygen_reference_correction_ev` changes the electronic O2 reference, while
`delta_mu_O_ev` changes the physical oxygen chemical potential. The former must
remain zero when experimental energy correction is enabled. O-rich requires
`delta_mu_O_ev = 0`; O-poor permits values <= 0. A non-zero legacy
`muO_shift_ev` is shown only as a migration case and is never silently mixed
with the new keys.

These `[references]` oxygen controls are deliberately independent of the
`[vacancies]` oxygen-reference mode, delta-mu grid, and T-pO2 mapping controls.

---

### 2️⃣ Run

Graphical interface for:

```bash
dopingflow run-all
```

Supports full workflow execution, stage ranges, individual stages, vacancies,
optional overrides, and log monitoring.

---

### 3️⃣ Results Explorer

The Results Explorer can load the main results database, phase-diagram output,
vacancy-analysis tables, or a custom CSV.

For the existing phase-diagram view, candidate metadata matching is now
portable across machines: the plotting helper first uses the candidate path and
then falls back to the final `composition/candidate` path components. This
prevents copied calculations from failing merely because the original absolute
path came from another workstation or HPC filesystem.

If corrected hull columns are present, the phase plot offers Raw/Corrected
selection. The corrected view comes from a separately rebuilt complete hull; it
is not a post-hoc shift of raw energy above hull.

The vacancy thermodynamic panel provides grand-potential envelopes, preferred
vacancy count, static stability intervals, T-pO2 maps, and vacancy formation
free energies. These quantities remain distinct from energy above hull.

---

### 4️⃣ Structure Viewer

Visual inspection of generated structures using `py3Dmol`.

Useful for checking dopant placement, relaxed geometries, vacancy structures,
and Monte Carlo outputs.

---

### Phase Diagram page

The dedicated **Phase Diagram** page is the recommended interface for phase
stability analysis. It reads `phase_diagram_results.csv` and, when available,
`vacancy_energy_above_hull.csv`.

It provides:

- exact chemical-system selection;
- Raw/Corrected hull selection;
- a two-dopant composition map where marker color is energy above hull;
- one-dimensional concentration curves;
- stable-point highlighting;
- decomposition tables;
- corrected/raw energy above hull versus oxygen-vacancy count;
- composition selection for vacancy-hull curves.

The page also edits these `[phase_diagram]` options directly:

```toml
[phase_diagram]
include_vacancy_minima = true
vacancy_results_directory = "vacancy-selected"
```

After saving the settings, run:

```bash
dopingflow phase-diagram -c input.toml
```

The additional output is:

```text
vacancy_energy_above_hull.csv
```

For each cation composition, the file contains the lowest-energy relaxed
structure at each investigated oxygen-vacancy count and its raw/corrected energy
above hull and decomposition.

A lower energy above hull after introducing vacancies means that the
oxygen-deficient composition is closer to its closed-system decomposition hull.
This does **not** mean the same thing as a negative vacancy formation free
energy, and it is not yet an oxygen-open grand-potential hull.

---

## ⚠️ Notes

- The GUI assumes it is launched from the project root unless another project
  root is entered explicitly.
- It uses the same `input.toml` as the CLI.
- Large workflows are better executed from CLI or HPC systems.
- The GUI is intended for development, testing, and interactive analysis.
- Phase-diagram conclusions are only as complete as the competing phases
  included in the calculation.

---

## Development

Relevant GUI source files:

```text
gui/
├── app.py
├── gui_config.py
├── phase_diagram_plots.py
├── pages/
│   └── Phase_Diagram.py
├── io_project.py
└── view_structure.py
```

---

© 2026 Kazem Zhour