# dopingflow GUI (Streamlit)

This folder contains the Streamlit-based graphical user interface for **dopingflow**.

The GUI provides an interactive way to:

- Build and edit `input.toml`
- Run workflow stages
- Monitor logs
- Visualize generated structures
- Explore the main database and per-system phase-diagram CSVs with Plotly
- Configure and run the unified vacancy workflow and explore its separate database
- Compare relaxed parent, generated vacancy, and relaxed vacancy structures
- Configure calculator-verified, global-calibrated, or chemistry-specific oxygen references for vacancy thermodynamics
- Select either the normal structure output or an existing directory containing many composition subdirectories

The GUI is optional. The CLI remains the primary interface for scripted and HPC workflows.

---

## Installation

The GUI dependencies are defined as an optional extra in the main `pyproject.toml`.

From the project root:

```bash
pip install -e ".[gui]"
```

If the vacancy oxygen calibration should automatically use the curated
experimental 298 K formation-enthalpy dataset, also install:

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
The GUI defaults continue to preserve those vacancy-analysis settings unchanged.

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
- Full workflow including vacancies and vacancies-only execution
- Optional overrides
- Log monitoring

---

### 3️⃣ Results Explorer

Selects `results_database.csv`, the combined phase-diagram result, the vacancy
database, or a custom CSV path and allows:

Choose **Phase diagram (energy above hull)** as the known result source, then
select the dopant for the x-axis and the dopant defining the fixed-concentration
curves. The matching co-doped system and compatible single-dopant boundary
systems are selected automatically. Each connected point is the minimum-energy
relaxed configuration at that composition, stars mark candidates on the convex
hull, and the pristine host is shown at the origin.

If corrected hull columns are present, the panel offers a Raw/Corrected switch.
The corrected view comes from a separately rebuilt complete hull; it is not a
post-hoc shift of raw energy above hull. The Input Builder keeps correction
disabled by default. Its model selector offers `manual`, forced `m0`, forced
`m1`, and `auto`. Manual mode exposes explicit correction terms. M0 is the
ordinary-oxide O term; M1 adds only independently supported
`oxide_cation:<Element>` terms, and auto mode falls back to parsimonious M0
unless M1 clears the configured leave-one-out improvement and one-standard-error
gates. `m1_elements = "workflow"` scopes candidate terms to the non-oxygen host
and all configured dopants; an explicit element array is also supported.

The calibration selector offers explicit `manifest` and complete
`phase_resolved` modes. Phase-resolved selection retains all strict,
non-generic ordinary oxide records whose non-oxygen elements are in the
host-and-dopant scope and that have
a curated `likely_mpid`. Missing structures can be fetched from the configured
Materials Project OPTIMADE endpoint into an immutable hash-validated cache.
Only geometry is fetched: calibration relaxation and hull filtering use the
selected ML backend, and doped candidates are not rerelaxed. The advanced panel
also exposes M1 compound/stoichiometry support, CV improvement, polyanion,
uncertainty, same-backend hull, conditioning, and exact-model reuse controls.
Disabling the hull filter writes an explicit `false` value rather than silently
restoring the 0.10 eV/atom default.

The correction panel compares the reference and candidate-relaxation
backend/model/task settings. These must align, and package-version or local
checkpoint changes require rebuilding references, refitting, and rerunning
stale relaxed candidates. A correction-enabled sequential run performs fit or
exact reuse once before its composition loop.

The known-source selector also detects static vacancy composition minima, exact
stability intervals, selected-condition best counts, and the
temperature-pressure map. Dedicated filters cover actual composition, dynamic
dopant percentages, vacancy count, ``delta_mu_O``, temperature, and ``pO2``.

When the compact static-lattice tables are present, a dedicated
``Vacancy thermodynamic plots`` panel provides interactive Plotly views of the
grand-potential envelope, grand potential versus vacancy count, preferred count
versus doping, the composition/oxygen-chemical-potential stability map, and an
T-pO2 map. The first four plots retain their original ``delta_mu_O``-based
definitions and controls. The T-pO2 tab has a standard-state selector, so that
map can be viewed with the configured ``delta_mu_O_standard(T)`` correction
or with the term intentionally omitted. Stability maps use fixed colors and
categorical legends. The temperature-pressure title and annotation identify whether the gas mapping uses
the NIST Shomate correction, a user table, or is approximate because the
standard-state thermal correction was omitted.

For calibrated oxygen-reference runs, every minima/pressure row records the
calibration scope, target chemistry, number of accepted reference oxides and fit
spread. `oxygen_calibration_report.json` provides the complete included/excluded
reference audit trail. The `global` mode uses all eligible ordinary binary
reference oxides; `chemistry-specific` refits using only oxides of the actual
host and present dopants. Neither mode invents a missing oxide stoichiometry.

The T-pO2 map combines the calibrated 298 K oxygen enthalpy reference with NIST
O2 gas enthalpy/entropy and pressure corrections. If
`solid_configurational_entropy = "ideal"`, the ideal occupied/vacant oxygen-site
mixing term is also applied in this T-dependent map; direct delta-mu plots remain
static-lattice quantities.

The Input Builder offers a continuous ``nist_shomate`` mode over 100--6000 K,
alongside custom ``user_table`` and qualitative ``none`` modes.

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