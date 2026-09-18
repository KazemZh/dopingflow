<p align="center">
  <img src="logo.png" width="500">
</p>

# dopingflow

**High-throughput ML-driven doping workflow for materials screening.**

`dopingflow` is a modular CLI pipeline for automated generation, screening,
relaxation, formation-energy analysis, phase stability, oxygen-vacancy studies,
and configurable oxidation-state analysis of doped crystal structures using
machine-learning interatomic potentials, graph neural networks, structural
chemistry, and optional DFT post-processing.

Designed for **reproducible, scalable materials-discovery workflows**.

---

## Documentation

- **Online HTML:** https://kazemzh.github.io/dopingflow/
- **Checked-in PDF user guide:** [dopingflow-user-guide.pdf](dopingflow-user-guide.pdf)
- **Oxidation-state guide:** [`docs/source/methods/oxidation_states.rst`](docs/source/methods/oxidation_states.rst)
- **Vacancy example:** [`examples/vacancies`](examples/vacancies)

---

## Installation

Clone the repository and create a Python 3.11 environment:

```bash
git clone https://github.com/KazemZh/dopingflow.git
cd dopingflow
conda create -n dopingflow python=3.11 pip -y
conda activate dopingflow
```

Install only the extras needed for the environment being used:

```bash
# MACE
pip install -e ".[mace]"

# GRACE
pip install -e ".[grace]"

# M3GNet
pip install -e ".[m3gnet]"

# UMA
pip install -e ".[uma]"

# ALIGNN bandgap support
pip install -e ".[alignn]"

# Oxidation-state optional Python dependencies
pip install -e ".[oxidation-toss]"    # TOSS-GNN Python deps
pip install -e ".[oxidation-chgnet]"  # CHGNet magnetic-moment analysis
pip install -e ".[oxidation-bertos]"  # BERTOS composition-token model

# GUI
pip install -e ".[gui]"

# Materials Project integration
pip install -e ".[mp]"

# Experimental energy-correction / oxygen-calibration datasets
pip install -e ".[corrections]"

# Development and tests
pip install -e ".[dev]"
```

### Recommended GPAW + oxidation GUI environment

GPAW is intentionally **not** declared as a pip optional dependency. On Linux,
`pip install gpaw` can fall back to a local source build and then require MPI
and compiler development headers. For the GPAW oxidation backend, use a
dedicated Conda environment and the precompiled conda-forge packages.

From the repository root, the recommended setup is:

```bash
conda create -n dopingflow_gpaw python=3.11 pip -y
conda activate dopingflow_gpaw
conda install -c conda-forge gpaw gpaw-data wannier90
pip install -e ".[gui]"
gpaw info
command -v wannier90.x
python -m streamlit run gui/app.py
```

`gpaw info` should complete successfully before starting a production GPAW
oxidation calculation. `wannier90` is installed in the same environment for the optional Wannier analysis route; `command -v wannier90.x` should resolve its executable. The same `dopingflow_gpaw` environment can then be
reused whenever the GPAW-backed oxidation page is needed.

The TOSS-GNN and BERTOS adapters also require local upstream repositories/model
files as documented in the oxidation-state guide; dopingflow does not silently
download or invent those external assets.

M3GNet and UMA have historically required incompatible dependency stacks; keep
backend environments isolated when their dependency requirements conflict.

### Separate GRACE and MACE environments for staged vacancy Monte Carlo

The staged vacancy workflow is specifically designed so GRACE and MACE **do not
need to coexist in one Python environment**. A typical setup is:

```bash
conda create -n dopingflow-grace python=3.11 pip -y
conda activate dopingflow-grace
pip install -e ".[grace]"

conda create -n dopingflow-mace python=3.11 pip -y
conda activate dopingflow-mace
pip install -e ".[mace]"
```

Both environments may use the same editable repository checkout.

UMA requires access to the pretrained FAIR-Chem models. After access is granted,
authenticate with:

```bash
hf auth login
```

### Optional environment variables

```bash
export ALIGNN_MODEL_DIR=/path/to/alignn/model
export MP_API_KEY=your_materials_project_api_key
```

---

## Workflow commands

Each stage can be run independently:

```bash
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
dopingflow vacancies-mc-search -c input.toml
dopingflow vacancies-finalize -c input.toml
dopingflow oxidation -c input.toml
dopingflow surface -c input.toml
```

The ordinary pipeline can be run with:

```bash
dopingflow run-all -c input.toml
```

`dopingflow vacancies` remains the one-process vacancy command. Use it for
symmetry enumeration or for Monte Carlo when all requested calculators are
available in the same Python environment.

For a GRACE-search/MACE-finalization workflow with incompatible environments,
use the two staged commands described below.

---

## Staged oxygen-vacancy Monte Carlo: GRACE → MACE

The staged workflow separates the expensive occupation search from the final
relaxation/thermodynamic calculator:

```text
source relaxed parent(s)
        ↓
2×2×2 replicated search cell
        ↓
GRACE joint cation + oxygen-vacancy Monte Carlo
        ↓
low-energy GRACE archive
        ↓
GRACE ranking and GRACE top-k selection at fixed vacancy count
        ↓
MACE single point on selected candidates
        ↓
MACE relaxation and relaxed-energy reranking
        ↓
MACE vacancy thermodynamics
```

### Stage 1 — GRACE environment

```bash
conda activate dopingflow-grace
dopingflow vacancies-mc-search -c input.toml --verbose
```

This command checks/builds only the `mc_*` calculator. It does not require MACE
in the GRACE environment.

### Stage 2 — MACE environment

```bash
conda activate dopingflow-mace
dopingflow vacancies-finalize -c input.toml --verbose
```

This command checks/builds only the ordinary final vacancy calculator. It loads
the persisted Stage-1 selection and continues from the same search tree.

### Parent selection and dedicated output tree

The staged workflow supports composition filtering, one-parent-per-composition
selection, and a dedicated result tree:

```toml
[vacancies]
parent_source = "directory"
parent_directory = "vacancy-selected"

parent_include = [
    "Ti_2.5Sb_2.5",
    "Ti_2.5Sb_5",
    "Ce_2.5Sb_2.5",
]
parent_pick = "lowest_energy"     # or "all"
output_directory = "vacancy-mc-grace-mace"
```

`parent_include` accepts common composition labels independently of element
order and the usual `2.5`/`2p5` notation. Thus `Ti_2.5Sb_5`, `Ti2p5_Sb5`, and
`Sb5_Ti2p5` match the same composition. A selector containing `/`, for example
`Sb5_Ti2p5/candidate_003`, is treated as an exact parent ID.

`parent_pick = "lowest_energy"` keeps the first selected candidate for each
composition. DopingFlow filtering writes `selected_candidates.txt` in ascending
relaxed-energy order, so this is the lowest-energy filtered parent.

`output_directory` keeps the complete staged vacancy study separate from the
source parent tree. GRACE writes its archive there and MACE resolves the same
mirrored parent IDs during finalization.

### Explicit vacancy counts and 2×2×2 search cell

The current large-supercell research profile studies one through four oxygen
vacancies explicitly:

```toml
max_vacancies_cap = 4
vacancy_counts = [1, 2, 3, 4]
supercell = [2, 2, 2]
```

For a 120-atom SnO2-based parent containing 40 cations and 80 oxygen atoms, the
`2×2×2` replication contains 960 atoms before vacancy removal: 320 cations and
640 oxygen sites. Dopant counts are multiplied by eight while their percentages
remain unchanged.

One, two, three, and four vacancies correspond to 0.15625%, 0.3125%, 0.46875%,
and 0.625% of the oxygen sublattice, respectively.

### Coupled cation/vacancy moves

Monte Carlo samples two move classes:

- `vacancy_swap`: move the vacancy marker between oxygen sites;
- `cation_swap`: exchange two different cation species.

For coupled dopant-vacancy ordering keep both weights non-zero:

```toml
mc_cation_move_weight = 0.5
mc_vacancy_move_weight = 0.5
```

The host and any number of dopant species are handled generically. Internal
vacancy markers are removed before ML energy calls.

### GRACE search settings

A production-oriented starting point for the present 960-atom coupled
occupation space is:

```toml
search_method = "monte-carlo"

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
mc_max_steps = 500000
mc_patience = 105000
mc_improvement_tolerance_eV = 0.001
mc_energy_window_eV = 1.0
mc_cation_move_weight = 0.5
mc_vacancy_move_weight = 0.5
sample_seed = 42
sample_max_saved = 100
```

`mc_max_steps` is the total trajectory ceiling, including the high-temperature
hold, the cooling ramp, and the final-temperature sampling. The current MC
implementation counts `mc_patience` from step 1. With a 5,000-step hold and a
50,000-step ramp, `mc_patience = 105000` ensures that a no-improvement stop
cannot occur before the annealing schedule plus roughly 50,000 additional trial
moves at 600 K. Any qualifying new global minimum resets that counter.

`mc_improvement_tolerance_eV = 0.001` means the best-so-far energy must improve
by at least 1 meV to reset the convergence clock. These values are a practical
starting profile for the current study, not a universal convergence guarantee.
Independent seeds and energy/motif agreement remain important convergence
checks.

### MACE finalization settings

```toml
backend = "mace"
model = "mh-1"
task = "matpes_r2scan"
device = "cuda"
gpu_id = 0

topk_per_vacancy_count = 20
optimizer = "bfgs"
fmax = 0.05
max_steps = 300
relax_mode = "atoms"
```

Stage 1 writes a GRACE low-energy archive and selects `topk_per_vacancy_count`
structures independently for each parent and each vacancy count. Stage 2 applies
MACE single points and relaxations only to those selected structures, then
reranks them by MACE relaxed energy.

**Current selection limitation:** MACE does not yet rescore every archived GRACE
candidate before top-k selection. A structure discarded by the GRACE ranking is
not reconsidered by MACE. Validate GRACE/MACE ranking agreement on a smaller
representative archive before a large production campaign.

### Search/final provenance and handoff

GRACE search energy and MACE final energy are recorded separately. A
cross-backend GRACE→MACE energy difference is not labeled as a same-calculator
relaxation energy. Search fingerprints include the search space, explicit
vacancy counts, MC calculator settings, and parent-structure hashes; final
calculator/relaxation settings are fingerprinted separately.

Typical staged global files are:

```text
<output_directory>/vacancy_mc_search_database.csv
<output_directory>/vacancy_mc_search_database.json
<output_directory>/vacancies_database.csv
<output_directory>/vacancies_database.json
```

Each mirrored parent receives `05_vacancies/` with `V_O_01` through `V_O_04`,
count-specific archives, rankings, `selected_candidates.txt`, MACE finalization
metadata, and thermodynamic outputs. The MACE finalize stage reads the persisted
GRACE selections from this same `output_directory`; it does not restart the
search from the source parent tree.

### Reference-state caveat when cations move

If cation swaps are enabled, defective `n=1,2,3,4` minima can differ from the
replicated source parent in both vacancy location and cation ordering. The
current `n=0` parent reference is not subjected to an independent cation-only
Monte Carlo search. Consequently, a vacancy formation free energy from such a
run can contain both vacancy-formation and cation-reordering contributions.

This is appropriate for a coupled ordering study. A strict vacancy formation
energy referenced to an equilibrated cation arrangement would require a
corresponding `n=0` cation-only equilibrium baseline.

---

## Vacancy thermodynamics

Enable static vacancy thermodynamics with:

```toml
[vacancies]
include_parent_reference = true
static_thermodynamic_analysis = true
```

The parent reference is required for cross-count vacancy thermodynamics. For a
fixed cation composition, DopingFlow compares final relaxed minima through an
oxygen reservoir rather than raw total-energy differences:

```text
ΔG_vac(n,T,pO2)
  = E_min(n) - E_min(0)
  + n μO(T,pO2)
  + ΔF_config(n,T)
```

The direct `delta_mu_O` analysis gives static-lattice grand-potential crossing
intervals. The finite-temperature pressure map can add NIST O2 Shomate
enthalpy/entropy terms and the ideal-gas pressure contribution.

For Monte Carlo:

```toml
solid_configurational_entropy = "none"   # static lattice
# or
solid_configurational_entropy = "ideal"  # ideal occupied/vacant mixing
```

The explicit canonical partition-function option `"configurational"` requires
exact symmetry enumeration and exact orbit degeneracies; it is not available
for Monte Carlo sampling.

### Oxygen reference calibration

Without a fitted solid-energy correction, vacancy thermodynamics may use either
raw/reference oxygen modes or experimentally calibrated modes:

```toml
oxygen_reference_mode = "reference_file"
# or: "same_calculator"
# or: "global"
# or: "chemistry-specific"
```

`global` fits an effective backend/model/task-specific oxygen reference from
eligible calculated binary oxides and experimental 298 K formation enthalpies.
`chemistry-specific` performs the same fit using only oxides relevant to the
actual host/dopant chemistry. The audit trail is written to
`oxygen_calibration_report.json`.

### Optional fitted energy correction in vacancy thermodynamics

If a compatible backend-specific experimental correction model has already been
fitted, the vacancy analysis can opt in:

```toml
apply_fitted_energy_correction = true
allow_legacy_energy_correction_provenance = false
oxygen_reference_mode = "reference_file"
```

The solid reaction correction is applied as
`ΔE_corrected = ΔE_raw + C(defect) - C(parent)`.

When this path is active, use a **raw same-backend oxygen reservoir** such as
`reference_file` or `same_calculator`. Do not combine it with the experimental
`global` or `chemistry-specific` oxygen-reference calibration because both paths
use experimental formation-enthalpy information for oxygen-related calibration.

---

## Oxidation-state analysis

The oxidation stage is independent of the ML potential used for structural
relaxation. The same MACE-, UMA-, GRACE-, or M3GNet-relaxed structures can be
analyzed with structural, ML, DFT, or explicitly combined strategies.

Available methods are:

- **Structural:** pymatgen bond valence.
- **ML:** pretrained TOSS-GNN, CHGNet magnetic-moment analysis, and BERTOS.
- **DFT:** GPAW single-point electronic descriptors, GPAW all-electron-density Bader analysis, Wannier descriptors, and validated
  EOS/charge-pumping formal assignments.
- **Combined:** an explicit set of methods from multiple groups. Every method is
  retained separately; labels are not averaged and no majority vote is used.

A small structural smoke test can start with:

```toml
[oxidation]
enabled = true
strategy = "structural"
methods = ["bond-valence"]
include_vacancy_free = true
include_oxygen_vacancies = true
# Optional: restrict expensive analysis to exact target IDs / safe IDs / glob patterns.
# target_include = ["Sb5_Ti2p5/candidate_014"]
output_dir = "06_oxidation"
mapping_tolerance = 1.2
fail_fast = false
```

Run it with:

```bash
dopingflow oxidation -c input.toml --strategy structural --methods bond-valence
```

For open-source DFT electronic descriptors, use the dedicated GPAW environment
described in the Installation section. In short:

```bash
conda create -n dopingflow_gpaw python=3.11 pip -y
conda activate dopingflow_gpaw
conda install -c conda-forge gpaw gpaw-data wannier90
pip install -e ".[gui]"
gpaw info
command -v wannier90.x
python -m streamlit run gui/app.py
```

Then configure one reusable GPAW parameter set rather than per-structure DFT
input files:

```toml
[oxidation.dft_electronic]
code = "gpaw"
output_root = "dft_oxidation"
execute = false              # true runs the single point directly
mode = "pw"
ecut_eV = 500.0
xc = "PBE"
kpts = [1, 1, 1]
gamma = true
smearing_eV = 0.05
convergence_density = 1e-5
spinpol = "auto"
```

The generated per-target GPAW directory can contain `oxidation.gpw`, `gpaw.txt`, `dos.csv`, `pdos_integrals.json`, `magnetic_moments.csv`, and `electronic_summary.json`. Cutoff, k-point, spin, smearing, and convergence settings must be converged for the target chemistry.

The stage can analyze both selected vacancy-free parents and relaxed oxygen
vacancy structures. `[oxidation].target_include` can restrict a run to one or more
exact target IDs, safe IDs (`/` represented as `__`), or shell-style glob patterns.
This is especially useful before enabling expensive GPAW, Bader, Wannier, or EOS
execution. When valid same-element parent mapping is available it also reports
parent-relative changes.

Important interpretation rules are enforced in the implementation:

- Bader charges are continuous descriptors and never become formal integer
  oxidation states automatically.
- CHGNet magnetic moments are stored separately from inferred oxidation labels;
  unsupported or ambiguous sites remain unresolved.
- BERTOS is composition-token level and is never fabricated into site-resolved
  assignments.
- Static Wannier centers are descriptors. With ``execute = true``, the native
  GPAW/Wannier90 route can generate them automatically for an isolated,
  non-spin-polarized Gamma-only occupied manifold from an ``oxidation.gpw``
  that contains wavefunctions. It detects the occupied bands, uses Bloch phases
  as the initial gauge (no arbitrary atomic projection choice), writes the GPAW
  eigenvalue/overlap files, runs ``wannier90.x``, and parses ``*_centres.xyz``.
  Metallic, spin-polarized, multi-k, or entangled cases still require an explicit
  projection/disentanglement workflow. Formal DFT labels are accepted only from
  an explicitly validated EOS/charge-pumping result with a documented assignment
  procedure.
- New GPAW single-point calculations are opt-in: `execute = false` is the safe default.
  When enabled, dopingflow runs GPAW directly from the relaxed structure; Bader execution
  separately requires the free `bader` executable.

Outputs are written under `[oxidation].output_dir` and include
`oxidation_results.json`, `oxidation_sites.csv`, `oxidation_comparison.json`,
`dft_followup_candidates.json`, and `meta.json`.

See the complete seven configuration examples and method-specific caveats in
[`docs/source/methods/oxidation_states.rst`](docs/source/methods/oxidation_states.rst).

---

## Recommended smoke test before a production campaign

For the first run on a new machine/backend combination, temporarily reduce the
study to one composition and a tiny MC trajectory:

```toml
parent_include = ["Ti_2.5Sb_2.5"]
vacancy_counts = [1]
mc_max_steps = 20
mc_patience = 20
sample_max_saved = 10
topk_per_vacancy_count = 3
```

Run GRACE search, inspect the generated tree, then run MACE finalization. After
that handoff succeeds, restore the `[1, 2, 3, 4]` production profile and benchmark
roughly 1,000–5,000 GRACE MC steps on one 960-atom composition before estimating
the full campaign cost.

---

## Graphical user interface

Install and launch:

```bash
pip install -e ".[gui]"
streamlit run gui/app.py
```

In addition to the main Input Builder/Run/Results pages, Streamlit discovers
dedicated pages under `gui/pages/`, including:

- **Oxidation States** — configures structural/ML/DFT/combined strategies,
  individual methods, parent and oxygen-vacancy targets, TOSS-GNN/CHGNet/BERTOS
  settings, GPAW/Bader/Wannier/EOS analysis, and optional candidate-limited
  DFT follow-up. It can save `[oxidation]`, run `dopingflow oxidation`, and inspect
  the generated CSV/JSON results. GPAW and Bader execution remain behind explicit
  method-level and follow-up `execute` gates plus a GUI confirmation.
- **Staged Vacancy MC** — edits `parent_include`, `parent_pick`,
  `output_directory`, `vacancy_counts`, `supercell`, GRACE `mc_*` settings,
  production MC controls, and MACE finalization settings. Its defaults reflect
  the current `[1,2,3,4]`, 500k/105k, 1 meV research profile, while existing
  values in `input.toml` are preserved when present. It also builds and can
  execute separate `conda run` commands for the GRACE and MACE environments.
- **Vacancy M0/M1 Energy Correction** — configures application of an already
  fitted correction to vacancy thermodynamics.
- **Phase Diagram** — explores raw/corrected hull results and vacancy-resolved
  energy above hull.
- **Oxidation States** — configures structural/ML/DFT oxidation analysis and provides a
  structure browser: select an analyzed parent or vacancy structure, choose a method,
  and inspect atom-by-atom oxidation states/descriptors. Results are also written per
  structure under ``06_oxidation/structures/<target_id>/``.

The staged GUI page explicitly warns that current top-k selection is performed
with GRACE before MACE finalization.

---

## Sequential workflow

For gradual composition-by-composition doping:

```bash
dopingflow sequential-run -c input.toml
```

This reuses the lowest-energy relaxed structure from each composition as the
base for the next composition.

---

## Logging

Logs are written below the project `logs/` directory. Use `--verbose` for more
detailed CLI output.

---

## Repository layout

```text
README.md
CHANGELOG.md
docs/
examples/
  vacancies/
gui/
  app.py
  vacancy_staged.py
  pages/
    Oxidation_States.py
    Vacancy_MC_Staged.py
    Vacancy_Energy_Correction.py
    Phase_Diagram.py
src/dopingflow/
  oxidation.py
  oxidation_structural.py
  oxidation_ml.py
  oxidation_dft.py
  vacancies.py
  vacancy_monte_carlo.py
  vacancy_mc_extensions.py
  vacancy_mc_staged.py
  vacancy_parent_selection_extensions.py
  vacancy_static_thermodynamics.py
tests/
```

---

## License

Proprietary and confidential.

© 2026 Kazem Zhour, RWTH Aachen University.

Unauthorized use, modification, or distribution is prohibited.
