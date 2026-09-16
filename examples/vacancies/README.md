# Oxygen vacancies

This example adds the single flat `[vacancies]` section to a project that has
already completed filtering and therefore contains `selected_candidates.txt`
and selected `candidate_*/02_relax/POSCAR` files.

Run:

```bash
dopingflow vacancies -c input.toml
```

Each selected candidate receives `05_vacancies/`, while the structure output
root receives `vacancies_database.csv` and `vacancies_database.json`. Model
weights are obtained by the selected backend on first use; choose a backend and
model installed in your environment.

The example uses the default symmetry search explicitly:

```toml
search_method = "enumeration"
supercell = [1, 1, 1]
```

Switch to `search_method = "monte-carlo"` to sample vacancy–anion swaps and
swaps between every distinct cation species in the relaxed parent. This supports
more than two dopants and retains the same ML screening, top-k relaxation,
reranking, and output layout. `supercell = [a, b, c]` controls the search cell
for either method.

For an isothermal search:

```toml
search_method = "monte-carlo"
mc_temperature_K = 300.0
mc_annealing = false
```

For annealing:

```toml
mc_temperature_K = 300.0
mc_annealing = true
mc_initial_temperature_K = 1200.0
mc_annealing_hold_steps = 500
mc_annealing_steps = 2000
```

The stopping, archive, tolerance, and move-weight controls are commented in
`input.toml`. Each vacancy-count directory records the resolved schedule and
acceptance statistics in `monte_carlo_summary.json`.

## Optional M0/M1 correction for vacancy thermodynamics

The checked-in example now includes the complete opt-in correction block, but
keeps it disabled so the example remains usable as a raw vacancy-only input:

```toml
[energy_correction]
enabled = false
experimental_source = "kingsbury"
model_family = "auto"
correction_terms = ["oxide"]
m1_elements = "workflow"
calibration_selection = "phase_resolved"
auto_fetch_phase_structures = true
reuse_fitted = true
```

To apply the same fitted backend-specific M0/M1 model used by corrected
formation energies and the corrected phase diagram to the vacancy parent and
vacancy minima, set `enabled = true`, fit the model first, and then enable the
vacancy option:

```toml
[vacancies]
static_thermodynamic_analysis = true
apply_fitted_energy_correction = true
allow_legacy_energy_correction_provenance = false
oxygen_reference_mode = "reference_file"
oxygen_reference_file = "reference_structures/reference_energies.json"
```

Run in this order:

```bash
dopingflow refs-build -c input.toml
dopingflow corrections-fit -c input.toml
dopingflow vacancies -c input.toml
```

The vacancy correction is
`ΔE_corrected = ΔE_raw + C(defect) - C(parent)`. It reuses the already fitted
model and therefore preserves one correction definition across ordinary bulk
formation/phase-diagram analysis and vacancy thermodynamics.

The correction fit and the corrected candidate/vacancy energies must have
compatible provenance. Known differences in backend, model, task, optimizer,
force tolerance (`fmax`), or `max_steps` are rejected. The legacy-provenance
option only accepts fields that were not recorded by older runs; it does not
waive a known mismatch.

The fitted M0/M1 vacancy correction must **not** be combined with
`oxygen_reference_mode = "global"` or `"chemistry-specific"`, because those
oxygen-reference modes also use experimental formation enthalpies and the two
routes would double count oxygen-related calibration. Use a raw same-backend
reservoir such as `reference_file` or `same_calculator` when M0/M1 correction is
active.

## Alternative calibrated oxygen reference

The checked-in example keeps the backward-compatible `reference_file` oxygen
reference so it can be used with an existing `reference_energies.json`. When the
M0/M1 vacancy correction is **off**, the same flat section also supports:

```toml
oxygen_reference_mode = "global"
# or
oxygen_reference_mode = "chemistry-specific"
```

These calibrated modes do **not** reuse a universal O2 correction. They derive
an effective per-O reference from real ordinary binary oxides already calculated
by `refs-build`, matching bulk-metal references, and experimental 298 K
formation enthalpies. `global` uses all eligible reference oxides;
`chemistry-specific` uses only oxides of the actual host/dopant cations in each
vacancy composition. The full audit trail is written to
`oxygen_calibration_report.json`.

The default automatic experimental source is the curated Kingsbury dataset and
requires:

```bash
pip install -e ".[corrections]"
```

A project may instead select `custom` or `kingsbury+custom` and provide an
explicit experimental CSV.

The example selects the built-in NIST O2 Shomate equations, which support any
requested temperature from 100 to 6000 K rather than only discrete JANAF rows.
For calibrated oxygen references, the T-pO2 map uses the 298 K enthalpy origin
consistent with the experimental formation-enthalpy fit, then adds the gas-phase
O2 enthalpy/entropy and pressure terms.

Solid configurational entropy remains optional:

```toml
solid_configurational_entropy = "none"   # default
# solid_configurational_entropy = "ideal"
# solid_configurational_entropy = "configurational"
```

`ideal` adds the binary occupied/vacant oxygen-site mixing entropy.
`configurational` uses a canonical partition function over exact symmetry-distinct
vacancy configurations and orbit degeneracies. It requires exact enumeration; if
a calculation was sampled, the analysis stops rather than guessing degeneracies.
It is also incompatible with Monte Carlo sampling; Monte Carlo runs must select
`none` or `ideal`.
When the whole exact set is relaxed, relaxed energies enter the partition function;
otherwise the full exact single-point spectrum supplies a configurational correction
to the relaxed static minimum. Both treatments affect finite-temperature outputs,
including `vacancy_formation_free_energy.csv/json` and the T-pO2 stability map.
Direct delta-mu intervals remain static-lattice quantities. Vibrational, zero-point,
magnetic, thermal-electronic, anharmonic, thermal-expansion and solid-pV
contributions are not part of this screening level.

Generate five figures using only the compact files:

```bash
python plot_static_vacancy_thermodynamics.py \
  --minima random_structures/vacancy_static_minima.csv \
  --intervals random_structures/vacancy_static_stability_intervals.csv \
  --best-counts random_structures/vacancy_static_best_counts.csv \
  --pressure-map random_structures/vacancy_static_pressure_map.csv \
  --composition Sb10_Ti5 --delta-mu-o -1.0 --temperature 900 \
  --x-dopant Sb --output-dir vacancy_static_plots
```
