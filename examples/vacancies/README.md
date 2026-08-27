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

The checked-in example keeps the backward-compatible `reference_file` oxygen
reference so it can be used with an existing `reference_energies.json`. The same
flat section now also supports:

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