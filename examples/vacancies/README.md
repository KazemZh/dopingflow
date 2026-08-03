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

The example enables Level-1 static-lattice thermodynamic analysis using the
existing reference file. Solid free energies are approximated by 0 K relaxed ML
energies; temperature and pressure enter only through the oxygen reservoir. Raw
totals must not be compared across oxygen contents. Without a user standard-state
O2 correction, the pressure map is qualitative.
The example selects the built-in NIST O2 Shomate equations, which support any
requested temperature from 100 to 6000 K rather than only the discrete JANAF rows.

Generate five figures using only the new compact files:

```bash
python plot_static_vacancy_thermodynamics.py \
  --minima random_structures/vacancy_static_minima.csv \
  --intervals random_structures/vacancy_static_stability_intervals.csv \
  --best-counts random_structures/vacancy_static_best_counts.csv \
  --pressure-map random_structures/vacancy_static_pressure_map.csv \
  --composition Sb10_Ti5 --delta-mu-o -1.0 --temperature 900 \
  --x-dopant Sb --output-dir vacancy_static_plots
```
