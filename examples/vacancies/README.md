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

The example enables compact thermodynamic analysis using the existing reference
file. Its O2 backend/model/task metadata must match the resolved vacancy
calculator. The detailed database keeps every configuration; the plotting-ready
files keep composition/count minima, exact stability intervals, and best counts
at selected `delta_mu_O` values. Raw totals must not be compared across oxygen
contents.

Generate four figures using only those compact files:

```bash
python plot_vacancy_analysis.py \
  --minima random_structures/vacancy_minima_by_composition.csv \
  --intervals random_structures/vacancy_stability_intervals.csv \
  --best-counts random_structures/vacancy_best_counts.csv \
  --composition Sb10_Ti5 --delta-mu-o -1.0 --x-dopant Sb \
  --output-dir vacancy_plots
```
