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
