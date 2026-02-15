# Enumerate Example — Systematic Dopant Screening

This example performs combinatorial screening of dopants and concentrations.

Use this mode when:
- You want automatic composition generation.
- You define allowed dopants and concentration levels.
- You limit the maximum number of dopants per structure.

Note:
The current implementation supports up to 3 dopants per structure via:
max_dopants_total = 3

## Run

```bash
dopingflow run-all -c input.toml
```

Expected Result:

```
random_structures/<composition>/
    ranking_scan.csv
    ranking_relax.csv
    ranking_relax_filtered.csv
    ...
```