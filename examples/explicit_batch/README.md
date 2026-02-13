# Explicit Example --- Multiple Target Compositions

This example runs the full workflow (00--07) for multiple explicitly
defined co-doped compositions in one run.

Each composition generates its own folder inside `random_structures/`.

## Run

```bash
dopingflow run-all -c input.toml
```


Results will be written to: 
- random_structures/
- results_database.csv

Each subfolder in random_strucutres/ (Sb5_Ti5/, Sb5_Zr5/, Sb10_Nb5/) contains full scan, relax, filter, bandgap, and formation results.
