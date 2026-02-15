# Smoke Test — Minimal Fast Run

This example performs a fast, reduced-cost test of the pipeline.

It is intended to verify that installation and core workflow steps work correctly.

This configuration:
- Uses a smaller supercell
- Uses small topk
- Uses top-N filtering
- Stops after step 04

## Run

```bash
dopingflow run-all -c input.toml --start 0 --stop 4
```

Expected Result:

```
random_structures/<composition>/
    ranking_scan.csv
    ranking_relax.csv
    ranking_relax_filtered.csv
```