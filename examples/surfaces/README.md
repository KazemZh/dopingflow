# Staged surface screening example

This example takes selected relaxed source structures (vacancy-free parents and, optionally, oxygen-vacancy structures) and performs:

1. low-index slab generation;
2. termination enumeration;
3. representative Sb/co-dopant placement at surface, subsurface, and bulk-like cation layers;
4. fast MLFF screening and relaxation;
5. surface-energy ranking using the periodic source reference evaluated with the **same calculator**;
6. optional refinement of the screening shortlist with a second MLFF.

For the current SnO2/ATO study the default starting facets are (110), (100), (101), and (001). They are configurable and are not hard-coded into the workflow.

## Separate GRACE and MACE environments

The two stages can be run in separate environments:

    conda activate dopingflow-grace
    dopingflow surface-scan -c examples/surfaces/input.toml

    conda activate dopingflow-mace
    dopingflow surface-refine -c examples/surfaces/input.toml

If both calculator dependencies are available in one environment, run both with:

    dopingflow surface -c examples/surfaces/input.toml

## Main outputs

The default output directory is `08_surfaces/`.

- `surface_screen_summary.csv`: every generated slab/termination/depth variant.
- `surface_screen_selected.csv`: top screening candidates per selected source structure.
- `surface_refine_summary.csv`: second-model results for the shortlist.
- `surface_final_selected.csv`: final top-k after refinement.

Only slabs whose composition is proportional to their periodic source structure receive the simple
`(E_slab - n E_bulk)/(2A)` surface-energy ranking. Non-stoichiometric terminations are
kept in the summary with an explicit non-computable status instead of being compared
by raw total energy.

The current co-dopant-depth scan relocates one representative atom of each selected
dopant species while preserving the total composition. It is a controlled screening
of depth preferences, not an exhaustive enumeration of every same-species dopant
permutation.

## Structure selection

Surface screening uses the same target-selection convention as Electronic Conductivity
and Oxidation States:

```toml
[surface]
source_root = "vacancy-selected"
include_vacancy_free = true
include_oxygen_vacancies = false
target_include = []
```

Set `include_oxygen_vacancies = true` to also expose relaxed O-vacancy
structures listed in `vacancies_database.json`. Use `target_include` for exact
IDs or wildcards when only selected parents/vacancy structures should be scanned.
