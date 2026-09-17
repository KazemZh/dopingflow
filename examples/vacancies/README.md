# Oxygen vacancies: staged GRACE → MACE example

This example shows the production-oriented two-environment Monte Carlo vacancy
workflow used for large coupled cation/vacancy occupation searches.

The workflow is deliberately split into two commands so **GRACE and MACE do not
need to be installed in the same Python environment**:

```bash
# Stage 1: GRACE environment
conda activate dopingflow-grace
dopingflow vacancies-mc-search -c examples/vacancies/input.toml --verbose

# Stage 2: MACE environment
conda activate dopingflow-mace
dopingflow vacancies-finalize -c examples/vacancies/input.toml --verbose
```

Both commands read the same `[vacancies]` section. When `output_directory` is
set, GRACE writes its archive and selected structures into that dedicated tree,
and the MACE finalize stage resolves the same mirrored parent IDs there. The
source parent structures are not modified.

## Parent selection

The checked-in example uses an existing multi-composition tree:

```toml
parent_source = "directory"
parent_directory = "vacancy-selected"
parent_pick = "lowest_energy"
output_directory = "vacancy-mc-grace-mace"
```

`parent_pick = "lowest_energy"` keeps the first selected candidate for each
composition. The normal DopingFlow filtering stage writes
`selected_candidates.txt` in ascending relaxed-energy order, so this is the
lowest-energy filtered parent for that composition. Use `parent_pick = "all"`
to process every selected parent.

`parent_include` can limit the study to named compositions. Common labels are
matched independently of element order and the usual decimal/`p` notation, so
for example these refer to the same composition:

```text
Ti_2.5Sb_5
Ti2p5_Sb5
Sb5_Ti2p5
```

An exact selector such as `Sb5_Ti2p5/candidate_003` chooses one specific parent.
Remove `parent_include` completely to process every discovered composition.

## Search space

The example studies exactly one and two oxygen vacancies:

```toml
vacancy_counts = [1, 2]
supercell = [2, 2, 2]
```

For the SnO2-based 120-atom parent used in the current project, a `2×2×2`
replication contains 960 atoms before vacancies: 320 cations and 640 oxygen
sites. Dopant numbers are multiplied by eight, while their percentages remain
unchanged. One and two oxygen vacancies correspond to 1/640 and 2/640 of the
oxygen sublattice, respectively.

The Monte Carlo state contains both the cation occupations and the oxygen-vacancy
occupation. `cation_swap` exchanges two different cation species; `vacancy_swap`
moves a vacancy marker between oxygen sites. Keep both move weights non-zero for
coupled dopant-vacancy ordering.

## Stage 1: GRACE Monte Carlo search

The example uses:

```toml
search_method = "monte-carlo"
mc_backend = "grace"
mc_model = "GRACE-1L-OMAT"
mc_device = "cuda"

mc_annealing = true
mc_initial_temperature_K = 1500.0
mc_annealing_hold_steps = 5000
mc_annealing_steps = 50000
mc_temperature_K = 600.0
mc_run_mode = "combined"
mc_max_steps = 200000
mc_patience = 100000
mc_energy_window_eV = 1.0
mc_cation_move_weight = 0.5
mc_vacancy_move_weight = 0.5
sample_max_saved = 100
```

`mc_max_steps` is the **total** trajectory length. The hot hold and cooling ramp
are part of that total. In the current implementation the no-improvement
`mc_patience` counter is active from step 1, so if the full annealing schedule is
required, set patience larger than `mc_annealing_hold_steps + mc_annealing_steps`.

The library-level default of 10,000 MC steps is intentionally conservative and
is suitable for smoke tests. The explicit 200,000-step ceiling in this example
is a production-oriented starting point for the much larger 960-atom coupled
occupation space; convergence should still be validated for the chemistry being
studied.

For every parent and vacancy count, Stage 1 writes a low-energy unique archive,
`ranking_scan.csv`, `monte_carlo_summary.json`, and `selected_candidates.txt`.
`sample_seed` controls trajectory reproducibility. `sample_max_saved` caps the
archive retained inside the configured GRACE energy window.

## Stage 2: MACE finalization

The final calculator is configured independently:

```toml
backend = "mace"
model = "mh-1"
task = "matpes_r2scan"
device = "cuda"
topk_per_vacancy_count = 20
```

The current staged selection path is:

```text
GRACE Monte Carlo archive
        ↓
GRACE ranking at fixed vacancy count
        ↓
GRACE top-k selected_candidates.txt
        ↓
MACE single point on selected candidates
        ↓
MACE relaxation
        ↓
MACE relaxed-energy reranking
        ↓
MACE vacancy thermodynamics
```

This distinction matters: **MACE does not yet rescore every archived GRACE
candidate before top-k selection.** Before a very large production campaign,
validate that GRACE and MACE rankings agree adequately on a smaller archive.
Search and final energies are stored with separate provenance; a GRACE-to-MACE
energy difference is not labeled as a same-calculator relaxation energy.

## Important reference-state caveat

With non-zero cation move weight, the defective `n=1` and `n=2` searches can
change cation ordering relative to the replicated source parent. The current
`n=0` reference is not subjected to a separate cation-only Monte Carlo search.
Therefore a reported vacancy formation free energy can contain both vacancy
formation and cation-reordering contributions. This is appropriate for the
coupled ordering study, but a rigorous vacancy formation energy referenced to an
equilibrated cation arrangement would require a corresponding `n=0` cation-only
baseline.

## Thermodynamics

The example enables static vacancy thermodynamics and ideal vacancy mixing:

```toml
static_thermodynamic_analysis = true
solid_configurational_entropy = "ideal"
oxygen_reference_mode = "reference_file"
```

Monte Carlo supports `solid_configurational_entropy = "none"` or `"ideal"`.
The explicit partition-function mode `"configurational"` requires exact symmetry
enumeration because Monte Carlo does not provide a complete set of exact orbit
degeneracies.

Raw total energies for different vacancy counts should not be compared directly.
They contain different numbers of oxygen atoms. Cross-count thermodynamics uses
an oxygen chemical potential, schematically

```text
ΔG_vac(n,T,pO2) = E_min(n) - E_min(0) + n μO(T,pO2) + ΔF_config
```

If a fitted backend-specific energy correction is later enabled for vacancy
thermodynamics, fit it first and keep the oxygen reservoir on a raw same-backend
mode such as `reference_file` or `same_calculator`. Do not combine that fitted
correction with the experimental `global` or `chemistry-specific` oxygen
calibration modes.

## Recommended first smoke test

Before launching all listed compositions, temporarily reduce the input to one
parent and a very small trajectory, for example:

```toml
parent_include = ["Ti_2.5Sb_2.5"]
mc_max_steps = 20
mc_patience = 20
sample_max_saved = 10
topk_per_vacancy_count = 3
```

Run Stage 1, inspect the generated search tree, then run Stage 2. After the
end-to-end handoff is confirmed on the target machine, restore the production
settings.
