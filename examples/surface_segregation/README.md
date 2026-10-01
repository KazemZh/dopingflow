# Surface segregation Monte Carlo example

This stage is run **after natural surface screening/refinement** and only on the
exact surface terminations selected by the user.

The MC preserves composition and swaps a dopant with a host cation:

```text
X@site_i + Sn@site_j  ->  Sn@site_i + X@site_j
```

Coordinates and the slab lattice remain fixed during the Markov chain. A user-selected
ML force field provides the single-point energy for Metropolis acceptance.

In the GUI, the model controls follow the selected backend automatically:

- M3GNet → default model, no task/head;
- GRACE → GRACE model dropdown, no task/head;
- UMA → UMA model dropdown + UMA task;
- MACE → MACE model dropdown (or custom checkpoint), with a head only for
  multi-head/custom checkpoints. `mh-1` defaults to `matpes_r2scan`.

Recommended first validation run:

```toml
temperature_K = 800.0
steps = 5000
burn_in = 1000
sample_interval = 20
```

After checking acceptance and convergence, increase to a production trajectory such as
`100000` steps or more and repeat with more than one random seed.

Main outputs for each selected surface:

- `site_occupancy.csv`: probability that each dopant occupies each cation site;
- `zone_occupancy.csv`: normalized surface/subsurface/bulk populations and effective ΔGseg;
- `mc_trace.csv`: energy and acceptance convergence;
- `zone_trace.csv`: sampled depth-zone populations versus MC step;
- `swap_statistics.csv`: attempted/accepted moves by dopant;
- `POSCAR_best_mc` and `POSCAR_final_mc`: representative sampled occupations.

The reported effective segregation free energy is

```text
ΔGseg,eff(zone vs bulk) = -kB T ln(rho_zone / rho_bulk)
```

where `rho_zone` is the mean dopant occupancy per available cation site in that depth zone.
This corrects for the fact that the bulk region usually contains more sites than the surface.

Negative values mean enrichment relative to bulk at the sampled temperature. This is an
occupancy-derived effective PMF/free-energy preference, not the old static two-structure
segregation energy.
