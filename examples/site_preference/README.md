# Dopant site-preference / ordering analysis

This optional stage analyses the relaxed structures already produced by dopingflow and
answers questions such as whether Sb–Ti prefers a first-neighbour shell, whether Sb–Sb
clustering is avoided, and which dopants associate with oxygen vacancies.

Start with analysis only:

~~~toml
[site_preference]
enabled = true
host_species = "Sn"
anion_species = ["O"]
include_vacancy_free = true
include_oxygen_vacancies = true
max_shells = 6
shell_tolerance_angstrom = 0.12
mapping_tolerance_angstrom = 1.5
~~~

Optional controlled pair-shell scan:

~~~toml
[site_preference.pair_scan]
enabled = true
execute = false
pairs = ["Sb-Ti", "Sb-Nb", "Ti-Nb"]
max_shells = 6
backend = "mace"
model = "small"
device = "cpu"
relax = true
fmax = 0.05
max_steps = 300
~~~

Set execute=true only when the selected MLFF environment is active. With execute=false
the shell structures are generated but no MLFF calculation is launched.

Optional finite-temperature cation-ordering Monte Carlo:

~~~toml
[site_preference.ordering_mc]
enabled = true
execute = false
temperature_K = 800
steps = 10000
burn_in = 2000
sample_interval = 20
max_targets = 5
backend = "mace"
model = "small"
device = "cpu"
relax_best = true
~~~

Run with:

~~~bash
dopingflow site-preference -c input.toml
~~~

Important interpretation details:

- Warren–Cowley alpha < 0 means association relative to random occupancy.
- Warren–Cowley alpha > 0 means avoidance.
- For three or more dopants, the stage also reports explicit triplet motifs: compact triangle,
  connected chain, isolated pair plus a third dopant, or dispersed. The neighbor-shell cutoff
  used to define motif connectivity is `motif_neighbor_shell_max`.
- Energy trends extracted from the existing structure ensemble are configuration-energy
  correlations, not isolated pair-binding energies.
- The controlled pair scan is the cleaner energetic test because all other cation sites are
  returned to the host species.
- Ordering MC preserves the overall composition and uses cation identity swaps. Forces are
  not evaluated during MC; only the best occupation is optionally relaxed afterwards.


## GUI result browser

The Streamlit **Dopant Site Preference** page presents results one structure at a time rather
than combining every composition in one plot.

1. Select the composition.
2. Select a vacancy-free or oxygen-vacancy structure.
3. Read the **At a glance** energy/status row.
4. Open only the relevant analysis tab:
   - **Dopant pairs**: nearest shell and distance for each pair.
   - **Local ordering**: Warren–Cowley alpha translated into association, random-like mixing,
     or avoidance for one selected pair at a time.
   - **O-vacancy relation**: nearest dopant–V_O shell and distance for vacancy structures.
   - **Three-dopant motifs**: compact/chain/pair+third/dispersed arrangement for one triplet.
   - **Compare same composition**: distance versus relative configuration energy using only
     directly comparable structures.
5. The **Controlled pair scan** is displayed one dopant pair at a time and highlights the
   lowest-energy shell/orbit relative to the farthest tested separation.
6. The **Finite-temperature ordering MC** is displayed one target at a time.

Full CSV tables remain available under **Raw / global result tables** for auditing and export,
but they are intentionally not the primary visualization.
