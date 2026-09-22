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


## GUI results explorer

The Streamlit **Dopant Site Preference** page is organized for scientific interpretation,
not for browsing raw CSV files.

The results area has four views:

- **Structure result** — select one composition and one relaxed structure. The page first
  gives a short scientific summary: energetic position, important dopant geometry, strongest
  local-order signal, vacancy environment when present, and three-dopant motif only when
  relevant. Pair geometry, local ordering, vacancy environment, and energetic context are
  then shown in separate compact sections.
- **Controlled pair scan** — select one dopant pair. The page states whether a closer
  arrangement or the largest tested separation is favored, reports the preferred shell,
  distance and ΔE, and then shows the supporting energy-versus-distance plot.
- **Ordering MC** — select one MC target and one dopant pair. The page translates
  finite-temperature Warren–Cowley α into association, avoidance, or random-like mixing
  and shows the shell-resolved values.
- **Raw data** — keeps the complete CSV tables available for auditing/export without
  cluttering the interpretation view.

For structure-level plots, only the selected structure is shown. Energetic rankings include
only structures with the same composition, structure type, and oxygen-vacancy count.
