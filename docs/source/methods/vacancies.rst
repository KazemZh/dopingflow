Oxygen-Vacancy Workflow
=======================

``dopingflow vacancies -c input.toml`` is the only public vacancy command. It
discovers filtered relaxed parents, determines charge-based vacancy counts,
enumerates symmetry-distinct arrangements, evaluates single-point ML energies,
selects the lowest-energy structures independently at each vacancy count,
relaxes that top-k set, reranks it, and writes parent-level and global summaries.

Configuration
-------------

All settings live in one flat ``[vacancies]`` table. Screening and relaxation
share one resolved ``backend``, ``model``, ``task`` and ``device`` combination.
The backend may be M3GNet, UMA, MACE, or GRACE.

Only chemistry-specific inputs are required: ``host_species`` and the parallel
``oxidation_state_elements``/``oxidation_state_values`` arrays. In addition,
``parent_directory`` is required only when ``parent_source = "directory"``.
All other keys have runtime defaults. In particular, the calculator defaults to
``backend = "m3gnet"``, ``model = "default"``, ``task = ""``, and
``device = "cpu"``. Backend-specific blank model/task values are normalized as
documented in :doc:`../input_file`; for example, MACE ``mh-1`` with a blank task
uses ``omat_pbe``. The complete example below deliberately selects MACE/CUDA
and therefore is not a listing of default values.

.. code-block:: toml

   [vacancies]
   enabled = true
   parent_source = "selected_candidates"
   include_parent_reference = true
   skip_if_done = true
   resume = true

   count_mode = "all_reachable"
   host_species = "Sn"
   host_oxidation_state = 4
   vacancy_species = "O"
   vacancy_compensation_charge = 2
   oxidation_state_elements = ["Sb", "Nb"]
   oxidation_state_values = [[3, 5], [5]]

   extra_vacancies = 0
   max_vacancies_cap = 8
   symprec = 1.0e-3
   angle_tolerance = 5.0
   mapping_tolerance = 1.0
   enumeration_mode = "auto"
   max_exact_raw_configs = 300000
   max_exact_unique_configs = 100000
   sample_budget = 20000
   sample_batch_size = 256
   sample_patience = 4000
   sample_seed = 42
   sample_max_saved = 50000
   minimum_vacancy_distance = 0.0

   backend = "mace"
   model = "medium-mpa-0"
   task = ""
   device = "cuda"
   gpu_id = 0
   n_workers = 1
   tf_threads = 1
   omp_threads = 1
   chunksize = 25

   topk_per_vacancy_count = 15
   energy_normalization = "per_vacancy"
   optimizer = "bfgs"
   fmax = 0.05
   max_steps = 300
   relax_mode = "atoms"
   cell_filter = "frechet"

Formal charge and vacancy range
-------------------------------

Actual dopant counts are read from each selected relaxed parent, not inferred
from requested percentages. A dopant in formal state :math:`z_d` replacing a
host in state :math:`z_h` contributes :math:`z_d-z_h`. Contributions from all
co-dopants are combined before a vacancy count is calculated. Mixed states are
represented as population counts, avoiding permutations over individual atoms.

For every reachable negative total charge :math:`\Delta Q`, the upper relevant
count is ``ceil(-delta_Q / vacancy_compensation_charge)``. The workflow explores
the continuous range from one through the largest upper count, then applies
``extra_vacancies``, ``max_vacancies_cap``, and the number of available oxygen
sites. Residual charge is retained for every scenario and count; exact
compensation is marked only when it is zero.

Eight Sb3+ atoms on Sn4+ sites give ``8 × (3 - 4) = -8`` and therefore generate
counts 1, 2, 3, and 4. Four Sb3+ plus four Nb5+ give ``-4 + 4 = 0`` and no
defective count unless ``extra_vacancies`` is positive. For ``delta_Q = -3``,
one vacancy has residual charge -1 and two vacancies have residual charge +1;
both occur because the generated range ends at ``ceil(3/2) = 2``.

These are formal assumptions defining the search space. They do not establish
actual oxidation states. Later electronic-structure work may require Bader
charges, projected densities of states, magnetic moments, local coordination,
and charge-density differences.

Parents, symmetry, and enumeration
----------------------------------

The default source is each composition's ``selected_candidates.txt``. The
corresponding ``01_scan/POSCAR`` supplies symmetry and the selected
``02_relax/POSCAR`` supplies coordinates. Sites are mapped by species and
nearest periodic position; unreliable mappings fail explicitly.

To process an existing collection containing many composition subdirectories,
select a directory source. The directory is resolved relative to ``input.toml``
unless it is absolute. Every immediate composition subdirectory may contain its
own ``selected_candidates.txt``::

   parent_source = "directory"
   parent_directory = "Mn-Nb-Ta-Ce-Ru-In-Sn-Sb"

The global ``vacancies_database.csv/json`` files are then written into that
parent directory. Every dopant found in any processed parent, including Ce in
this example, must have an entry in the flat oxidation-state arrays.

Vacancy occupancy vectors are canonicalized under the parent's symmetry
permutations. ``exact`` enumerates all combinations and records exact orbit
degeneracy. ``sample`` uses ``sample_seed``, deduplicates canonical keys, and
sets degeneracy to null because it is not exact. ``auto`` selects exact mode
below ``max_exact_raw_configs`` and restarts in sampled mode if the exact unique
limit is exceeded. ``minimum_vacancy_distance`` filters close periodic pairs.

Screening, relaxation, and interpretation
------------------------------------------

The parent and every defective structure use the same common calculator
factory. Single-point energies and relaxed energies are ranked only among
structures having the same parent and vacancy count. The parent does not consume
a top-k slot. Its prior relaxed energy is reused only when metadata prove
calculator and relaxation compatibility; otherwise a consistency relaxation is
written below the vacancy output without modifying the original parent.

Raw totals across different vacancy counts are not defect formation energies,
because the structures contain different numbers of oxygen atoms. Thermodynamic
comparison requires an oxygen chemical potential, for example
``E_defect - E_parent + n * mu_O``.

Level-1 static-lattice thermodynamic analysis
---------------------------------------------

Set ``static_thermodynamic_analysis = true`` to add the sixth analysis phase. The
backward-compatible default is ``false``; generation, screening, and relaxation
remain unchanged when it is disabled. Actual dopant counts are read from parent
structures. Parents having the same rounded directory name are never combined
unless their integer species counts and cation-site totals are identical.

**Static-lattice approximation:** solid free energies are approximated by 0 K
relaxed ML energies. Temperature and pressure enter only through the oxygen-gas
chemical potential. Vibrational, zero-point, configurational, thermal electronic,
magnetic, anharmonic, thermal-expansion, and solid-pV contributions are neglected.
This is not a complete finite-temperature phase diagram.

For actual composition :math:`c`, the analysis selects the lowest converged
relaxed energy :math:`E_{min}(c,n)` across every selected parent and relaxed
configuration at fixed vacancy count :math:`n`. Unconverged or missing relaxed
energies are excluded by default; there is no silent single-point fallback.
The cross-count intercept and oxygen grand potential are

.. math::

   A(c,n) = E_{min}(c,n) - E_{min}(c,0) + n\mu_O^{ref}

.. math::

   \Delta\Omega(c,n,\Delta\mu_O) = A(c,n) + n\Delta\mu_O

and the preferred count is

.. math::

   n_{best}(c,\Delta\mu_O) = \operatorname*{argmin}_n \Delta\Omega(c,n,\Delta\mu_O).

Thus there is no universal best vacancy count without a stated
``delta_mu_O``. Exact pairwise line crossings—not a coarse grid—define the
stability intervals, and ties are retained explicitly.

For the optional ideal-gas reservoir map,

.. math::

   \Delta\mu_O(T,p) = \Delta\mu_O^{standard}(T)
   + \frac{1}{2}k_BT\ln\left(\frac{p_{O_2}}{p^{standard}}\right).

``oxygen_standard_state_mode = "nist_shomate"`` uses the published piecewise
NIST Chemistry WebBook O2 Shomate equations (Chase 1998) for continuous
``H(T)-H(298.15)`` and ``S(T)`` values between 100 and 6000 K at 1 bar. Thus the
printed discrete JANAF temperatures do not limit the requested grid;
extrapolation outside the coefficient ranges fails. This convention does not add
an explicit O2 zero-point energy. ``user_table`` linearly interpolates a supplied
per-O correction. With ``none``, the correction is zero and pressure results are
marked qualitative relative to the standard pressure at the same temperature.

``oxygen_reference_mode = "reference_file"`` reads O2 and the per-O
``muO_shift_ev`` from ``reference_energies.json`` and verifies its
backend/model/task metadata. Incompatible references fail. Metadata-free
references fail unless ``allow_unverified_oxygen_reference = true`` is set
deliberately, in which case outputs remain marked unverified.
``same_calculator`` evaluates the configured O2 structure with the vacancy
calculator; optional molecular relaxation uses atomic coordinates only, never a
cell filter. Solid-trained foundation models may nevertheless describe isolated
O2 poorly. ``explicit`` accepts a user-supplied per-O value. ``none`` writes
composition minima but no stability intervals or best-count claims.

The Results Explorer keeps the stability map, grand-potential envelope,
grand potential versus vacancy count, and preferred count versus doping in their
original ``delta_mu_O``-based form. Their energies and axes are independent of
the gas standard-state conversion. Only the complete T-pO2 map offers ``Include
delta_mu_O_standard(T)`` and ``Omit delta_mu_O_standard(T)`` views. The omitted
view uses only the ideal-gas pressure term and is explicitly labeled approximate.

The explicitly named static-lattice outputs are:

- ``vacancy_static_minima.csv/json``: one minimum per exact integer
  composition and vacancy count.
- ``vacancy_static_stability_intervals.csv/json``: exact lower-envelope windows.
- ``vacancy_static_best_counts.csv/json``: preferred counts and ties at requested
  oxygen chemical potentials.
- ``vacancy_static_pressure_map.csv/json``: T-pO2 preferred counts, including
  the oxygen standard-state mode, source, and approximation status used by plots.
- ``vacancy_static_analysis_metadata.json``: reference verification, exclusions,
  missing counts, failed compositions, checksum, and resolved settings.

The earlier compact filenames are also written as compatibility aliases.

These results establish relative stability only within the generated doped-host
structure family. They do not prove stability against decomposition into all
competing phases; that requires a later oxygen-grand-potential convex hull. Even
with a gas standard-state correction, missing solid free-energy terms remain.

Outputs and resume
------------------

Each parent receives ``05_vacancies/`` containing ``parent_reference/``,
``vacancy_counts.csv/json``, ``vacancy_results.csv/json``, and one
``V_O_NN/`` group per count. Each configuration retains ``00_generate``,
``01_scan``, and ``02_relax`` provenance. The structure output root also receives
``vacancies_database.csv`` and ``vacancies_database.json``. These remain
separate from normal formation and phase-diagram databases and retain every
configuration for provenance. The compact files above are the plotting sources.

Configuration and source checksums form a stable fingerprint. Compatible
complete parents are skipped, compatible internal scan/relax metadata are
resumed, and incompatible results are recomputed rather than silently reused.

Run directly or append the one run-all step:

.. code-block:: bash

   dopingflow vacancies -c input.toml
   dopingflow run-all -c input.toml --until vacancies
   dopingflow run-all -c input.toml --only vacancies
