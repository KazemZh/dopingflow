Oxygen-Vacancy Workflow
=======================

DopingFlow supports two vacancy execution modes built from the same flat
``[vacancies]`` configuration table:

* ``dopingflow vacancies -c input.toml`` runs the complete vacancy workflow in
  one Python environment. It supports symmetry enumeration or Monte Carlo and
  remains useful when all requested ML backends coexist in that environment.
* ``dopingflow vacancies-mc-search -c input.toml`` followed by
  ``dopingflow vacancies-finalize -c input.toml`` splits a Monte Carlo study
  into two environments. This is the recommended path when GRACE is used for
  occupation search and MACE is used for final single points, relaxation,
  reranking, and thermodynamics.

The staged commands deliberately perform separate dependency checks: the search
process imports/builds only the ``mc_*`` calculator, while the finalize process
imports/builds only the ordinary final calculator. GRACE and MACE therefore do
not need to be installed together.

Core scientific workflow
------------------------

For each selected relaxed parent, the vacancy stage can:

#. determine vacancy counts from formal oxidation-state compensation or use an
   explicit ``vacancy_counts`` research-design list;
#. replicate the parent with ``supercell``;
#. search vacancy/cation occupations by symmetry enumeration or Metropolis Monte
   Carlo;
#. retain low-energy structures independently at each fixed vacancy count;
#. relax and rerank selected structures with the final calculator;
#. optionally compare vacancy counts using an oxygen chemical potential and
   finite-temperature oxygen-gas corrections.

Raw total energies for different vacancy counts are not directly comparable,
because the structures contain different numbers of oxygen atoms. Cross-count
thermodynamics requires an oxygen reservoir.

Basic configuration
-------------------

All vacancy settings live in one flat table. The ordinary ``backend``, ``model``,
``task`` and ``device`` fields define the final/reference calculator. Monte Carlo
may optionally define an independent ``mc_backend``, ``mc_model``, ``mc_task``,
``mc_device`` and ``mc_gpu_id`` calculator.

A minimal explicit-count Monte Carlo setup is::

   [vacancies]
   enabled = true
   parent_source = "directory"
   parent_directory = "vacancy-selected"

   host_species = "Sn"
   host_oxidation_state = 4
   vacancy_species = "O"
   vacancy_compensation_charge = 2
   oxidation_state_elements = ["Sb", "Ti"]
   oxidation_state_values = [[3, 5], [4]]

   search_method = "monte-carlo"
   vacancy_counts = [1, 2]
   supercell = [2, 2, 2]

   mc_backend = "grace"
   mc_model = "GRACE-1L-OMAT"
   mc_task = ""
   mc_device = "cuda"
   mc_gpu_id = 0

   backend = "mace"
   model = "mh-1"
   task = "matpes_r2scan"
   device = "cuda"
   gpu_id = 0

Formal charge and explicit vacancy counts
----------------------------------------

Actual dopant counts are read from the parent structure. They are not inferred
from the requested nominal percentages. A dopant in formal state :math:`z_d`
replacing a host in state :math:`z_h` contributes :math:`z_d-z_h`. Contributions
from all co-dopants are combined.

With charge-derived generation, each reachable negative total charge
:math:`\Delta Q` contributes an upper relevant vacancy count

.. math::

   n_{max} = \left\lceil\frac{-\Delta Q}{q_{vac}}\right\rceil,

where ``vacancy_compensation_charge`` is :math:`q_{vac}`. ``extra_vacancies``
and ``max_vacancies_cap`` then bound the generated range.

For a study where vacancy counts are chosen explicitly, use for example::

   vacancy_counts = [1, 2]

The explicit list replaces the charge-derived generation range for the search.
Formal-charge scenarios are still recorded as metadata for interpretation.

Formal oxidation-state assignments define a search-space model; they do not
establish the actual electronic oxidation state of each cation.

Parent discovery
----------------

``parent_source = "selected_candidates"`` uses the normal workflow output root.
``parent_source = "directory"`` processes an existing multi-composition tree::

   parent_source = "directory"
   parent_directory = "vacancy-selected"

Each discovered parent uses the selected scan structure for symmetry mapping and
the corresponding relaxed structure for coordinates/energies. The staged search
and finalize commands can additionally filter and route parents with the keys
below.

Staged parent filtering
~~~~~~~~~~~~~~~~~~~~~~~

``parent_include`` is optional and applies to the staged Monte Carlo commands.
It may contain composition selectors::

   parent_include = [
       "Ti_2.5Sb_2.5",
       "Ce_2.5Sb_5",
       "In_7.5Sb_7.5",
   ]

Common composition labels are canonicalized independently of element order and
``2.5``/``2p5`` notation. Therefore ``Ti_2.5Sb_5``, ``Ti2p5_Sb5`` and
``Sb5_Ti2p5`` match the same composition. A selector containing ``/`` is treated
as an exact parent ID, for example ``Sb5_Ti2p5/candidate_003``. A requested
selector that matches no discovered parent causes a clear error instead of being
silently ignored.

``parent_pick`` controls how many selected parents are retained per composition::

   parent_pick = "lowest_energy"   # or "all"

The filtering stage writes ``selected_candidates.txt`` in ascending relaxed
energy, and parent discovery preserves that order. ``lowest_energy`` therefore
keeps the first/lowest-energy selected candidate for each composition. ``all`` is
the backward-compatible behavior.

Dedicated staged output directory
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

The staged workflow can be kept separate from the source parent tree::

   output_directory = "vacancy-mc-grace-mace"

A relative path is resolved relative to ``input.toml``; an absolute path is also
accepted. Source scan/relaxed structures are still read from ``parent_directory``
or ``structure.outdir``, but result paths are mirrored beneath
``output_directory``. Both the GRACE search and MACE finalize command resolve the
same parent IDs there, so finalization continues directly from the persisted
search selection.

Typical global staged outputs are::

   vacancy-mc-grace-mace/
   ├── vacancy_mc_search_database.csv
   ├── vacancy_mc_search_database.json
   ├── vacancies_database.csv
   ├── vacancies_database.json
   └── <composition>/
       └── <candidate>/
           └── 05_vacancies/

Symmetry enumeration
--------------------

The default ``search_method = "enumeration"`` canonicalizes vacancy occupation
vectors under the parent's symmetry operations.

``enumeration_mode = "exact"`` enumerates all combinations and records exact
orbit degeneracies. ``sample`` samples canonicalized combinations and cannot
claim exact degeneracies. ``auto`` starts from the exact combinatorial estimate
and switches to sampling if configured limits are exceeded.

The explicit configurational partition-function thermodynamics requires exact
enumeration and exact orbit degeneracies. It is not available from the Monte
Carlo search path.

Coupled cation/vacancy Monte Carlo
---------------------------------

Set::

   search_method = "monte-carlo"

The Monte Carlo state is a fixed-site occupation state at a fixed composition
and fixed number of vacancies. Two move classes are available:

``vacancy_swap``
   Exchange the internal vacancy marker with an occupied site on the configured
   vacancy-species sublattice.

``cation_swap``
   Exchange two sites occupied by different cation species. The implementation
   is generic and supports the host plus any number of dopant species present in
   the parent.

Internal vacancy markers are removed before every ML calculator call. To sample
coupled cation/vacancy ordering, keep both move weights non-zero::

   mc_cation_move_weight = 0.5
   mc_vacancy_move_weight = 0.5

Supercell interpretation
~~~~~~~~~~~~~~~~~~~~~~~~

``supercell = [na, nb, nc]`` replicates the search parent before occupation
sampling. For the current SnO2-based 120-atom parent, ``[2, 2, 2]`` gives 960
atoms before vacancies: 320 cations and 640 oxygen sites. A 2.5% dopant represented
by one dopant among 40 cations in the original parent becomes eight dopants among
320 cations, so the concentration remains 2.5%.

One and two vacancies in the 640-site oxygen sublattice correspond to 0.15625%
and 0.3125% of oxygen sites, respectively.

Annealing and stopping
~~~~~~~~~~~~~~~~~~~~~~

With ``mc_annealing = true``, the schedule holds the initial temperature, cools
linearly to the target temperature, then continues at the target temperature::

   mc_annealing = true
   mc_initial_temperature_K = 1500.0
   mc_annealing_hold_steps = 5000
   mc_annealing_steps = 50000
   mc_temperature_K = 600.0

``mc_max_steps`` is the **total** trajectory length. Hold and cooling steps are
part of that total. ``mc_run_mode`` may be ``fixed``, ``converged`` or
``combined``.

For ``converged`` and ``combined`` runs, ``mc_patience`` counts consecutive
steps without a new global best larger than ``mc_improvement_tolerance_eV``.
The current counter is active from step 1, including the hot hold and cooling
ramp. Therefore, if the complete annealing schedule must be reached, choose
``mc_patience`` larger than
``mc_annealing_hold_steps + mc_annealing_steps``.

The library defaults (10,000 maximum steps and 2,000 patience) are intentionally
small enough for development and smoke tests. A large coupled ``2 x 2 x 2``
study should normally set a larger ceiling explicitly and establish convergence
empirically. A practical starting point for the present 960-atom search is::

   mc_run_mode = "combined"
   mc_max_steps = 200000
   mc_patience = 100000

   mc_annealing_hold_steps = 5000
   mc_annealing_steps = 50000

These numbers are not a universal convergence guarantee.

Archive behavior
~~~~~~~~~~~~~~~~

Unique accepted occupation states are archived up to ``sample_max_saved``. If
``mc_energy_window_eV`` is set, states above the configured energy window from
the current archived minimum are discarded. ``sample_seed`` makes the trajectory
reproducible for fixed software/model/runtime behavior.

Each vacancy-count group records ``monte_carlo_summary.json`` with the step count,
stop reason, best energy/step, attempted and accepted move counts, annealing
schedule, and search-calculator provenance.

Staged GRACE -> MACE execution
------------------------------

The recommended split-environment sequence is::

   conda activate dopingflow-grace
   dopingflow vacancies-mc-search -c input.toml --verbose

   conda activate dopingflow-mace
   dopingflow vacancies-finalize -c input.toml --verbose

Stage 1: ``vacancies-mc-search``
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

The search command:

* requires ``search_method = "monte-carlo"``;
* checks/builds only the ``mc_*`` backend;
* discovers, filters and optionally redirects the parent set;
* runs joint cation/vacancy Monte Carlo independently for each requested vacancy
  count;
* writes generated candidate POSCARs and GRACE search metadata;
* ranks the archive by search energy at fixed parent and vacancy count;
* writes ``ranking_scan.csv`` and ``selected_candidates.txt``;
* writes per-parent ``mc_search_results.csv/json`` and the global
  ``vacancy_mc_search_database.csv/json``.

A search fingerprint includes search-space settings, explicit vacancy counts,
MC-calculator settings, and hashes of the source parent structures. It excludes
the final calculator/relaxation settings. A completed compatible search can
therefore be reused when only MACE finalization settings change.

Stage 2: ``vacancies-finalize``
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

The finalize command:

* checks/builds only the ordinary final backend;
* requires a completed compatible Stage-1 search;
* loads the persisted ``selected_candidates.txt`` for each vacancy count;
* evaluates each selected structure with a final-calculator single point before
  relaxation;
* relaxes with the configured final calculator and optimizer;
* reports the same-backend relaxation energy change from the final single point
  to the final relaxed energy;
* reranks the selected structures by final relaxed energy;
* writes the ordinary ``vacancy_results.csv/json`` and global
  ``vacancies_database.csv/json``;
* runs the configured static vacancy thermodynamic analysis with final-calculator
  energies.

Search-energy provenance and final-energy provenance remain separate. A
GRACE-to-MACE difference is not presented as a same-calculator relaxation energy.

Current cross-backend selection limitation
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

At present the staged path is::

   GRACE archive
      -> GRACE ranking
      -> GRACE top-k
      -> MACE single point on GRACE-selected candidates
      -> MACE relaxation and reranking

MACE does **not** yet single-point every archived GRACE candidate before selecting
its relaxation top-k. Consequently, a structure ranked poorly by GRACE can be
excluded even if MACE would rank it favorably. Before a large production study,
validate GRACE-to-MACE ranking agreement on a representative smaller archive.

Reference-state caveat for cation moves
---------------------------------------

When ``mc_cation_move_weight`` is non-zero, defective ``n=1``/``n=2`` minima may
change both the vacancy location and cation ordering relative to the replicated
source parent. The current ``n=0`` parent reference is not subjected to an
independent cation-only Monte Carlo search.

Therefore, a vacancy formation free energy from a coupled cation/vacancy run can
contain both vacancy-formation and cation-reordering contributions relative to
the source parent. This is acceptable when the research question is the coupled
ordering landscape. For a strict vacancy formation free energy referenced to an
equilibrated cation arrangement, a corresponding ``n=0`` cation-only equilibrium
baseline would be required.

Final relaxation and ranking
----------------------------

Within a fixed parent and fixed vacancy count, candidates are ranked by the
appropriate stage energy. The final calculator may be M3GNet, UMA, MACE, or
GRACE. Typical MACE settings are::

   backend = "mace"
   model = "mh-1"
   task = "matpes_r2scan"
   device = "cuda"
   gpu_id = 0

   topk_per_vacancy_count = 20
   optimizer = "bfgs"
   fmax = 0.05
   max_steps = 300
   relax_mode = "atoms"

Static vacancy thermodynamics
-----------------------------

Enable::

   static_thermodynamic_analysis = true

For each cation composition and vacancy count, DopingFlow selects the lowest
converged relaxed final-calculator energy :math:`E_{min}(c,n)`. With oxygen
chemical potential :math:`\mu_O`, the static cross-count term is

.. math::

   \Delta G_{vac}^{static}(n) =
   E_{min}(c,n) - E_{min}(c,0) + n\mu_O.

The direct ``delta_mu_O`` analysis constructs exact pairwise crossings of the
linear oxygen-grand-potential branches rather than inferring transitions from a
coarse grid.

Finite-temperature oxygen gas
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

For calibrated/reference oxygen modes with NIST Shomate thermodynamics,

.. math::

   \mu_O(T,p) = \mu_O^{0}
   + \frac{1}{2}[H_{O_2}(T)-H_{O_2}(298)-T S_{O_2}(T)]
   + \frac{1}{2}k_BT\ln(p/p^\circ).

``oxygen_standard_state_mode = "nist_shomate"`` supplies the continuous O2
enthalpy/entropy contribution over its supported temperature range. The pressure
term is ideal-gas oxygen.

Solid configurational treatment
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

``solid_configurational_entropy = "none"``
   Static-lattice solid contribution only.

``solid_configurational_entropy = "ideal"``
   Adds ideal occupied/vacant mixing on the oxygen sublattice.

``solid_configurational_entropy = "configurational"``
   Evaluates a canonical partition function over exact symmetry-distinct vacancy
   configurations and exact orbit degeneracies. This requires
   ``search_method = "enumeration"`` and ``enumeration_mode = "exact"`` and is
   incompatible with Monte Carlo sampling.

For the exact partition-function mode,

.. math::

   \Delta F_{config}(n,T) =
   -k_B T \ln\left[\sum_i g_i
   \exp\left(-\frac{E_i-E_{min}}{k_B T}\right)\right].

Finite-temperature vacancy formation free energies are written to
``vacancy_formation_free_energy.csv/json``. Solid vibrational, zero-point,
magnetic, thermal-electronic, anharmonic, thermal-expansion, and solid-pV terms
remain outside this screening-level treatment.

Oxygen reference choices
------------------------

The vacancy analysis supports raw/reference modes such as ``reference_file`` and
``same_calculator`` and experimental calibration modes ``global`` and
``chemistry-specific``.

``global`` fits one backend/model/task-specific effective oxygen reference from
eligible calculated binary oxides plus experimental 298 K formation enthalpies.
``chemistry-specific`` repeats the fit for the actual host/dopant chemistry of
each vacancy composition. Included/excluded references, residuals, spread, and
provenance are written to ``oxygen_calibration_report.json``.

See :doc:`oxygen_calibration` for the calibration equations and data requirements.

Fitted energy correction compatibility
--------------------------------------

When a compatible fitted backend-specific energy-correction model has already
been produced, vacancy thermodynamics can opt in with::

   apply_fitted_energy_correction = true
   allow_legacy_energy_correction_provenance = false

The solid vacancy reaction correction is applied as

.. math::

   \Delta E_{vac}^{corr} =
   [E_{def}^{raw} - E_{parent}^{raw}]
   + [C_{def} - C_{parent}].

Use a raw same-backend oxygen reservoir such as ``reference_file`` or
``same_calculator`` with this path. Do not combine a fitted experimental
formation-energy correction with ``oxygen_reference_mode = "global"`` or
``"chemistry-specific"`` because both paths use experimental formation-enthalpy
information for oxygen-related calibration.

See :doc:`vacancy_energy_correction` for provenance checks and uncertainty
propagation.

Recommended production workflow
-------------------------------

For the current large-supercell GRACE/MACE study, a practical sequence is:

#. Configure ``parent_include`` and ``parent_pick = "lowest_energy"``.
#. Use ``output_directory`` to isolate the vacancy study from source parents.
#. Set ``vacancy_counts = [1, 2]`` and ``supercell = [2, 2, 2]``.
#. First smoke-test one composition with about 20 MC steps and top-k 3.
#. In the GRACE environment run ``vacancies-mc-search``.
#. Inspect the search archive, acceptance statistics, and candidate count.
#. In the MACE environment run ``vacancies-finalize``.
#. Confirm end-to-end provenance and output handoff.
#. Restore the production MC ceiling and process the full requested parent set.
#. Validate search convergence and GRACE/MACE ranking agreement before drawing
   conclusions from the full campaign.

The Streamlit GUI includes a dedicated **Staged Vacancy MC** page that edits these
settings, shows the two ``conda run`` commands, and can launch each stage in its
own named conda environment.
