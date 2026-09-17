Staged Vacancy Monte Carlo Configuration
========================================

This page documents the extra ``[vacancies]`` controls used by the two-stage
Monte Carlo workflow:

::

   dopingflow vacancies-mc-search -c input.toml
   dopingflow vacancies-finalize -c input.toml

The staged path is intended for cases where the fast search backend and final
backend live in incompatible Python environments, for example GRACE for the
occupation search and MACE for final single points, relaxation, reranking, and
thermodynamics.

The ordinary one-process command remains available::

   dopingflow vacancies -c input.toml

Use it for symmetry enumeration or for Monte Carlo when all requested backends
coexist in one environment.

Parent source and selection
---------------------------

``parent_source``
~~~~~~~~~~~~~~~~~

``"selected_candidates"`` uses the normal structure output tree.
``"directory"`` uses an explicit multi-composition tree supplied by
``parent_directory``.

Example::

   parent_source = "directory"
   parent_directory = "/home/user/project/vacancy-selected"

``parent_include``
~~~~~~~~~~~~~~~~~~

Optional string array limiting staged search/finalization to selected
compositions or exact parents::

   parent_include = [
       "Ti_2.5Sb_2.5",
       "Ti_2.5Sb_5",
       "Ce_2.5Sb_2.5",
   ]

For common composition directory labels, element order and decimal/``p``
notation are canonicalized. Thus ``Ti_2.5Sb_5``, ``Ti2p5_Sb5`` and
``Sb5_Ti2p5`` match the same composition.

A selector containing ``/`` is treated as an exact parent ID, for example::

   parent_include = ["Sb5_Ti2p5/candidate_003"]

A requested selector that matches no discovered parent is an error.

``parent_pick``
~~~~~~~~~~~~~~~

Controls how many selected parents are kept per composition:

``"all"``
   Keep every discovered selected parent. This is the backward-compatible
   behavior.

``"lowest_energy"``
   Keep the first selected parent for each composition. The normal DopingFlow
   filtering stage writes ``selected_candidates.txt`` in ascending relaxed
   energy, so the first selected parent is the lowest-energy filtered candidate.

Example::

   parent_pick = "lowest_energy"

``output_directory``
~~~~~~~~~~~~~~~~~~~~

Optional dedicated output root for the complete staged study::

   output_directory = "vacancy-mc-grace-mace"

Relative paths are resolved from the directory containing ``input.toml``.
Source parent structures remain in ``parent_directory`` or
``structure.outdir``. Search and finalization mirror the selected
``composition/candidate`` IDs below ``output_directory`` and therefore share a
persistent on-disk handoff.

Explicit vacancy counts and supercell
-------------------------------------

``vacancy_counts``
~~~~~~~~~~~~~~~~~~

Optional explicit positive integer list. When present, it defines the vacancy
counts searched rather than deriving the generation range from formal charge.
Formal-charge scenarios are still retained as metadata.

For the present study::

   vacancy_counts = [1, 2, 3, 4]

When using the explicit 1--4-vacancy research design, keep the cap consistent::

   max_vacancies_cap = 4

``supercell``
~~~~~~~~~~~~~

Three positive integers applied before the occupation search::

   supercell = [2, 2, 2]

For a 120-atom SnO2-based source parent with 40 cations and 80 oxygen sites, a
2x2x2 replication contains 960 atoms before vacancies: 320 cations and 640
oxygen sites. Dopant counts are multiplied by eight while their percentages are
unchanged.

One, two, three, and four vacancies correspond to 0.15625%, 0.3125%, 0.46875%,
and 0.625% of the 640-site oxygen sublattice, respectively.

Stage-1 Monte Carlo calculator
------------------------------

The search calculator is independent of the final calculator::

   mc_backend = "grace"
   mc_model = "GRACE-1L-OMAT"
   mc_task = ""
   mc_device = "cuda"
   mc_gpu_id = 0

``vacancies-mc-search`` checks and builds only this calculator. The ordinary
``backend`` calculator is not instantiated in the search environment.

Coupled cation/vacancy moves
----------------------------

The Monte Carlo state contains both the vacancy occupation and cation species
occupation at fixed composition and fixed vacancy count.

``mc_vacancy_move_weight`` controls moves that exchange a vacancy marker with an
occupied vacancy-species site. ``mc_cation_move_weight`` controls swaps between
two sites occupied by different cation species.

For coupled dopant-vacancy ordering::

   mc_cation_move_weight = 0.5
   mc_vacancy_move_weight = 0.5

At least one move weight must be positive. Internal vacancy markers are removed
before ML energy evaluation.

Annealing and stopping controls
-------------------------------

A production-oriented starting point for the present 960-atom coupled search is::

   mc_annealing = true
   mc_initial_temperature_K = 1500.0
   mc_annealing_hold_steps = 5000
   mc_annealing_steps = 50000
   mc_temperature_K = 600.0

   mc_run_mode = "combined"
   mc_max_steps = 500000
   mc_patience = 105000
   mc_improvement_tolerance_eV = 0.001

   mc_energy_window_eV = 1.0
   sample_seed = 42
   sample_max_saved = 100

``mc_max_steps`` is the total trajectory length. The hot hold and cooling ramp
are included in this total.

``mc_run_mode`` accepts:

``"fixed"``
   Stop only when ``mc_max_steps`` is reached.

``"converged"``
   Stop after ``mc_patience`` consecutive steps without a new global-best
   improvement larger than ``mc_improvement_tolerance_eV``.

``"combined"``
   Apply both the maximum-step ceiling and no-improvement stopping condition.

The current no-improvement counter is active from step 1, including the high-
temperature hold and cooling ramp. For the present schedule, the annealing phase
lasts 5,000 + 50,000 = 55,000 trial moves. Setting ``mc_patience = 105000``
therefore prevents a no-improvement stop before the complete annealing schedule
plus roughly 50,000 additional trial moves at 600 K. Any new best structure that
improves the previous global minimum by at least 1 meV resets the patience
counter.

The package-level 10,000-step/2,000-patience defaults are intentionally small
and are useful for development/smoke tests. The larger 500,000-step ceiling and
105,000-step patience above are project-specific production starting values, not
a universal convergence guarantee. Independent replicas/seeds and low-energy
motif agreement provide stronger convergence evidence than a single trajectory.

Archive and selection
---------------------

``sample_max_saved`` caps the retained unique accepted occupation archive.
``mc_energy_window_eV`` limits the archive to structures sufficiently close to
the current archived minimum.

After Stage 1, each fixed vacancy-count group is sorted by the MC-search energy.
``topk_per_vacancy_count`` determines how many structures are written to
``selected_candidates.txt`` for finalization. With ``vacancy_counts = [1, 2, 3,
4]``, this selection is performed independently for all four vacancy counts.

Current staged selection sequence::

   GRACE archive
      -> GRACE ranking
      -> GRACE top-k
      -> MACE single point on selected candidates
      -> MACE relaxation
      -> MACE relaxed-energy reranking

MACE does not currently rescore the complete GRACE archive before the top-k is
chosen. A GRACE-low/MACE-high ranking disagreement can therefore affect which
structures reach final relaxation. Validate cross-backend ranking agreement on a
smaller archive before a large production campaign.

Stage-2 final calculator
------------------------

Example::

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
   cell_filter = "frechet"

``vacancies-finalize`` checks/builds only this final calculator. It reads the
persisted Stage-1 selection, evaluates a final-backend single point before
relaxation, relaxes, reranks by final relaxed energy, and then runs the configured
vacancy thermodynamic analysis.

The pre-relaxation final-backend single point is stored separately from the GRACE
search energy. A GRACE-to-MACE energy difference is never reported as a
same-calculator relaxation energy change.

Two-environment execution
-------------------------

A typical environment setup is::

   conda create -n dopingflow-grace python=3.11 pip -y
   conda activate dopingflow-grace
   pip install -e ".[grace]"

   conda create -n dopingflow-mace python=3.11 pip -y
   conda activate dopingflow-mace
   pip install -e ".[mace]"

Run Stage 1::

   conda activate dopingflow-grace
   dopingflow vacancies-mc-search -c input.toml --verbose

Then Stage 2::

   conda activate dopingflow-mace
   dopingflow vacancies-finalize -c input.toml --verbose

Both stages must use the same ``input.toml`` and compatible search settings. A
search fingerprint prevents finalization from silently consuming a search made
with a different search space, MC calculator, or source parent structure.

Output layout
-------------

With ``output_directory = "vacancy-mc-grace-mace"`` the main files are::

   vacancy-mc-grace-mace/
   ├── vacancy_mc_search_database.csv
   ├── vacancy_mc_search_database.json
   ├── vacancies_database.csv
   ├── vacancies_database.json
   └── <composition>/
       └── <candidate>/
           └── 05_vacancies/
               ├── mc_search_meta.json
               ├── mc_search_results.csv
               ├── mc_search_results.json
               ├── parent_reference/
               ├── V_O_01/
               │   ├── monte_carlo_summary.json
               │   ├── ranking_scan.csv
               │   ├── selected_candidates.txt
               │   └── mc_*/
               ├── V_O_02/
               ├── V_O_03/
               └── V_O_04/

Each selected Stage-1 configuration contains a generated POSCAR and GRACE search
metadata. Stage 2 adds a final-backend single-point record, relaxed structure,
and final metadata in the same dedicated output tree.

Thermodynamic compatibility
---------------------------

Monte Carlo may use::

   solid_configurational_entropy = "none"

or::

   solid_configurational_entropy = "ideal"

The explicit canonical partition-function mode ``"configurational"`` requires
exact symmetry enumeration and exact orbit degeneracies and is therefore not
available for Monte Carlo.

If an experimentally fitted solid-energy correction is applied to vacancy
thermodynamics, use a raw same-backend oxygen reservoir such as
``reference_file`` or ``same_calculator``. Do not combine the fitted solid-energy
correction with ``oxygen_reference_mode = "global"`` or
``"chemistry-specific"`` because both paths use experimental formation-enthalpy
information for oxygen-related calibration.

Reference-state caveat when cations move
----------------------------------------

The current Stage-2 ``n=0`` parent reference is the replicated source parent; it
is not subjected to an independent cation-only Monte Carlo search. When
``mc_cation_move_weight`` is non-zero, the defective ``n=1,2,3,4`` minima may
therefore contain both vacancy rearrangement and cation reordering relative to
the source parent.

This is appropriate for a coupled ordering search. A strict vacancy formation
free energy referenced to an equilibrated cation arrangement would require an
independently equilibrated ``n=0`` cation-ordering baseline.

GUI
---

The dedicated Streamlit page ``gui/pages/Vacancy_MC_Staged.py`` exposes the
staged parent selection, output routing, explicit vacancy counts, supercell,
GRACE calculator, MC schedule, move weights, and MACE finalization settings. It
also builds separate ``conda run`` commands for the GRACE and MACE environments.

See :doc:`methods/vacancies` for the complete scientific workflow and
:doc:`examples/vacancies` for the checked-in production-oriented example.
