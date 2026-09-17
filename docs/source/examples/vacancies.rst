Staged vacancy Monte Carlo example
==================================

The complete production-oriented example is in ``examples/vacancies``. It uses
GRACE for the coupled cation/vacancy Monte Carlo occupation search and MACE for
the final single points, relaxation, reranking, and thermodynamic analysis.

The two stages intentionally run in separate environments::

   conda activate dopingflow-grace
   dopingflow vacancies-mc-search -c examples/vacancies/input.toml --verbose

   conda activate dopingflow-mace
   dopingflow vacancies-finalize -c examples/vacancies/input.toml --verbose

The same ``input.toml`` is used by both commands. GRACE and MACE therefore do not
need to coexist in one Python environment.

Parent filtering and output handoff
-----------------------------------

The example selects an existing multi-composition parent tree and routes the
entire staged study into a dedicated output directory::

   [vacancies]
   parent_source = "directory"
   parent_directory = "vacancy-selected"
   parent_pick = "lowest_energy"
   output_directory = "vacancy-mc-grace-mace"

``parent_include`` optionally limits the study to named compositions or exact
``composition/candidate`` IDs. Common composition labels are canonicalized, so
``Ti_2.5Sb_5``, ``Ti2p5_Sb5``, and ``Sb5_Ti2p5`` match the same composition.

``parent_pick = "lowest_energy"`` keeps the first parent listed in each
composition's ``selected_candidates.txt``. DopingFlow's filter stage writes that
file in ascending relaxed-energy order. Use ``parent_pick = "all"`` to process
every selected parent.

``output_directory`` mirrors the selected parent IDs into a separate result
tree. Source parent structures remain untouched. The GRACE search writes its
archive and ``selected_candidates.txt`` there; MACE finalization resolves the
same result tree and continues from those saved selections.

Search space
------------

The checked-in example explicitly studies one and two oxygen vacancies in a
``2 x 2 x 2`` replicated parent::

   vacancy_counts = [1, 2]
   supercell = [2, 2, 2]

For the project's 120-atom SnO2-based parent, this gives 960 atoms before
vacancy removal: 320 cations and 640 oxygen sites. Dopant counts are multiplied
by eight but their percentages are unchanged.

The Monte Carlo state contains both cation occupations and oxygen-vacancy
occupations. ``cation_swap`` exchanges two distinct cation species and
``vacancy_swap`` moves the vacancy marker across the oxygen sublattice. Keeping
both move weights non-zero samples coupled dopant-vacancy ordering.

GRACE search stage
------------------

A production-oriented starting schedule for this large occupation space is::

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
   sample_seed = 42
   sample_max_saved = 100

``mc_max_steps`` is the total trajectory length, including the hot hold and
cooling ramp. In the current implementation the no-improvement patience counter
is active from step 1. If completing the full annealing schedule is required,
choose ``mc_patience`` larger than
``mc_annealing_hold_steps + mc_annealing_steps``.

The library default of 10,000 steps is deliberately conservative and useful for
smoke tests. A larger explicit ceiling is appropriate for the 960-atom coupled
occupation problem, but convergence should be established for the actual
chemistry rather than inferred from a fixed step count.

MACE finalization stage
-----------------------

The final calculator is independent of the GRACE environment::

   backend = "mace"
   model = "mh-1"
   task = "matpes_r2scan"
   device = "cuda"
   topk_per_vacancy_count = 20

The current selection sequence is::

   GRACE archive
      -> GRACE ranking at fixed n
      -> GRACE top-k
      -> MACE single point on selected candidates
      -> MACE relaxation
      -> MACE relaxed-energy reranking
      -> MACE thermodynamics

MACE does not yet rescore every archived GRACE structure before the top-k is
chosen. For a new chemistry, compare GRACE and MACE rankings on a smaller archive
before committing to a large production campaign.

Reference-state caveat
----------------------

When cation swaps are enabled, the ``n=1`` and ``n=2`` defective minima can
contain a different cation ordering from the replicated source parent. The
current ``n=0`` reference is not subjected to a separate cation-only Monte Carlo
search. Consequently, vacancy formation free energies from this coupled study
can contain both vacancy-formation and cation-reordering contributions. A
strict vacancy formation energy referenced to an equilibrated cation arrangement
would require a corresponding ``n=0`` cation-only baseline.

Thermodynamics
--------------

Monte Carlo supports ``solid_configurational_entropy = "none"`` and ``"ideal"``.
The explicit canonical partition-function mode requires exact symmetry
enumeration and exact orbit degeneracies, so it is not available for the Monte
Carlo path.

Cross-count thermodynamics requires an oxygen chemical potential; raw ML total
energies for different vacancy counts are not directly comparable because the
structures contain different numbers of oxygen atoms.

See :doc:`../methods/vacancies` for the complete staged workflow, file layout,
fingerprints, provenance, and thermodynamic interpretation. See
:doc:`../methods/oxygen_calibration` and
:doc:`../methods/vacancy_energy_correction` for the two alternative oxygen/
formation-energy calibration paths and their compatibility rules.
