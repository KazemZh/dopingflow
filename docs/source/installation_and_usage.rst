Installation, Usage, and Outputs
=================================

This page explains how to install **dopingflow**, run the ordinary workflow,
and run the staged GRACE-to-MACE vacancy Monte Carlo path when the two backends
live in separate environments.

Installation
------------

Clone the repository and install only the extras needed in a given environment::

   git clone https://github.com/KazemZh/dopingflow.git
   cd dopingflow
   conda create -n dopingflow python=3.11 pip -y
   conda activate dopingflow
   pip install -U pip

Examples::

   pip install -e ".[mace]"
   pip install -e ".[grace]"
   pip install -e ".[m3gnet]"
   pip install -e ".[uma]"
   pip install -e ".[alignn]"
   pip install -e ".[gui]"
   pip install -e ".[corrections]"
   pip install -e ".[dev]"

Verify the CLI::

   dopingflow --help

Separate GRACE and MACE environments
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

The staged vacancy workflow deliberately avoids requiring GRACE and MACE in the
same Python environment. A typical setup is::

   conda create -n dopingflow-grace python=3.11 pip -y
   conda activate dopingflow-grace
   pip install -e ".[grace]"

   conda create -n dopingflow-mace python=3.11 pip -y
   conda activate dopingflow-mace
   pip install -e ".[mace]"

Both environments may use the same editable repository checkout and the same
project ``input.toml``.

Required inputs
---------------

Refer to :ref:`Required Input Files <input_file_req>` and
:doc:`input_file`. Staged vacancy-specific controls are documented separately in
:doc:`input_file_vacancy_mc`.

Running the ordinary workflow
-----------------------------

All commands accept ``-c/--config``. If omitted, ``input.toml`` in the current
directory is used.

Run the ordinary pipeline with::

   dopingflow run-all -c input.toml

The normal stage order is::

   refs -> corrections -> generate -> scan -> relax -> filter -> bandgap
        -> formation -> collect -> alloy-hull -> phase-diagram -> vacancies

The correction stage is a no-op unless ``[energy_correction].enabled = true``.
The default ``run-all`` stop remains the phase-diagram stage; append vacancies
explicitly when required::

   dopingflow run-all -c input.toml --until vacancies

Individual stages
~~~~~~~~~~~~~~~~~

::

   dopingflow refs-build -c input.toml
   dopingflow corrections-fit -c input.toml
   dopingflow generate -c input.toml
   dopingflow scan -c input.toml
   dopingflow relax -c input.toml
   dopingflow filter -c input.toml
   dopingflow bandgap -c input.toml
   dopingflow formation -c input.toml
   dopingflow collect -c input.toml
   dopingflow alloy-hull -c input.toml
   dopingflow phase-diagram -c input.toml
   dopingflow vacancies -c input.toml
   dopingflow surface -c input.toml

``dopingflow vacancies`` is the one-process vacancy command. It supports the
symmetry-enumeration workflow and Monte Carlo when all requested calculators are
available in the same environment.

Resuming and partial runs
~~~~~~~~~~~~~~~~~~~~~~~~~

Resume from a stage::

   dopingflow run-all -c input.toml --from relax

Stop at a stage::

   dopingflow run-all -c input.toml --until filter

Print the plan without running::

   dopingflow run-all -c input.toml --dry-run

Run only a subset inside a range::

   dopingflow run-all -c input.toml --from refs --until collect \
     --only refs,corrections,generate,scan

Filter-specific overrides remain available through ``run-all``::

   dopingflow run-all -c input.toml --from relax --until filter \
     --filter-only Sb5_Zr5
   dopingflow run-all -c input.toml --from filter --until filter --force
   dopingflow run-all -c input.toml --from filter --until filter --window-mev 50
   dopingflow run-all -c input.toml --from filter --until filter --topn 12

Staged vacancy Monte Carlo
--------------------------

Use the staged commands when the Monte Carlo search calculator and final
calculator are installed in different environments. The recommended GRACE/MACE
sequence is::

   conda activate dopingflow-grace
   dopingflow vacancies-mc-search -c input.toml --verbose

   conda activate dopingflow-mace
   dopingflow vacancies-finalize -c input.toml --verbose

``vacancies-mc-search`` checks/builds only the ``mc_*`` calculator and runs the
occupation search. ``vacancies-finalize`` checks/builds only the ordinary final
vacancy calculator and consumes the persisted Stage-1 selection.

Both stages use the same flat ``[vacancies]`` table. A representative staged
setup is::

   [vacancies]
   parent_source = "directory"
   parent_directory = "vacancy-selected"
   parent_pick = "lowest_energy"
   output_directory = "vacancy-mc-grace-mace"

   parent_include = [
       "Ti_2.5Sb_2.5",
       "Ti_2.5Sb_5",
       "Ce_2.5Sb_2.5",
   ]

   search_method = "monte-carlo"
   vacancy_counts = [1, 2]
   supercell = [2, 2, 2]

   mc_backend = "grace"
   mc_model = "GRACE-1L-OMAT"
   mc_task = ""
   mc_device = "cuda"
   mc_gpu_id = 0

   mc_annealing = true
   mc_initial_temperature_K = 1500.0
   mc_annealing_hold_steps = 5000
   mc_annealing_steps = 50000
   mc_temperature_K = 600.0
   mc_run_mode = "combined"
   mc_max_steps = 200000
   mc_patience = 100000
   mc_improvement_tolerance_eV = 1.0e-5
   mc_energy_window_eV = 1.0
   mc_cation_move_weight = 0.5
   mc_vacancy_move_weight = 0.5
   sample_seed = 42
   sample_max_saved = 100

   backend = "mace"
   model = "mh-1"
   task = "matpes_r2scan"
   device = "cuda"
   gpu_id = 0
   topk_per_vacancy_count = 20

``parent_include`` accepts common composition labels independently of element
order and ``2.5``/``2p5`` notation. A selector containing ``/`` is treated as
an exact ``composition/candidate`` parent ID.

``parent_pick = "lowest_energy"`` keeps the first selected parent for each
composition. The normal filter stage writes ``selected_candidates.txt`` in
ascending relaxed-energy order.

``output_directory`` keeps the complete staged result tree separate from source
parents. Stage 1 and Stage 2 mirror and reuse the same parent IDs there.

``mc_max_steps`` is the total trajectory length; the high-temperature hold and
cooling ramp are part of that number. In ``combined`` and ``converged`` modes,
the no-improvement patience counter is active from step 1. If the complete
annealing schedule must be reached, choose::

   mc_patience > mc_annealing_hold_steps + mc_annealing_steps

The 200,000/100,000 values above are a project-specific production starting
point for a large 2x2x2 coupled occupation search, not a universal convergence
guarantee.

Current cross-backend selection behavior
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

The present staged selection sequence is::

   GRACE archive
      -> GRACE ranking
      -> GRACE top-k
      -> MACE single point on GRACE-selected candidates
      -> MACE relaxation
      -> MACE relaxed-energy reranking

MACE does not currently rescore every archived GRACE candidate before top-k
selection. Validate GRACE/MACE ranking agreement on a smaller archive before a
large production campaign.

Vacancy reference-state caveat
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

When cation swaps are enabled, defective ``n > 0`` minima can differ from the
replicated source parent in both vacancy location and cation ordering. The
current ``n=0`` parent reference is not subjected to an independent cation-only
Monte Carlo search. A vacancy formation free energy from such a run can therefore
contain both vacancy-formation and cation-reordering contributions. A strict
vacancy formation energy referenced to an equilibrated cation arrangement would
require a corresponding ``n=0`` cation-ordering baseline.

Outputs overview
----------------

References
~~~~~~~~~~

``refs-build`` writes ``reference_structures/reference_energies.json`` plus the
relaxed host/reference structures and calculator/relaxation provenance.

Correction fit
~~~~~~~~~~~~~~

When enabled, ``corrections-fit`` writes a signature-specific directory below::

   reference_structures/corrections/<backend-model-task>-<signature-hash>/

The directory contains the fitted parameters, reports, metadata, accepted and
rejected calibration data, and any required relaxed calibration structures.

Generation
~~~~~~~~~~

The generation stage writes one composition directory under
``[structure].outdir`` containing the generated POSCAR and metadata.

Scan
~~~~

Per composition, scan writes ``ranking_scan.csv`` and candidate
``01_scan/POSCAR`` / ``01_scan/meta.json`` files.

Relax
~~~~~

Per candidate, relaxation writes ``02_relax/POSCAR`` and ``02_relax/meta.json``.
Per composition it writes ``ranking_relax.csv``.

Filter
~~~~~~

Filtering writes ``ranking_relax_filtered.csv`` and ``selected_candidates.txt``.
The latter is ordered by relaxed energy and is therefore also the basis for
``parent_pick = "lowest_energy"`` in the staged vacancy workflow.

Bandgap
~~~~~~~

Bandgap prediction writes the candidate ``03_band/meta.json`` records and a
per-composition summary.

Formation and collect
~~~~~~~~~~~~~~~~~~~~~

Formation writes candidate ``04_formation/meta.json`` records and a
per-composition formation table. Collection writes the project-level
``results_database.csv``.

Alloy hull and phase diagram
~~~~~~~~~~~~~~~~~~~~~~~~~~~~

The restricted alloy-hull stage writes ``alloy_hull_results.csv`` and
``alloy_hull_summary.json``. The full phase-diagram stage writes
``phase_diagram_results.csv`` plus one exact-system table below
``phase_diagrams/``.

One-process vacancy outputs
~~~~~~~~~~~~~~~~~~~~~~~~~~~

For the ordinary ``dopingflow vacancies`` path, each selected parent receives a
``05_vacancies/`` tree. The normal global files are::

   <structure.outdir>/vacancies_database.csv
   <structure.outdir>/vacancies_database.json

If ``parent_source = "directory"``, the global files are written under that
parent root instead.

Staged vacancy outputs
~~~~~~~~~~~~~~~~~~~~~~

With ``output_directory = "vacancy-mc-grace-mace"``, Stage 1 writes::

   vacancy-mc-grace-mace/vacancy_mc_search_database.csv
   vacancy-mc-grace-mace/vacancy_mc_search_database.json

and each mirrored parent receives count-specific search groups such as::

   <output_directory>/<composition>/<candidate>/05_vacancies/V_O_01/
   <output_directory>/<composition>/<candidate>/05_vacancies/V_O_02/

These groups contain ``monte_carlo_summary.json``, ``ranking_scan.csv``,
``selected_candidates.txt``, and archived generated structures/metadata.

Stage 2 adds final-backend single-point records, relaxed structures, final
rankings, and global::

   vacancy-mc-grace-mace/vacancies_database.csv
   vacancy-mc-grace-mace/vacancies_database.json

Search-energy and final-energy provenance remain separate. A GRACE-to-MACE
energy difference is not reported as a same-calculator relaxation-energy change.

Vacancy thermodynamics
~~~~~~~~~~~~~~~~~~~~~~

With ``static_thermodynamic_analysis = true``, the vacancy stage writes the
static minima, exact direct-``delta_mu_O`` stability intervals, selected-condition
best counts, T-pO2 map, and finite-temperature vacancy-formation free-energy
outputs.

Raw total energies across different oxygen contents must not be compared
without an oxygen reservoir. Monte Carlo supports
``solid_configurational_entropy = "none"`` and ``"ideal"``. The explicit
partition-function mode requires exact symmetry enumeration and exact orbit
degeneracies.

Use ``oxygen_standard_state_mode = "nist_shomate"`` for the built-in O2 gas
standard-state thermal correction or ``user_table`` for a validated alternative.

Surface workflow
~~~~~~~~~~~~~~~~

The surface stage is not part of ``run-all``. Execute it after inspecting the
selected bulk candidates::

   dopingflow surface -c input.toml

It writes generated/relaxed slab files and a global ``surface_summary.csv``.

Sequential doping workflow
--------------------------

Run gradual composition-by-composition doping with::

   dopingflow sequential-run -c input.toml

In ``mode = "full"``, each sequential step performs::

   generate -> scan -> relax -> filter -> optional bandgap -> formation -> collect

The lowest-energy relaxed structure from each step is copied to
``best_relaxed/POSCAR`` and becomes the base for the next composition.

``mode = "recompute_energies"`` reuses the existing relaxed sequential
structures and reruns only formation-energy evaluation and database collection.
This is useful when changing thermodynamic references without regenerating or
rerelaxing structures.

Tips
----

Use ``--verbose`` for detailed logs::

   dopingflow vacancies-mc-search -c input.toml --verbose

Before a full staged production run, perform an end-to-end smoke test on one
composition with a tiny trajectory, for example::

   parent_include = ["Ti_2.5Sb_2.5"]
   mc_max_steps = 20
   mc_patience = 20
   sample_max_saved = 10
   topk_per_vacancy_count = 3

Then benchmark a larger but still short GRACE trajectory on one representative
2x2x2 parent before estimating the total campaign cost.
