Installation, Usage, and Outputs
=================================

This page explains how to install **dopingflow** and how to run the workflow
either step-by-step or using the single orchestration command.

Installation
------------

Clone the repository and install in editable mode:

::

   git clone KazemZh/dopingflow
   cd dopingflow
   python -m venv .venv
   source .venv/bin/activate
   pip install -U pip
   pip install -e .

Verify the CLI is available:

::

   dopingflow --help


Required Inputs
---------------

Refer to :ref:`Required Input Files <input_file_req>` page.


Running the Workflow
--------------------

All commands accept ``-c/--config`` to specify the TOML file.
If omitted, ``input.toml`` in the current directory is used.

Run the full pipeline with one command
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

To run the complete workflow in order:

::

   dopingflow run-all -c input.toml

This executes:

``refs -> generate -> scan -> relax -> filter -> bandgap -> formation -> collect -> alloy-hull -> phase-diagram``

Vacancies are an optional final run-all step, leaving the default stop unchanged::

   dopingflow run-all -c input.toml --until vacancies
   dopingflow run-all -c input.toml --only vacancies

The surface stage is not included in ``run-all`` and must be executed separately:

::

   dopingflow surface -c input.toml

This design allows users to first inspect and validate the final database
before generating surface structures.


Resuming and partial runs (run-all)
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

You can resume from a given stage:

::

   dopingflow run-all -c input.toml --from relax

You can stop at a stage (inclusive). This is useful if you do not want to run bandgap yet:

::

   dopingflow run-all -c input.toml --until filter

You can print the planned steps without running them:

::

   dopingflow run-all -c input.toml --dry-run

You can run only a subset of steps inside a selected range:

::

   dopingflow run-all -c input.toml --from refs --until collect --only refs,generate,scan


Filtering controls inside run-all
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

The filter stage supports optional overrides (passed through by ``run-all``):

- Restrict filtering to a single composition folder:

  ::

     dopingflow run-all -c input.toml --from relax --until filter --filter-only Sb5_Zr5

- Force re-filtering even if outputs exist:

  ::

     dopingflow run-all -c input.toml --from filter --until filter --force

- Override filtering mode by specifying one of:

  ::

     dopingflow run-all -c input.toml --from filter --until filter --window-mev 50
     dopingflow run-all -c input.toml --from filter --until filter --topn 12


Step-by-step execution
~~~~~~~~~~~~~~~~~~~~~~

Step 00: build and relax thermodynamic reference structures:

::

   dopingflow refs-build -c input.toml

Step 01: structure generation:

::

   dopingflow generate -c input.toml

Step 02: scan (symmetry-unique enumeration / sampling + ML-based single-point energies):

::

   dopingflow scan -c input.toml

Step 03: relax scanned candidates (ML backend + ASE optimizer):

::

   dopingflow relax -c input.toml

Step 04: filter relaxed candidates:

::

   dopingflow filter -c input.toml

Optional Step 05: predict bandgap (ALIGNN):

Before running bandgap, set the model path:

::

   export ALIGNN_MODEL_DIR=/path/to/your/alignn/model_root
   dopingflow bandgap -c input.toml

Step 06: formation energies:

::

   dopingflow formation -c input.toml

Step 07: collect results into one CSV database:

::

   dopingflow collect -c input.toml

Step 08: generate surfaces and optionally relax slabs:

::

   dopingflow surface -c input.toml

Unified oxygen-vacancy workflow:

::

   dopingflow vacancies -c input.toml

This one command performs charge-count determination, symmetry enumeration,
single-point screening, fixed-count top-k selection, ML relaxation, and optional
oxygen-grand-potential analysis. Its seven logged phases are implementation
details, not separate public commands.

Outputs Overview
----------------

This section summarizes the main outputs created by each stage.

Step 00 (refs-build)
~~~~~~~~~~~~~~~~~~~~

Writes:

- ``reference_structures/reference_energies.json``

This file contains:

- relaxed host unit-cell and supercell energies
- relaxed reference structure energies
- metadata about backend, optimizer, device, and convergence settings
- reference information used for formation energy evaluation

Additional outputs:

- ``reference_structures/relaxed/host_unit_relaxed.POSCAR``
- ``reference_structures/relaxed/host_supercell_<a>x<b>x<c>_relaxed.POSCAR``
- ``reference_structures/relaxed/refs/<name>_relaxed.POSCAR``


Step 01 (generate)
~~~~~~~~~~~~~~~~~~

Writes a structure folder per composition under ``[structure].outdir`` (default: ``random_structures``):

- ``<outdir>/<composition_tag>/POSCAR``
- ``<outdir>/<composition_tag>/metadata.json``


Step 02 (scan)
~~~~~~~~~~~~~~

Inside each ``<composition_tag>/`` folder, writes:

- ``ranking_scan.csv`` (top-k single-point energies)
- ``scan_summary.txt`` (human-readable summary)

Candidate structures:

::

   <composition_tag>/candidate_###/01_scan/POSCAR
   <composition_tag>/candidate_###/01_scan/meta.json

The scan stage evaluates structures using a selected ML backend
(e.g. M3GNet, UMA, MACE, GRACE).


Step 03 (relax)
~~~~~~~~~~~~~~~

For each candidate:

- ``candidate_###/02_relax/POSCAR``
- ``candidate_###/02_relax/meta.json``

Also writes per composition folder:

- ``ranking_relax.csv``

The relaxation stage uses:

- ML interatomic potentials for forces
- ASE optimizers for structural relaxation


Step 04 (filter)
~~~~~~~~~~~~~~~~

Writes per composition folder:

- ``ranking_relax_filtered.csv`` (filtered candidate table)
- ``selected_candidates.txt`` (names of kept candidates)


Step 05 (bandgap)
~~~~~~~~~~~~~~~~~

Writes per composition folder:

- ``bandgap_alignn_summary.csv``

Writes per candidate:

- ``candidate_###/03_band/meta.json``


Step 06 (formation)
~~~~~~~~~~~~~~~~~~~

Writes per composition folder:

- ``formation_energies.csv``

Writes per candidate:

- ``candidate_###/04_formation/meta.json``


Step 07 (collect)
~~~~~~~~~~~~~~~~~

Writes one flat CSV in the workflow root:

- ``results_database.csv``

This file is a compact “database view” across compositions and selected candidates,
combining scan/relax/filter/bandgap/formation results where available.

Step 08 (restricted alloy hull)
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

Writes ``alloy_hull_results.csv`` and ``alloy_hull_summary.json`` for the
configured substitutional alloy line.


Step 09 (phase diagram)
~~~~~~~~~~~~~~~~~~~~~~~

Writes the combined ``phase_diagram_results.csv`` plus one exact-system CSV
under ``phase_diagrams/`` for every chemical system represented by candidates.


Step 10 (vacancies, optional)
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

Each selected parent receives ``candidate_###/05_vacancies/`` with the parent
reference, per-count ranking tables, generated structures, and selected relaxed
structures. Global tables remain separate from ``results_database.csv``::

   <structure.outdir>/vacancies_database.csv
   <structure.outdir>/vacancies_database.json

With ``[vacancies].thermodynamic_analysis = true``, the same command also writes
``vacancy_minima_by_composition.csv``, ``vacancy_stability_intervals.csv``, and
``vacancy_best_counts.csv``. The first detailed database contains every
configuration; the compact files contain minima and oxygen-chemical-potential
stability results. Never compare raw totals across different oxygen contents.


Optional surface workflow
~~~~~~~~~~~~~~~~~~~~~~~~~

Generates slab structures from selected candidates and optionally relaxes them.

Input:

- ``results_database.csv`` (from Step 07)

Writes per candidate:

::

   <outdir>/<composition_tag>/candidate_###/hkl_h_k_l/term_###/

Files per slab:

- ``POSCAR`` (generated slab)
- ``CONTCAR`` (relaxed slab, if enabled)
- ``meta.json`` (slab metadata)

Optional relaxation outputs:

- ``surface_relax.log``
- ``surface_relax.traj``
- ``surface_relax.json``

Global output:

::

   <outdir>/surface_summary.csv

The surface stage uses:

- pymatgen for slab generation
- the same ML backend abstraction as Step 03 for relaxation
- ASE optimizers for slab relaxation

Tips
----

- Use ``--verbose`` with any command for more detailed logs:

  ::

     dopingflow run-all -c input.toml --verbose

- If bandgap is not configured yet (no ``ALIGNN_MODEL_DIR``), stop before bandgap:

  ::

     dopingflow run-all -c input.toml --until filter

- Run surface generation only after verifying final candidates:

  ::

     dopingflow collect -c input.toml
     dopingflow surface -c input.toml

- Start with a single composition and one candidate to validate surface settings:

  ::

     composition_tag = "Sb50"
     selection_mode = "id"
     candidate_id = 1

Sequential Doping Workflow
--------------------------

The sequential doping workflow is designed for gradual composition-by-composition
doping studies. Instead of generating each target concentration independently
from the pristine host, the workflow uses the lowest-energy relaxed structure
from the previous composition as the base structure for the next composition.

Command
-------

::

   dopingflow sequential-run -c input.toml

Execution Logic
---------------

For each composition generated from the ``[doping]`` section, the workflow creates
a separate step folder under ``[sequential].outdir``:

::

   sequential_structures/
      step_001_Sb2p5/
      step_002_Sb5/
      step_003_Sb7p5/
      ...

In ``mode = "full"``, each step runs:

::

   generate -> scan -> relax -> filter -> optional bandgap -> formation -> collect

After relaxation, the lowest-energy relaxed candidate is copied to:

::

   step_xxx_<composition>/best_relaxed/POSCAR

This structure is then used as the base POSCAR for the next sequential step.

Modes
-----

full
~~~~

Runs the complete sequential workflow for each composition.

::

   [sequential]
   mode = "full"

recompute_energies
~~~~~~~~~~~~~~~~~~

Reuses existing relaxed sequential structures and reruns only formation-energy
evaluation and database collection.

This is useful when changing the thermodynamic reference, for example:

::

   oxides_ref = ["Sb2O5"]

without regenerating, scanning, or relaxing structures again.

::

   [sequential]
   mode = "recompute_energies"

Outputs
-------

Each step writes its own local database:

::

   sequential_structures/step_001_Sb2p5/results_database.csv
   sequential_structures/step_002_Sb5/results_database.csv

At the end, all step databases are merged into the project-root database:

::

   results_database.csv

Bandgap Control
---------------

The bandgap stage can be skipped in sequential calculations using:

::

   [bandgap]
   enabled = false

When disabled, the final database is still written, but the ``bandgap_eV`` column
is left empty.
