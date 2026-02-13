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

``refs -> generate -> scan -> relax -> filter -> bandgap -> formation -> collect``


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

Step 00: reference energies:

::

   dopingflow refs-build -c input.toml

Step 01: structure generation:

::

   dopingflow generate -c input.toml

Step 02: scan (symmetry-unique enumeration + M3GNet single-point energies):

::

   dopingflow scan -c input.toml

Step 03: relax scanned candidates (M3GNet Relaxer):

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


Outputs Overview
----------------

This section summarizes the main outputs created by each stage.

Step 00 (refs-build)
~~~~~~~~~~~~~~~~~~~~

Writes:

- ``reference_structures/reference_energies.json``

This file contains:

- the relaxed pristine supercell energy
- per-atom chemical potentials for host and dopant species
- metadata about reference structures and relaxation settings

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
- candidate structures:

  ::

     <composition_tag>/candidate_###/01_scan/POSCAR
     <composition_tag>/candidate_###/01_scan/meta.json

Step 03 (relax)
~~~~~~~~~~~~~~~

For each candidate:

- ``candidate_###/02_relax/POSCAR``
- ``candidate_###/02_relax/meta.json``

Also writes per composition folder:

- ``ranking_relax.csv``

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


Tips
----

- Use ``--verbose`` with any command for more detailed logs:

  ::

     dopingflow run-all -c input.toml --verbose

- If bandgap is not configured yet (no ``ALIGNN_MODEL_DIR``), stop before bandgap:

  ::

     dopingflow run-all -c input.toml --until filter
