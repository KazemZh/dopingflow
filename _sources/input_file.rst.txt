.. _input_file_spec:

Input File Specification (input.toml)
=====================================

The workflow is fully controlled through a single TOML configuration file.

Each stage reads only the parameters relevant to it.
All paths are interpreted relative to the directory containing ``input.toml``.

The following sections are supported:

- ``[structure]``
- ``[doping]``
- ``[generate]``
- ``[scan]``
- ``[relax]``
- ``[filter]``
- ``[bandgap]``
- ``[formation]``
- ``[database]`` (optional)

Not all sections are required for every stage. Each stage reads only what it needs.


[structure]
-----------

Defines the base structure and global output directory.

base_poscar (string)
~~~~~~~~~~~~~~~~~~~~
Path to the pristine structure file (POSCAR format).

Used in:
- Step 00 (reference construction)
- Step 01 (structure generation)

supercell (array of 3 integers)
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
Supercell expansion applied to the pristine structure:

::

   supercell = [nx, ny, nz]

Used in:
- Step 00
- Step 01

outdir (string, default: "random_structures")
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
Directory where all generated composition folders are written.

This directory becomes the root for:
- Step 01 outputs
- Step 02–06 per-composition subfolders


[doping]
--------

Defines substitutional doping behavior.

host_species (string, required)
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
Element symbol of the host species to be substituted.

Used in:
- Step 01 (generation)
- Step 02 (scan)
- Step 06 (formation)

mode (string)
~~~~~~~~~~~~~
Defines doping mode:

- ``"explicit"`` — user provides exact compositions.
- ``"enumerate"`` — workflow constructs compositions combinatorially.

max_dopants_total (integer)
~~~~~~~~~~~~~~~~~~~~~~~~~~~
Maximum number of distinct dopant species allowed per structure.

Only enforced in enumerate mode.

dopant_elements (array of strings)
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
List of possible dopant species.

required_elements (array of strings, optional)
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
Subset of dopants that must appear in enumerated compositions.

total_levels (array of floats)
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
Allowed total dopant percentages (relative to host sites).

per_dopant_levels (array of floats)
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
Discrete allowed concentrations per dopant.


[generate]
----------

Controls structure writing and reproducibility (Step 01).

seed_base (integer)
~~~~~~~~~~~~~~~~~~~
Base seed for deterministic random substitution.

Ensures reproducible structure generation.

poscar_order (array of strings, required)
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
Defines element ordering in written POSCAR files.

Must be non-empty.

Used consistently in:
- Step 01
- Step 02
- Step 03


[scan]
------

Controls symmetry enumeration and single-point prescreening (Step 02).

poscar_in (string, default: "POSCAR")
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
Filename inside each composition folder used as enumeration input.

topk (integer)
~~~~~~~~~~~~~~
Number of lowest-energy configurations retained.

symprec (float)
~~~~~~~~~~~~~~~
Tolerance passed to ``SpacegroupAnalyzer`` for symmetry detection.

max_enum (integer)
~~~~~~~~~~~~~~~~~~
Maximum allowed number of raw combinatorial configurations.
Prevents runaway combinatorics.

max_unique (integer)
~~~~~~~~~~~~~~~~~~~~
Maximum allowed number of symmetry-unique configurations.

nproc (integer)
~~~~~~~~~~~~~~~
Number of parallel worker processes for energy evaluation.

chunksize (integer)
~~~~~~~~~~~~~~~~~~~
Chunk size used in multiprocessing pool.

anion_species (array of strings)
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
Species excluded from substitutional enumeration (e.g., oxygen).


[relax]
-------

Controls structural relaxation (Step 03).

fmax (float)
~~~~~~~~~~~~
Maximum force convergence criterion (eV/Å).

n_workers (integer)
~~~~~~~~~~~~~~~~~~~
Number of parallel relaxation workers.

tf_threads (integer)
~~~~~~~~~~~~~~~~~~~~
TensorFlow intra/inter-op thread count per worker.

omp_threads (integer)
~~~~~~~~~~~~~~~~~~~~~
OpenMP thread count per worker.

skip_if_done (boolean)
~~~~~~~~~~~~~~~~~~~~~~
If true, skip composition folder if ``ranking_relax.csv`` exists.

skip_candidate_if_done (boolean)
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
If true, skip candidate if ``02_relax/meta.json`` exists.


[filter]
--------

Controls candidate selection after relaxation (Step 04).

mode (string)
~~~~~~~~~~~~~
Either:

- ``"window"`` — keep structures within an energy window
- ``"topn"`` — keep lowest N structures

window_meV (float)
~~~~~~~~~~~~~~~~~~
Energy window (meV above minimum) used when mode = ``"window"``.

max_candidates (integer)
~~~~~~~~~~~~~~~~~~~~~~~~
Number of candidates kept when mode = ``"topn"``.

skip_if_done (boolean)
~~~~~~~~~~~~~~~~~~~~~~
Skip filtering if output files already exist.


[bandgap]
---------

Controls ALIGNN bandgap prediction (Step 05).

skip_if_done (boolean)
~~~~~~~~~~~~~~~~~~~~~~
Skip bandgap calculation if summary file exists.

cutoff (float)
~~~~~~~~~~~~~~
Radial cutoff for graph construction (Å).

max_neighbors (integer)
~~~~~~~~~~~~~~~~~~~~~~~
Maximum neighbors per atom in graph.

Note:
The ALIGNN model directory must be defined via environment variable:

::

   ALIGNN_MODEL_DIR=/path/to/model


[formation]
-----------

Controls formation energy calculation (Step 06).

skip_if_done (boolean)
~~~~~~~~~~~~~~~~~~~~~~
Skip formation calculation if output exists.

normalize (string)
~~~~~~~~~~~~~~~~~~
Defines reported energy normalization:

- ``"total"`` — total formation energy (eV)
- ``"per_dopant"`` — eV per substituted dopant atom
- ``"per_host"`` — eV per atom in supercell

Formation energies require:

- Step 00 reference construction completed
- ``reference_structures/reference_energies.json``


[database] (optional)
---------------------

Controls final database collection (Step 07).

skip_if_done (boolean, default: true)
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
If true, do not overwrite existing ``results_database.csv``.

If this section is omitted, default behavior applies.


Design Principles
-----------------

- All stages are deterministic given fixed input.
- All randomness is seed-controlled.
- Each stage can be skipped independently via ``skip_if_done``.
- Each stage writes explicit metadata for full reproducibility.
- All outputs are composition-folder scoped except final database export.
