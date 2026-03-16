.. _input_file_spec:

Input File Specification (input.toml)
=====================================

The workflow is fully controlled through a single TOML configuration file.

Each stage reads only the parameters relevant to it.
All paths are interpreted relative to the directory containing ``input.toml``.

The following sections are supported:

- ``[references]``
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

---------------------------------------------------------------------

[references]
------------

Step 00 — Reference construction and relaxation.

This stage prepares **all thermodynamic reference structures** and writes:

::

   reference_structures/reference_energies.json

It performs:

- Relaxation of the host oxide unit cell
- Construction and relaxation of the host supercell
- Relaxation of metal reference phases (metal mode)
- Relaxation of oxide reference phases (oxide mode)
- Relaxation of O₂ gas (oxide mode)
- Storage of all relaxed POSCAR files for reuse

Common Parameters
~~~~~~~~~~~~~~~~~

reference_mode (string)
^^^^^^^^^^^^^^^^^^^^^^^

Choose reference scheme:

- ``"metal"``
- ``"oxide"``

skip_if_done (boolean)
^^^^^^^^^^^^^^^^^^^^^^

Skip reconstruction if JSON cache exists.

fmax (float)
^^^^^^^^^^^^

Force convergence criterion used in relaxation (eV/Å).

supercell (array of 3 integers)
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

Supercell used later for doping.
The host supercell is constructed and relaxed at this stage.

host (string)
^^^^^^^^^^^^^

Chemical formula of the host oxide (e.g. ``"SnO2"``).

host_dir (string)
^^^^^^^^^^^^^^^^^

Directory containing ``<host>.POSCAR``.

Metal Reference Mode
~~~~~~~~~~~~~~~~~~~~

Used when ``reference_mode = "metal"``.

metal_ref (array of strings)
^^^^^^^^^^^^^^^^^^^^^^^^^^^^

List of metal element symbols used as reference phases.

metals_dir (string)
^^^^^^^^^^^^^^^^^^^

Directory containing ``<Element>.POSCAR`` files.

Example::

   reference_structures/metals/Sn.POSCAR
   reference_structures/metals/Sb.POSCAR

Oxide Reference Mode
~~~~~~~~~~~~~~~~~~~~

Used when ``reference_mode = "oxide"``.

oxides_ref (array of strings)
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

List of dopant oxide formulas (e.g. ``"Sb2O5"``).

oxides_dir (string)
^^^^^^^^^^^^^^^^^^^

Directory containing oxide POSCAR files.

gas_ref (string)
^^^^^^^^^^^^^^^^

Gas reference formula (typically ``"O2"``).

gas_dir (string)
^^^^^^^^^^^^^^^^

Directory containing gas POSCAR file.

oxygen_mode (string)
^^^^^^^^^^^^^^^^^^^^

Currently supports:

- ``"O-rich"``
- ``"O-poor"``

muO_shift_ev (float)
^^^^^^^^^^^^^^^^^^^^

Optional chemical potential shift applied to oxygen (eV).

---------------------------------------------------------------------

[structure]
-----------

Defines workflow I/O only.

outdir (string, default: "random_structures")
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

Directory where generated composition folders are written.

This directory becomes the root for:

- Step 01 outputs
- Step 02–06 per-composition subfolders

---------------------------------------------------------------------

[doping]
--------

Defines substitutional doping behavior.

host_species (string, required)
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

Element symbol of the host species to be substituted.

mode (string)
~~~~~~~~~~~~~

Defines doping mode:

- ``"explicit"`` — user provides exact compositions.
- ``"enumerate"`` — workflow constructs compositions combinatorially.

Explicit Mode
~~~~~~~~~~~~~

compositions (array of tables)
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

List of dictionaries:

::

   compositions = [
       { Sb = 5 },
       { Sb = 5, Zr = 5 }
   ]

Percentages are defined relative to host sites.

Enumerate Mode
~~~~~~~~~~~~~~

dopants (array of strings)
^^^^^^^^^^^^^^^^^^^^^^^^^^

Allowed dopant elements.

must_include (array of strings)
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

Dopants that must appear in each composition.

max_dopants_total (integer)
^^^^^^^^^^^^^^^^^^^^^^^^^^^

Maximum number of distinct dopants per structure.

allowed_totals (array of floats)
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

Allowed total dopant percentages.

levels (array of floats)
^^^^^^^^^^^^^^^^^^^^^^^^

Discrete concentration values per dopant.

---------------------------------------------------------------------

[generate]
----------

Step 01 — Structure generation.

This step generates one doped structure per composition by substituting
host atoms inside the **relaxed host supercell** produced by ``refs-build``.

Important
~~~~~~~~~

- ``refs-build`` must be executed first.
- The relaxed host supercell is loaded from:

  ``reference_structures/reference_energies.json``

- No supercell is constructed in this step.

Parameters
~~~~~~~~~~

seed_base (integer)
~~~~~~~~~~~~~~~~~~~

Base seed used for deterministic random substitution.

Each composition generates a stable hash-based seed.

poscar_order (array of strings, optional)
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

Defines element ordering in written POSCAR files.

If empty, pymatgen default ordering is used.

Example::

   poscar_order = ["Zr", "Ti", "Sb", "Sn", "O"]

clean_outdir (boolean, default: true)
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

If true, existing output directory is deleted before writing new structures.

Output
~~~~~~

For each composition:

::

   <outdir>/<composition_tag>/
       POSCAR
       metadata.json

---------------------------------------------------------------------

[scan]
------

Step 02 — Dopant configuration prescreening using M3GNet.

For each generated structure folder inside ``[structure].outdir``:

1. Generates doped configurations on the cation sublattice
2. Identifies symmetry-unique configurations
3. Evaluates single-point energies using M3GNet
4. Ranks configurations by energy
5. Keeps the lowest-energy ``topk`` candidates
6. Writes candidate folders and ranking files

Depending on the scan mode, configurations are obtained either by:

- exact symmetry-unique enumeration
- random symmetry-unique sampling

This step operates only on subfolders created in Step 01.

Parameters
~~~~~~~~~~

poscar_in (string, default: "POSCAR")
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

Filename inside each composition folder used as input.

topk (integer)
~~~~~~~~~~~~~~

Number of lowest-energy configurations retained.

symprec (float)
~~~~~~~~~~~~~~~

Tolerance used for symmetry detection in ``SpacegroupAnalyzer``.

mode (string, default: "auto")
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

Scan strategy.

Possible values:

- ``auto``  
  Automatically chooses exact enumeration for manageable problems and switches
  to sampling when the configuration space becomes too large.

- ``exact``  
  Forces full symmetry-unique enumeration of all configurations.

- ``sample``  
  Uses random symmetry-unique sampling instead of full enumeration.

max_enum (integer)
~~~~~~~~~~~~~~~~~~

Maximum allowed number of raw combinatorial configurations in exact mode.

If this limit is exceeded and ``mode = "auto"``, the scan automatically switches
to sampling mode.

max_unique (integer)
~~~~~~~~~~~~~~~~~~~~

Maximum allowed number of symmetry-unique configurations in exact mode.

Prevents excessive memory usage for very large configuration spaces.

nproc (integer)
~~~~~~~~~~~~~~~

Number of parallel worker processes used for M3GNet energy evaluation.

chunksize (integer)
~~~~~~~~~~~~~~~~~~~

Chunk size used in the multiprocessing pool.

anion_species (array of strings)
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

Species excluded from substitutional enumeration.
Typically contains oxygen:

::

   anion_species = ["O"]

host_species (from [doping])
~~~~~~~~~~~~~~~~~~~~~~~~~~~~

Used to define the cation sublattice.
Must match the host element defined in ``[doping]``.

skip_if_done (boolean, default: true)
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

If true, skip a structure folder if ``ranking_scan.csv`` already exists.

Sampling parameters
~~~~~~~~~~~~~~~~~~~

Used only when ``mode = "sample"`` or when ``mode = "auto"`` switches to sampling.

sample_budget (integer)
~~~~~~~~~~~~~~~~~~~~~~~

Maximum number of random sampling attempts.

sample_batch_size (integer)
~~~~~~~~~~~~~~~~~~~~~~~~~~~

Number of new symmetry-unique sampled configurations evaluated per batch.

sample_patience (integer)
~~~~~~~~~~~~~~~~~~~~~~~~~

Sampling stops after this many sampled configurations fail to improve the
current best candidate.

sample_seed (integer)
~~~~~~~~~~~~~~~~~~~~~

Random seed used for reproducible sampling.

sample_max_saved (integer)
~~~~~~~~~~~~~~~~~~~~~~~~~~

Maximum number of sampled canonical configurations stored to avoid duplicates.

Output
~~~~~~

For each composition folder:

::

   ranking_scan.csv
   scan_summary.txt
   candidate_001/01_scan/POSCAR
   candidate_001/01_scan/meta.json
   ...

Each candidate folder contains:

- symmetry-unique configuration
- single-point M3GNet energy
- dopant site signature
- scan metadata

[relax]
-------

Step 03 — Structural relaxation.

Relaxes the symmetry-selected candidates using the pretrained M3GNet Relaxer.
For each structure folder in ``[structure].outdir``, the candidates from
``candidate_*/01_scan/POSCAR`` are relaxed in parallel.

fmax (float)
~~~~~~~~~~~~

Maximum force convergence criterion (eV/Å).
Relaxation stops when the maximum atomic force falls below this threshold.

n_workers (integer)
~~~~~~~~~~~~~~~~~~~

Number of parallel relaxation workers (one candidate per worker process).

tf_threads (integer)
~~~~~~~~~~~~~~~~~~~~

TensorFlow thread count per worker.
Keep small (typically 1) when using multiple workers.

omp_threads (integer)
~~~~~~~~~~~~~~~~~~~~~

OpenMP thread count per worker.
Keep small to avoid CPU oversubscription.

skip_if_done (boolean)
~~~~~~~~~~~~~~~~~~~~~~

Skip an entire composition folder if ``ranking_relax.csv`` already exists.

skip_candidate_if_done (boolean)
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

Skip an individual candidate if
``candidate_*/02_relax/meta.json`` and ``POSCAR`` already exist.

Notes
~~~~~

- Species ordering in the relaxed ``POSCAR`` follows
  ``[generate].poscar_order``.
  If ``poscar_order`` is empty, the default pymatgen ordering is used.

---------------------------------------------------------------------

[filter]
--------

Step 04 — Candidate selection.

mode (string)
~~~~~~~~~~~~~

- ``"window"``
- ``"topn"``

window_meV (float)
~~~~~~~~~~~~~~~~~~

Energy window above the lowest relaxed energy (in meV).

Candidates with:

    E_relaxed <= E_min + window_meV

are retained.

A value of 0 keeps only the lowest-energy structure.

If no candidate satisfies the filtering criteria, the workflow raises an error.

max_candidates (integer)
~~~~~~~~~~~~~~~~~~~~~~~~

Number of candidates kept when mode = ``"topn"``.

skip_if_done (boolean)
~~~~~~~~~~~~~~~~~~~~~~

Skip filtering if output exists.

---------------------------------------------------------------------

[bandgap]
---------

Step 05 — Bandgap prediction using a local ALIGNN model.

Requires environment variable ``ALIGNN_MODEL_DIR`` pointing to
a local ALIGNN model directory.

skip_if_done (bool)
~~~~~~~~~~~~~~~~~~~

If true, previously computed bandgap results are reused.

Behavior:
- If ``candidate_*/03_band/meta.json`` already exists, the stored bandgap value is reused and prediction is skipped for that candidate.
- The summary CSV is rebuilt from existing metadata.
- This allows safe re-running of the workflow without recomputing already processed candidates.

If a candidate prediction fails:
- The error is recorded in ``candidate_*/03_band/meta.json``.
- The workflow continues with remaining candidates.
- Failed candidates appear in the summary CSV with ``NaN`` bandgap.

cutoff (float)
~~~~~~~~~~~~~~

Neighbor cutoff radius (Å) used to construct the atomic graph
for ALIGNN inference. Must be > 0.

max_neighbors (int)
~~~~~~~~~~~~~~~~~~~

Maximum number of neighbors retained per atom when building the graph.
Must be > 0.

---------------------------------------------------------------------

[formation]
-----------

Step 06 — Formation energy calculation.

Formation energies are computed using the chemical potentials written by
``refs-build`` in ``reference_structures/reference_energies.json``.

The reference scheme (metal or oxide) is automatically determined from
``[references].reference_mode`` and no additional user input is required here.

skip_if_done (boolean)
~~~~~~~~~~~~~~~~~~~~~~

If true, skip formation calculation if ``formation_energies.csv`` already exists
in a composition folder.

normalize (string)
~~~~~~~~~~~~~~~~~~

Defines how the reported formation energy is normalized:

- ``"total"`` — total formation energy (eV)
- ``"per_dopant"`` — eV per substituted dopant atom
- ``"per_host"`` — eV per atom in the supercell

Formation energies require:

- Successful execution of ``refs-build``
- ``reference_structures/reference_energies.json``
- Relaxed candidate structures in ``candidate_*/02_relax/``

Notes
~~~~~

- The same substitution formula is used for both reference schemes.
- In ``metal`` mode, elemental metal references define the chemical potentials.
- In ``oxide`` mode, chemical potentials are derived from oxide references
  and the chosen oxygen condition (``O-rich`` or ``O-poor``).
- The selected reference mode is stored in
  ``candidate_*/04_formation/meta.json`` for reproducibility.

---------------------------------------------------------------------

[database]
----------

Step 07 — Final database collection.

skip_if_done (boolean, default: true)
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

If true, do not overwrite existing ``results_database.csv``.

---------------------------------------------------------------------

Design Principles
-----------------

- All stages are deterministic given fixed input.
- All randomness is seed-controlled.
- Each stage can be skipped independently.
- Each stage writes metadata for full reproducibility.
- The relaxed host supercell is constructed once and reused.