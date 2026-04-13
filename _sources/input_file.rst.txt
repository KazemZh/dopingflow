.. _input_file_spec:

Input File Specification (input.toml)
=====================================

The workflow is fully controlled through a single TOML configuration file.

Each stage reads only the parameters relevant to it.

Execution Model
---------------

Each stage defines both:

- **what it does** (physics / workflow logic)
- **how it runs** (CPU / GPU / parallelization)

There is **no global hardware section**. Instead:

- ``[scan]`` controls screening
- ``[relax]`` controls structural relaxation
- ``[bandgap]`` controls band gap prediction

Each stage independently defines:

- ``device`` (cpu or cuda)
- ``gpu_id`` (for GPU execution)

This design gives full flexibility and avoids cross-stage conflicts.

Execution Behavior
------------------

+-----------+-------------------+--------------------------+
| Stage     | Device            | Strategy                 |
+===========+===================+==========================+
| scan      | CPU               | multiprocessing          |
+-----------+-------------------+--------------------------+
| relax     | CPU / GPU         | workers or single GPU    |
+-----------+-------------------+--------------------------+
| bandgap   | CPU               | multiprocessing          |
+-----------+-------------------+--------------------------+
| bandgap   | GPU               | batched inference        |
+-----------+-------------------+--------------------------+

This design allows each stage to control its own parallelization strategy
while sharing a consistent hardware configuration.

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

Step 02 — Dopant configuration prescreening using machine-learning interatomic potentials.

For each generated structure folder inside ``[structure].outdir``:

1. Generates doped configurations on the cation sublattice
2. Identifies symmetry-unique configurations
3. Evaluates single-point energies using the selected ML backend
4. Ranks configurations by energy
5. Keeps the lowest-energy ``topk`` candidates
6. Writes candidate folders and ranking files

Depending on the scan mode, configurations are obtained either by:

- exact symmetry-unique enumeration
- random symmetry-unique sampling

This step operates only on subfolders created in Step 01.

Parameters
~~~~~~~~~~

backend (string, default: "m3gnet")
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

Selects the ML model used for energy prediction.

Available options:

- ``"m3gnet"`` — TensorFlow-based universal interatomic potential
- ``"uma"`` — FAIR-Chem universal model (requires Hugging Face access)
- ``"mace"`` — MACE foundation models (Materials Project / OMAT / MPA)
- ``"grace"`` — GRACE graph neural network models (if installed)

Each backend has its own supported models and execution characteristics.

model (string, default: "default")
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

Specifies the pretrained model variant used by the selected backend.

Behavior depends on backend:

- ``m3gnet``:
  - ``"default"`` → loads the standard M3GNet universal model

- ``uma``:
  - ``"uma-s-1p2"``
  - ``"uma-s-1p1"``
  - ``"uma-m-1p1"``

- ``mace``:
  - ``"small"``
  - ``"medium"``
  - ``"large"``
  - ``"small-mpa-0"``
  - ``"medium-mpa-0"``
  - ``"large-mpa-0"``
  - ``"small-omat-0"``
  - ``"medium-omat-0"``

- ``grace``:
  - ``GRACE-1L-OMAT``
  - ``GRACE-1L-OMAT-M-base``
  - ``GRACE-1L-OMAT-M``
  - ``GRACE-1L-OMAT-L-base``
  - ``GRACE-1L-OMAT-L``
  - ``GRACE-2L-OMAT``
  - ``GRACE-2L-OMAT-M-base``
  - ``GRACE-2L-OMAT-M``
  - ``GRACE-2L-OMAT-L-base``
  - ``GRACE-2L-OMAT-L``
  - ``GRACE-1L-OAM``
  - ``GRACE-1L-OAM-M``
  - ``GRACE-1L-OAM-L``
  - ``GRACE-2L-OAM``
  - ``GRACE-2L-OAM-M``
  - ``GRACE-2L-OAM-L``
  - ``GRACE-1L-SMAX-L``
  - ``GRACE-1L-SMAX-OMAT-L``
  - ``GRACE-2L-SMAX-M``
  - ``GRACE-2L-SMAX-L``
  - ``GRACE-2L-SMAX-OMAT-M``
  - ``GRACE-2L-SMAX-OMAT-L``

task (string, default: "")
~~~~~~~~~~~~~~~~~~~~~~~~~~

Optional task specification (used only for certain backends).

- ``uma`` requires a task:
  - ``"omat"``
  - ``"oc20"``
  - ``"oc22"``
  - ``"oc25"``
  - ``"omol"``
  - ``"odac"``
  - ``"omc"``

- ``m3gnet``, ``mace``, ``grace``:
  - Not used → keep empty (``""``)

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

device (string, default: "cpu")
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

Execution device:

- ``"cpu"``
- ``"cuda"``

Behavior depends on backend:

- ``m3gnet``:
  - GPU mode uses a single worker (TensorFlow limitation)

- ``uma``:
  - GPU supported via PyTorch

- ``mace``:
  - GPU supported and recommended for performance

- ``grace``:
  - GPU support depends on model implementation

n_workers (integer)
~~~~~~~~~~~~~~~~~~~

Number of parallel worker processes.

- Used only when ``device = "cpu"``
- Some backends may internally limit parallelism:

  - ``m3gnet`` (GPU): forces single worker
  - ``mace``: typically runs efficiently in single-process mode
  
- Ignored when ``device = "cuda"`` (GPU mode uses a single worker)

chunksize (integer)
~~~~~~~~~~~~~~~~~~~

Chunk size used in the multiprocessing pool.

Relevant only when ``device = "cpu"``.

gpu_id (integer, default: 0)
~~~~~~~~~~~~~~~~~~~~~~~~~~~~

GPU index used when ``device = "cuda"``.

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

Used when:

- ``mode = "sample"``
- or ``mode = "auto"`` selects sampling for large configuration spaces

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

Notes
~~~~~

- If ``device = "cuda"``, scan runs on a single GPU and ``n_workers`` is ignored.
- If ``device = "cpu"``, parallelization is controlled via ``n_workers`` and ``chunksize``.
- GPU mode is recommended for faster single-structure evaluation, while CPU mode scales better across many configurations.
- The scan backend can be selected via ``backend``.
- Different backends have different accuracy/speed trade-offs:

  - ``m3gnet``: stable, general-purpose
  - ``uma``: high-quality FAIR-Chem models (requires authentication)
  - ``mace``: fast and scalable foundation models
  - ``grace``: advanced GNN-based models (experimental)

- For large configuration spaces, sampling mode is recommended.

- GPU acceleration is backend-dependent and may not always scale with multiprocessing.

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
- single-point energy (from selected backend)
- dopant site signature
- scan metadata

The following metadata is recorded:

- backend
- model
- task
- energy_sp_eV
- configuration details

[relax]
-------

Step 03 — Structural relaxation using machine-learning interatomic potentials.

This stage relaxes the low-energy candidates selected in Step 02.
For each structure folder in ``[structure].outdir``, the workflow reads:

::

   candidate_*/01_scan/POSCAR

and writes relaxed structures and metadata to:

::

   candidate_*/02_relax/

Relaxation is performed using the selected ML backend together with an ASE optimizer.

Supported backends:

- ``"m3gnet"``
- ``"uma"``
- ``"mace"``
- ``"grace"``

Supported optimizers:

- ``"bfgs"``
- ``"lbfgs"``
- ``"fire"``
- ``"mdmin"``
- ``"quasinewton"``

Relaxation stops when either:

- the maximum atomic force drops below ``fmax``
- the number of optimizer steps reaches ``max_steps``

Parameters
~~~~~~~~~~

backend (string, default: "m3gnet")
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

Selects the ML backend used for structural relaxation.

Available options:

- ``"m3gnet"`` — TensorFlow-based universal interatomic potential
- ``"uma"`` — FAIR-Chem universal model (requires Hugging Face access)
- ``"mace"`` — MACE foundation models
- ``"grace"`` — GRACE graph neural network models (if installed)

model (string, default: "default")
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

Specifies the pretrained model variant used by the selected relaxation backend.

Behavior depends on backend:

- ``m3gnet``:
  - ``"default"`` → loads the standard pretrained M3GNet model

- ``uma``:
  - ``"uma-s-1p2"``
  - ``"uma-s-1p1"``
  - ``"uma-m-1p1"``

- ``mace``:
  - ``"small"``
  - ``"medium"``
  - ``"large"``
  - ``"small-mpa-0"``
  - ``"medium-mpa-0"``
  - ``"large-mpa-0"``
  - ``"small-omat-0"``
  - ``"medium-omat-0"``

- ``grace``:
  - ``GRACE-1L-OMAT``
  - ``GRACE-1L-OMAT-M-base``
  - ``GRACE-1L-OMAT-M``
  - ``GRACE-1L-OMAT-L-base``
  - ``GRACE-1L-OMAT-L``
  - ``GRACE-2L-OMAT``
  - ``GRACE-2L-OMAT-M-base``
  - ``GRACE-2L-OMAT-M``
  - ``GRACE-2L-OMAT-L-base``
  - ``GRACE-2L-OMAT-L``
  - ``GRACE-1L-OAM``
  - ``GRACE-1L-OAM-M``
  - ``GRACE-1L-OAM-L``
  - ``GRACE-2L-OAM``
  - ``GRACE-2L-OAM-M``
  - ``GRACE-2L-OAM-L``
  - ``GRACE-1L-SMAX-L``
  - ``GRACE-1L-SMAX-OMAT-L``
  - ``GRACE-2L-SMAX-M``
  - ``GRACE-2L-SMAX-L``
  - ``GRACE-2L-SMAX-OMAT-M``
  - ``GRACE-2L-SMAX-OMAT-L``

task (string, default: "")
~~~~~~~~~~~~~~~~~~~~~~~~~~

Optional task specification used only for ``uma``.

Allowed values for ``uma``:

- ``"omat"``
- ``"oc20"``
- ``"oc22"``
- ``"oc25"``
- ``"omol"``
- ``"odac"``
- ``"omc"``

For ``m3gnet``, ``mace``, and ``grace``, this parameter is ignored and should be left empty.

optimizer (string, default: "bfgs")
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

ASE optimizer used during relaxation.

Available options:

- ``"bfgs"``
- ``"lbfgs"``
- ``"fire"``
- ``"mdmin"``
- ``"quasinewton"``

fmax (float)
~~~~~~~~~~~~

Maximum force convergence criterion (eV/Å).

Relaxation is considered converged when the maximum atomic force falls below this threshold.

max_steps (integer, default: 300)
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

Maximum number of optimizer steps.

If this limit is reached before the force threshold is satisfied, the relaxation stops and the final structure is still written to disk.

device (string, default: "cpu")
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

Execution device:

- ``"cpu"``
- ``"cuda"``

Behavior depends on backend and runtime environment.
GPU execution uses a single effective worker.

gpu_id (integer, default: 0)
~~~~~~~~~~~~~~~~~~~~~~~~~~~~

GPU index used when ``device = "cuda"``.

n_workers (integer)
~~~~~~~~~~~~~~~~~~~

Number of parallel relaxation workers (one candidate per worker process).

Relevant mainly when ``device = "cpu"``.

tf_threads (integer)
~~~~~~~~~~~~~~~~~~~~

TensorFlow thread count per worker.

Mainly relevant for the ``m3gnet`` backend.
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

Skip an individual candidate if both of the following already exist:

::

   candidate_*/02_relax/meta.json
   candidate_*/02_relax/POSCAR

Notes
~~~~~

- If ``device = "cuda"``, relaxation uses a single effective worker and ``n_workers`` is ignored.
- If ``device = "cpu"``, parallelization is controlled via ``n_workers``.
- The relaxed ``POSCAR`` follows the species ordering defined by ``[generate].poscar_order``.
- If ``poscar_order`` is empty, the default pymatgen ordering is used.

Output
~~~~~~

For each candidate:

::

   candidate_###/02_relax/POSCAR
   candidate_###/02_relax/meta.json

For each structure folder:

::

   ranking_relax.csv

The metadata records:

- backend
- model
- task
- optimizer
- relaxed energy
- convergence target
- final maximum force
- optimizer step count
- convergence status
- walltime
- link to the original scan metadata

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

device (string, default: "cpu")
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

Execution device:

- ``"cpu"``
- ``"cuda"``

gpu_id (integer, default: 0)
~~~~~~~~~~~~~~~~~~~~~~~~~~~~

GPU index used when ``device = "cuda"``.

batch_size (integer, default: 32)
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

Batch size used for GPU inference.

n_workers (integer)
~~~~~~~~~~~~~~~~~~~

Number of CPU workers used for parallel bandgap prediction.

Notes
~~~~~

- CPU mode → multiprocessing over structures using ``n_workers``
- GPU mode → batched inference using ``batch_size``
- ``n_workers`` is ignored when using GPU

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