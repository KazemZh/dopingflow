3. Candidate Relaxation
=======================

Implementation
--------------

This stage is implemented in:

::

   src/dopingflow/relax.py

The public entry point is:

::

   run_relax(...)


Purpose
-------

This stage performs **full structural relaxation** of the low-energy candidates
selected during the scanning stage.

For each candidate structure:

- atomic positions are relaxed
- lattice vectors are relaxed
- the final total energy is extracted
- relaxed structures are written to disk
- a relaxation ranking is generated

This stage refines the single-point screening results by allowing full geometry
optimization.


Inputs
------

This stage uses settings from the following sections of ``input.toml``:

- ``[structure]``: provides the output directory containing structure folders.
- ``[generate]``: provides the species ordering used when writing POSCAR files.
- ``[relax]``: controls convergence tolerance, parallelism, and skipping logic.

The relaxation stage processes:

::

   <outdir>/<structure_folder>/candidate_*/01_scan/POSCAR


Method Summary
--------------

For each structure folder inside ``[structure].outdir``:

1. Detect all ``candidate_*`` subfolders.
2. For each candidate:
   a. Read ``01_scan/POSCAR``.
   b. Perform full structural relaxation.
   c. Write relaxed structure to ``02_relax/POSCAR``.
   d. Write relaxation metadata.
3. Rank candidates by relaxed total energy.
4. Write ``ranking_relax.csv`` per structure folder.


Relaxation Model
----------------

Structural relaxation is performed using:

- Interatomic potential: **M3GNet (pretrained)**
- Optimizer: internal Relaxer (FIRE-based)
- Convergence criterion: maximum force below ``[relax].fmax``

Both:

- atomic coordinates
- lattice parameters

are allowed to relax.


Parallel Execution
------------------

Relaxation is parallelized **over candidates** using multiprocessing.

Each worker process:

- initializes its own M3GNet Relaxer instance
- sets thread limits to control TensorFlow and OpenMP usage
- runs independently

The number of parallel workers is controlled by:

::

   [relax]
   n_workers = ...

Additional thread control parameters:

::

   tf_threads
   omp_threads

These settings allow tuning CPU usage on workstations or HPC systems.


Ranking Logic
-------------

After all candidates are processed within a structure folder:

- Only successfully relaxed candidates (status ``ok``) are ranked.
- Ranking is based on ascending relaxed total energy.

The resulting CSV file:

::

   ranking_relax.csv

contains:

- candidate name
- relaxed rank
- relaxed energy
- original single-point rank
- original single-point energy
- signature
- status
- walltime
- error message (if any)


Reproducibility and Skipping
----------------------------

Two levels of skipping are supported:

Folder-level skipping
~~~~~~~~~~~~~~~~~~~~~

If:

::

   [relax].skip_if_done = true

and ``ranking_relax.csv`` already exists in a structure folder,
the entire folder is skipped.

Candidate-level skipping
~~~~~~~~~~~~~~~~~~~~~~~~

If:

::

   [relax].skip_candidate_if_done = true

and:

::

   candidate_*/02_relax/meta.json
   candidate_*/02_relax/POSCAR

already exist, that individual candidate is skipped.

This enables safe restart of interrupted runs.


Outputs
-------

For each candidate:

::

   candidate_###/
       01_scan/
       02_relax/
           POSCAR
           meta.json

The relaxation metadata includes:

- relaxed total energy
- walltime
- species counts
- convergence target (fmax)
- link to original scan metadata

For each structure folder:

::

   ranking_relax.csv

which ranks relaxed candidates by energy.


Error Handling
--------------

If relaxation fails:

- an error is written to ``02_relax/meta.json``
- the candidate is marked with status ``fail``
- ranking proceeds with remaining successful candidates

Failures do not abort the entire structure folder.


Notes and Limitations
---------------------

- Relaxation uses a pretrained ML potential (not DFT-level relaxation).
- No charge states or external fields are included.
- No temperature or vibrational effects are considered.
- Energies are total energies from the ML model.
- Convergence depends on the chosen ``fmax`` threshold.
- Parallel performance depends on CPU availability and thread configuration.
