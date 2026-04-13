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

This stage performs **structural relaxation** of the low-energy candidates
selected during the scanning stage.

For each candidate structure:

- atomic positions are relaxed
- lattice vectors are optimized together with atomic positions
- the final total energy is extracted
- relaxed structures are written to disk
- a relaxation ranking is generated

This stage refines the single-point screening results by allowing full geometry
optimization with the selected machine-learning interatomic potential backend.


Inputs
------

This stage uses settings from the following sections of ``input.toml``:

- ``[structure]``: provides the output directory containing structure folders
- ``[generate]``: provides the species ordering used when writing POSCAR files
- ``[relax]``: controls backend, model, optimizer, convergence, execution mode, and skipping logic

The relaxation stage processes:

::

   <outdir>/<structure_folder>/candidate_*/01_scan/POSCAR


Method Summary
--------------

For each structure folder inside ``[structure].outdir``:

1. Detect all ``candidate_*`` subfolders.
2. For each candidate:
   a. Read ``01_scan/POSCAR``.
   b. Initialize the selected ML backend.
   c. Relax the structure with the selected ASE optimizer.
   d. Write the relaxed structure to ``02_relax/POSCAR``.
   e. Write relaxation metadata to ``02_relax/meta.json``.
3. Rank candidates by relaxed total energy.
4. Write ``ranking_relax.csv`` per structure folder.


Relaxation Model
----------------

Structural relaxation is performed using a selectable machine-learning backend.

Supported backends:

- **M3GNet**
- **UMA**
- **MACE**
- **GRACE**

The backend is used as an ASE calculator, and relaxation is carried out using
an ASE optimizer.

Supported optimizers:

- **BFGS**
- **LBFGS**
- **FIRE**
- **MDMin**
- **QuasiNewton**

The stopping criteria are:

- maximum atomic force below ``[relax].fmax``
- or optimizer step count reaching ``[relax].max_steps``

Both:

- atomic coordinates
- lattice parameters

are allowed to relax.


Backend-Specific Notes
----------------------

M3GNet
~~~~~~

- Uses the pretrained M3GNet model through an ASE calculator
- Supports CPU and CUDA execution
- On GPU, the workflow uses a single effective worker for safe TensorFlow usage

UMA
~~~

- Uses FAIR-Chem pretrained models
- Requires both a selected ``model`` and a ``task``
- Supports CPU and GPU execution depending on the environment

MACE
~~~~

- Uses pretrained MACE foundation models
- Integrated through the ASE calculator interface
- Supports CPU and GPU execution

GRACE
~~~~~

- Uses GRACE foundation models through the tensorpotential interface
- Integrated through the ASE calculator interface
- Availability depends on the installed GRACE/tensorpotential environment


Parallel Execution
------------------

Relaxation is parallelized **over candidates** using multiprocessing.

Each worker process:

- initializes its own backend model or calculator
- applies thread limits where relevant
- relaxes one candidate at a time

The number of parallel workers is controlled by:

::

   [relax]
   n_workers = ...

Additional thread control parameters:

::

   tf_threads
   omp_threads

These settings are mainly relevant in CPU mode.
When ``device = "cuda"``, the workflow uses a single effective worker.


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
- convergence status
- final maximum force
- optimizer step count
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

- backend
- model
- task
- optimizer
- relaxed total energy
- walltime
- species counts
- convergence target (fmax)
- maximum allowed optimizer steps
- actual optimizer step count
- final maximum force
- convergence status
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

- Relaxation uses pretrained ML interatomic potentials rather than DFT.
- No charge states or external electric fields are included.
- No temperature or vibrational effects are considered.
- Energies are total energies predicted by the selected ML backend.
- Convergence depends on both the chosen ``fmax`` threshold and ``max_steps``.
- Parallel performance depends on backend, hardware, and thread configuration.