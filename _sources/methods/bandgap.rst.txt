5. Bandgap Prediction
=====================

Implementation
--------------

This stage is implemented in:

::

   src/dopingflow/bandgap.py

The public entry point is:

::

   run_bandgap(...)


Purpose
-------

This stage predicts the **electronic band gap** of relaxed candidate structures
using a locally available **ALIGNN** model.

Band gaps are evaluated for candidates produced by the previous steps
(typically after relaxation, and optionally after filtering) and written to:

- a per-folder summary CSV
- a per-candidate metadata record

This is a machine-learning inference step; no electronic-structure calculation is
performed here.


Inputs
------

This stage uses settings from the following sections of ``input.toml``:

- ``[structure]``: provides the output directory containing structure folders.
- ``[bandgap]``: defines graph-construction parameters and skipping behavior.

In addition, this stage requires an environment variable pointing to a local
ALIGNN model directory:

::

   ALIGNN_MODEL_DIR=/path/to/your/alignn_model_folder


Method Summary
--------------

For each structure folder inside ``[structure].outdir``:

1. Determine the list of candidate structures to evaluate:

   a. If ``selected_candidates.txt`` exists, only those candidates are used.
   b. Otherwise, all ``candidate_*/02_relax/POSCAR`` files are used.

2. For each selected candidate:

   a. Read the relaxed ``POSCAR`` (from Step 03).
   b. Convert the structure to a graph representation (DGL graph) using a
      neighbor cutoff and maximum neighbor count.
   c. Run a forward pass of the ALIGNN model to obtain the predicted band gap.
   d. Write a per-candidate metadata file.

3. Write a per-folder summary CSV listing band gaps for all evaluated candidates.


Selection of Candidates
-----------------------

This stage supports two selection modes:

- **Filtered selection (recommended)**:
  If ``selected_candidates.txt`` exists in the folder, the band gap is computed
  only for those candidates. This integrates naturally with Step 04 filtering.

- **Fallback selection**:
  If no selection list is present, all relaxed candidates are used.

The relaxed structures are always taken from:

::

   candidate_*/02_relax/POSCAR


Model and Graph Construction
----------------------------

ALIGNN model loading
~~~~~~~~~~~~~~~~~~~~

The model is loaded from a local directory under ``ALIGNN_MODEL_DIR``.
The code searches for a folder containing a ``config.json`` and a checkpoint
file (e.g. ``checkpoint_*.pt`` or ``*.pt``), then loads:

- the model configuration (from ``config.json``)
- the checkpoint weights

Graph parameters
~~~~~~~~~~~~~~~~

For each structure, a graph is constructed using:

- ``[bandgap].cutoff``: neighbor cutoff distance
- ``[bandgap].max_neighbors``: maximum neighbors per atom

These parameters control how local environments are encoded for inference.

The DGL backend is set to PyTorch in this stage (via ``DGLBACKEND=pytorch``),
which must be configured before importing DGL/ALIGNN.


Outputs
-------

Per-folder summary
~~~~~~~~~~~~~~~~~~

For each structure folder, this stage writes:

::

   bandgap_alignn_summary.csv

This file contains:

- ``candidate``: candidate directory name
- ``bandgap_eV_ALIGNN_MBJ``: predicted band gap (eV)

The CSV is sorted by band gap (ascending) for convenience.

Per-candidate metadata
~~~~~~~~~~~~~~~~~~~~~~

For each evaluated candidate, this stage writes:

::

   candidate_XXX/03_band/meta.json

This metadata includes:

- predicted band gap value
- model provenance (model directory, config path, checkpoint path)
- input structure path
- structure size and composition
- graph parameters used for inference
- walltime for prediction


Reproducibility and Skipping
----------------------------

If:

::

   [bandgap].skip_if_done = true

and ``bandgap_alignn_summary.csv`` already exists for a folder, that folder is
skipped.

Given the same relaxed POSCARs, model checkpoint, and graph parameters, this stage
is deterministic.


Notes and Limitations
---------------------

- Predicted band gaps depend on the chosen ALIGNN model and its training domain.
- Band gaps are ML predictions and should not be interpreted as DFT-quality values
  unless the chosen model has been validated for the target material class.
- This stage assumes candidate structures are already relaxed; it does not perform
  geometry optimization.
