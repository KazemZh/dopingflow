2. Symmetry-Reduced Energy Pre-screening (Scanning)
===================================================

Implementation
--------------

This stage is implemented in:

::

   src/dopingflow/scan.py

The public entry point is:

::

   run_scan(...)


Purpose
-------

This stage performs an efficient **energy pre-screening** of dopant arrangements
by enumerating **symmetry-unique** configurations on a selected sublattice and
evaluating their **single-point energies** using a machine-learned interatomic
potential.

For each generated structure folder, the method:

- infers the substitutional sublattice and dopant counts from the input structure
- generates dopant configurations either by **exact enumeration** or **random sampling**
- removes symmetry-equivalent configurations
- predicts a single-point energy for each configuration using M3GNet
- keeps the **top-k lowest-energy** candidates for downstream relaxation


Inputs
------

This stage uses settings from the following sections of ``input.toml``:

- ``[structure]``: provides the output directory containing generated structures.
- ``[doping]``: provides the host species definition.
- ``[generate]``: provides the species ordering used when writing POSCAR files.
- ``[scan]``: controls enumeration limits, symmetry tolerance, parallelism, and selection.


Method Summary
--------------

For each structure subdirectory in ``[structure].outdir``:

1. Read the structure file specified by ``[scan].poscar_in``.
2. Identify the **enumeration sublattice** (all non-anion sites) and infer:
   - host species count on the sublattice
   - dopant species counts on the sublattice
3. Estimate the total number of raw (non-symmetry-reduced) configurations.
4. Construct a parent structure and compute symmetry operations acting on the sublattice.
5. Decide the scan strategy according to ``[scan].mode``:

   - ``exact``: perform full symmetry-unique enumeration
   - ``sample``: generate configurations using random symmetry-unique sampling
   - ``auto``: use enumeration when feasible, otherwise switch to sampling

6. Generate configurations:

   - enumeration: iterate over all dopant permutations
   - sampling: randomly generate dopant arrangements and filter duplicates

7. Reduce configurations to symmetry-unique representations.

8. Evaluate single-point energies in parallel using M3GNet.

9. Select the lowest-energy ``topk`` candidates and write them to ``candidate_*/01_scan``.
10. Write a CSV ranking file and a human-readable summary.


Enumeration Sublattice Definition
---------------------------------

The enumerated sublattice is inferred from the input structure using the rule:

- **anion sites** are excluded (defined by ``[scan].anion_species``)
- **all remaining sites** form the enumeration sublattice

This design makes the method general across crystals where doping occurs on a
cation (or non-anion) sublattice.

The dopant counts are inferred directly from the input POSCAR by counting
species on the enumeration sublattice. The host species is provided by
``[doping].host_species``.


Combinatorics and Safety Limits
-------------------------------

Raw configuration count
~~~~~~~~~~~~~~~~~~~~~~~

Given a sublattice with :math:`N` sites and fixed dopant counts, the number of
raw configurations (before symmetry reduction) grows combinatorially.

To avoid runaway enumeration, the workflow:

- estimates the raw count and enforces ``[scan].max_enum``
- enforces a hard limit on symmetry-unique configurations via ``[scan].max_unique``

If ``mode = "auto"`` and the estimated configuration count exceeds
``[scan].max_enum``, the scan automatically switches to sampling mode.

If ``mode = "exact"``, exceeding these limits stops the scan to prevent
infeasible enumeration.

This protects against infeasible compositions and/or large supercells.


Maximum number of dopant species per scan
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

The current symmetry-unique enumerator is implemented explicitly for up to
**three distinct dopant species** on the enumerated sublattice.

If the inferred dopant set contains more than three species, the scan raises an
error.

This limit applies to the scanning enumeration procedure (not necessarily to
other stages).


Symmetry Reduction
------------------

To avoid evaluating symmetry-equivalent configurations, the scan:

1. Constructs a **parent** structure where all sites of the enumeration sublattice
   are assigned to the host species.
2. Uses a space-group analysis (with tolerance ``[scan].symprec``) to obtain
   symmetry operations.
3. Converts symmetry operations into **permutations of sublattice indices**.
4. For each dopant labeling, computes a canonical representation under these permutations.
5. Keeps only labelings with unique canonical keys.

This yields the set of symmetry-unique dopant configurations.

Sampling Mode
-------------

When ``mode = "sample"`` (or when ``mode = "auto"`` switches to sampling),
the workflow generates configurations by random sampling instead of full
combinatorial enumeration.

The algorithm:

1. randomly assigns dopants to sublattice sites
2. converts the labeling to a canonical symmetry representation
3. discards duplicates already encountered
4. accumulates unique configurations until a batch is formed
5. evaluates the batch using M3GNet

Sampling is controlled by the parameters:

- ``sample_budget`` — maximum number of sampling attempts
- ``sample_batch_size`` — number of unique structures evaluated per batch
- ``sample_patience`` — early stopping when no better structures are found
- ``sample_seed`` — random seed for reproducibility
- ``sample_max_saved`` — maximum stored canonical keys to avoid duplicates

Sampling allows the scan to explore large configuration spaces that would
otherwise be infeasible to enumerate.


Energy Model and Parallel Evaluation
------------------------------------

Each symmetry-unique configuration is evaluated using a **single-point**
energy prediction with M3GNet.

Key points:

- the structure is not relaxed in this stage
- only the energy is predicted (fast screening)
- energies are computed in parallel using multiprocessing

To improve robustness when using TensorFlow-based models, worker processes are
created using the ``spawn`` start method.


Selection of Top-k Candidates
-----------------------------

The workflow keeps the ``[scan].topk`` configurations with the **lowest predicted**
single-point energies and writes each to:

::

   <structure_dir>/candidate_###/01_scan/POSCAR
   <structure_dir>/candidate_###/01_scan/meta.json

A CSV ranking file is written to the structure directory:

::

   <structure_dir>/ranking_scan.csv

The ranking CSV contains:

- candidate name
- rank (single-point)
- predicted single-point energy
- a short dopant-position signature


Reproducibility
---------------

For a fixed input structure and scan settings:

- enumeration mode produces deterministic results
- sampling mode is stochastic but reproducible when ``sample_seed`` is fixed
- inferred dopant counts
- symmetry analysis (controlled by ``symprec``)
- enumeration ordering
- the ML model used for prediction


Parallel evaluation does not affect the final ranking, since candidates are
selected based on energy values.


Outputs
-------

For each processed structure folder:

- ``candidate_*/01_scan/POSCAR``: POSCAR files of selected low-energy candidates
- ``candidate_*/01_scan/meta.json``: metadata describing scan settings and counts
- ``ranking_scan.csv``: ranked list of candidates with predicted energies
- ``scan_summary.txt``: human-readable summary of the scan


Notes and Limitations
---------------------

- This stage performs **single-point ML energy screening only**; it does not relax structures.
- Symmetry reduction depends on the tolerance ``symprec``; too large values may merge distinct
  configurations, too small values may reduce symmetry detection.
- Enumeration can become infeasible for large sublattices and/or high dopant counts;
  limits ``max_enum`` and ``max_unique`` are enforced to prevent runaway runtime/memory.
- The current label enumerator supports up to **three distinct dopant species**.
- Predicted single-point energies are surrogate ML values and should be interpreted
  as a ranking heuristic rather than absolute thermodynamic energies.
- Sampling mode does not guarantee discovery of the global minimum but
  efficiently identifies low-energy candidates for subsequent relaxation.