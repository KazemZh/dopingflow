1. Structure Generation
========================

Implementation
--------------

This stage is implemented in:

::

   src/dopingflow/generate.py

The public entry point is:

::

   run_generate(...)

Purpose
-------

This stage generates an initial set of doped structures starting from a pristine
unit cell. The output is a directory of subfolders, each containing a VASP
``POSCAR`` and a small metadata file describing the effective composition and
the random seed used for site selection.

The generation step supports two workflows:

- **Explicit compositions**: the user provides exact dopant percentages.
- **Enumerated compositions**: the workflow constructs a systematic set of
  dopant combinations and doping levels under user-defined constraints.


Inputs
------

This stage uses settings from three sections of ``input.toml``:

- ``[structure]``: provides the pristine structure file and the supercell size.
- ``[doping]``: defines the doping mode and composition rules.
- ``[generate]``: controls structure-writing details and reproducible randomness.


Method Summary
--------------

1. Read the pristine structure from ``[structure].base_poscar``.
2. Build the supercell using ``[structure].supercell``.
3. Identify all sites matching the substitution host species
   (``[doping].host_species``).
4. For each requested composition:

   a. Convert requested dopant percentages to integer substitution counts
      by rounding.
   b. Randomly choose host sites to substitute, using a deterministic seed.
   c. Optionally reorder species in the written POSCAR.
   d. Write ``POSCAR`` and ``metadata.json`` to an output subdirectory.

The output directory is ``[structure].outdir`` (default: ``random_structures``).


Composition Handling
--------------------

Requested vs effective composition
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

Doping levels are provided as percentages *relative to the number of host sites*
in the generated supercell. Since the number of sites is discrete, requested
percentages are converted to integer substitution counts by rounding.

The workflow therefore distinguishes:

- **requested composition**: the percentages from the input
- **effective composition**: the percentages implied by the rounded integer counts

If rounding changes any dopant level, warnings are reported and the **effective**
composition is stored in the metadata.

A basic consistency check is applied:

- the total number of substituted atoms may not exceed the number of host sites
- the total requested dopant percentage may not exceed 100%


Explicit mode
~~~~~~~~~~~~~

In explicit mode, each composition is provided directly by the user as a mapping
``element -> percent``. Each composition produces exactly one structure.

This mode is recommended when:

- specific compositions are already known or desired
- only a small number of target compositions are needed


Enumerate mode
~~~~~~~~~~~~~~

In enumerate mode, the workflow generates composition dictionaries automatically
from:

- a list of possible dopant species
- an optional list of dopants that must appear
- a set of allowed total dopant levels
- a set of discrete per-dopant levels

The workflow enumerates combinations of distinct dopants of size:

.. math::

   k \in \{1, \dots, k_{\max}\}

where :math:`k_{\max}` is controlled by the input parameter
``[doping].max_dopants_total``.

**Interpretation**:

- you may provide many candidate dopant elements in the input
- but each generated structure contains at most ``max_dopants_total`` distinct dopant species
  *when using enumerate mode*

(Explicit mode is not inherently limited unless the user adopts the same constraint.)


Reproducible Random Substitution
--------------------------------

The actual substitutional sites are selected randomly, but deterministically:

- a **composition tag** is constructed from the effective composition
- a stable seed is derived from the tag and a base seed
  (``[generate].seed_base``)

This ensures that rerunning the workflow with unchanged input produces identical
structures, while still providing randomized site selection.


Directory Naming and Collision Handling
---------------------------------------

Each generated structure is written to:

::

   <outdir>/<composition_tag>/POSCAR
   <outdir>/<composition_tag>/metadata.json

The folder tag is constructed from the **effective** composition. If two different
requested compositions round to the same effective composition, a suffix is added:

::

   <tag>__v2, <tag>__v3, ...

This guarantees that all generated structures are preserved.


POSCAR Species Ordering
-----------------------

To control the species order in the written POSCAR, you may provide:

::

   [generate]
   poscar_order = [...]

If this list is empty, the structure is written using pymatgen’s default ordering.
If non-empty, sites are reordered to match the given preference (and any remaining
species are appended).


Outputs
-------

For each generated structure:

- ``POSCAR``: doped supercell structure
- ``metadata.json``: provenance information, including:

  - host species and number of host sites
  - requested and effective compositions
  - rounded substitution counts
  - seed used for deterministic substitution
  - composition tag and input file name


Notes and Limitations
---------------------

- This stage performs **structure generation only** and does not evaluate stability.
- Rounding is unavoidable for small supercells; larger supercells reduce rounding error.
- Enumerated composition counts can grow combinatorially with the number of dopants and levels;
  users should choose constraints accordingly.
