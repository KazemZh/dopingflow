6. Formation Energy Evaluation
==============================

Implementation
--------------

This stage is implemented in:

::

   src/dopingflow/formation.py

The public entry point is:

::

   run_formation(...)


Purpose
-------

This stage computes the **formation energy** of relaxed doped structures using
reference energies constructed in Step 00.

It combines:

- the relaxed total energy of each doped candidate structure (:math:`E_{\mathrm{doped}}`)
- the relaxed total energy of the pristine supercell (:math:`E_{\mathrm{pristine}}`)
- elemental chemical potentials (:math:`\mu_i`) for host and dopant species

Formation energies are written per composition folder to:

- ``formation_energies.csv`` (summary table)
- ``candidate_*/04_formation/meta.json`` (per-candidate provenance)


Inputs
------

This stage uses settings from the following sections of ``input.toml``:

- ``[structure]``: provides the output directory containing structure folders.
- ``[doping]``: defines the substitution host species.
- ``[scan]``: provides the anion species list used to identify dopants.
- ``[formation]``: controls skipping and the normalization convention.

It also requires the reference-energy JSON from Step 00:

::

   reference_structures/reference_energies.json


Formation Energy Framework
--------------------------

Substitutional doping model
~~~~~~~~~~~~~~~~~~~~~~~~~~~

The workflow assumes substitutional doping on a host sublattice.
Dopants are identified as all species that are:

- not equal to the host species (``[doping].host_species``), and
- not in the anion list (``[scan].anion_species``)

The set of dopant counts :math:`n_i` is extracted from each candidate POSCAR.

Formation energy definition
~~~~~~~~~~~~~~~~~~~~~~~~~~~

The formation energy is defined as:

.. math::

   E_{\mathrm{form}} =
   E_{\mathrm{doped}}
   - E_{\mathrm{pristine}}
   + \sum_i n_i \left( \mu_{\mathrm{host}} - \mu_i \right)

where:

- :math:`E_{\mathrm{doped}}` is the relaxed total energy of the doped supercell
- :math:`E_{\mathrm{pristine}}` is the relaxed total energy of the pristine supercell
- :math:`\mu_{\mathrm{host}}` is the host chemical potential (per atom)
- :math:`\mu_i` is the dopant chemical potential (per atom)
- :math:`n_i` is the number of dopant atoms of species :math:`i` in the supercell

This corresponds to replacing :math:`n_i` host atoms by :math:`n_i` dopant atoms
for each dopant species :math:`i`, while keeping the same supercell size.

Reference energies are taken from Step 00 and must be consistent with the
supercell size and host species used here.


Method Summary
--------------

For each structure folder inside ``[structure].outdir``:

1. Load the reference data from:

   ::

      reference_structures/reference_energies.json

   extracting :math:`E_{\mathrm{pristine}}` and chemical potentials :math:`\mu_i`.

2. Determine which candidates to evaluate:

   a. If ``selected_candidates.txt`` exists, only those candidates are used.
   b. Otherwise, all ``candidate_*/02_relax/POSCAR`` files are used.

3. For each selected candidate:

   a. Read the relaxed energy :math:`E_{\mathrm{doped}}` from
      ``candidate_*/02_relax/meta.json``.
   b. Read species counts from ``candidate_*/02_relax/POSCAR`` and infer dopant
      counts under the substitutional model.
   c. Evaluate :math:`E_{\mathrm{form}}` using the equation above.
   d. Apply the requested normalization (see below).
   e. Write ``candidate_*/04_formation/meta.json``.

4. Write ``formation_energies.csv`` in the folder, sorted by total formation
   energy.


Normalization Options
---------------------

This stage supports three reporting modes controlled by:

::

   [formation]
   normalize = "total" | "per_dopant" | "per_host"

The internal formation energy is always computed as a **total supercell energy**
(:math:`E_{\mathrm{form}}` in eV). The reported value can be:

- ``total``:
  report :math:`E_{\mathrm{form}}` in eV (no normalization)

- ``per_dopant`` (default):
  report :math:`E_{\mathrm{form}} / N_{\mathrm{dop}}`, where
  :math:`N_{\mathrm{dop}} = \sum_i n_i` is the total number of dopant atoms

- ``per_host``:
  report :math:`E_{\mathrm{form}} / N_{\mathrm{atoms}}`, where
  :math:`N_{\mathrm{atoms}}` is the total number of atoms in the pristine supercell
  (as stored in the reference JSON)

Note:
``per_host`` currently uses the total number of atoms in the pristine supercell.
If you later want normalization per *host-sublattice* site, that quantity can be
stored explicitly in the reference JSON and used here.


Outputs
-------

Per-folder summary
~~~~~~~~~~~~~~~~~~

For each structure folder, this stage writes:

::

   formation_energies.csv

Columns:

- ``candidate``: candidate directory name
- ``E_doped_eV``: relaxed total energy of the doped candidate
- ``E_form_eV_total``: total formation energy in eV
- ``E_form_<normalize>``: normalized formation energy (according to config)
- ``n_dopant_atoms``: total dopant atoms :math:`N_{\mathrm{dop}}`
- ``dopant_counts``: compact dopant count string (e.g. ``Sb:2;Zr:1``)

Rows are sorted by ``E_form_eV_total`` (ascending).

Per-candidate metadata
~~~~~~~~~~~~~~~~~~~~~~

For each evaluated candidate, this stage writes:

::

   candidate_XXX/04_formation/meta.json

This file includes:

- full formation-energy definition string from the reference JSON
- :math:`E_{\mathrm{doped}}`, :math:`E_{\mathrm{pristine}}`
- chemical potentials used for the involved species
- inferred dopant counts
- total formation energy and the reported normalized value


Reproducibility and Skipping
----------------------------

If:

::

   [formation].skip_if_done = true

and ``formation_energies.csv`` already exists for a folder, that folder is
skipped.

Given unchanged relaxed energies, POSCARs, reference JSON, and configuration,
this stage is deterministic.


Notes and Limitations
---------------------

- This stage assumes substitutional doping and uses a simple species-based
  dopant identification rule (host vs anions vs dopants).
- No charged-defect corrections, finite-size corrections, entropy terms, or
  competing-phase chemical potential bounds are included.
- The absolute values depend on the reference energies and the chosen bulk
  phases used to define :math:`\mu_i`.
