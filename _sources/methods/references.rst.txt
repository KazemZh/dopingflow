0. Reference Energy Construction
=================================

Implementation
--------------

This stage is implemented in:

::

   src/dopingflow/refs.py

The public entry point is:

::

   run_refs_build(...)


Purpose
-------

This stage constructs the reference energies required for
formation energy evaluation of doped structures.

The following quantities are computed:

- The relaxed total energy of the pristine supercell
- The elemental chemical potentials of all species involved
  (host + dopants)

All reference data are written to:

::

   reference_structures/reference_energies.json

These references are later used for formation energy evaluation.


Inputs
------

This stage uses settings from:

- ``[structure]``: provides the pristine structure and supercell.
- ``[doping]``: defines the host species and dopant set.
- ``[references]``: controls reference source, relaxation settings, and caching behavior.


Method Summary
--------------

1. Read the pristine unit-cell structure.
2. Apply the workflow supercell expansion.
3. Relax the pristine supercell.
4. Identify all elements involved (host + dopants).
5. For each element:
   a. Obtain a bulk structure (local or external source).
   b. Relax the bulk structure.
   c. Compute per-atom chemical potential.
6. Store all reference quantities in a JSON file.


Formation Energy Framework
--------------------------

The workflow assumes substitutional doping.

The formation energy is defined as:

.. math::

   E_{\mathrm{form}} =
   E_{\mathrm{doped}}
   - E_{\mathrm{pristine}}
   + \sum_i n_i \left( \mu_{\mathrm{host}} - \mu_i \right)

where:

- :math:`E_{\mathrm{doped}}` is the relaxed total energy of the doped supercell
- :math:`E_{\mathrm{pristine}}` is the relaxed total energy of the pristine supercell
- :math:`\mu_i` is the chemical potential of dopant species :math:`i`
- :math:`\mu_{\mathrm{host}}` is the chemical potential of the substituted host species
- :math:`n_i` is the number of substituted atoms of species :math:`i`

This corresponds to removing host atoms and inserting dopant atoms
while preserving total lattice size.


Pristine Supercell Energy
-------------------------

The pristine reference energy is computed by:

1. Reading the unit-cell structure defined in the input.
2. Applying the supercell defined in ``[structure].supercell``.
3. Performing full structural relaxation.
4. Extracting the final total energy.

Both atomic positions and lattice vectors are allowed to relax.


Elemental Chemical Potentials
-----------------------------

For each relevant element, a bulk structure is:

- Obtained either from a local file or an external database.
- Fully relaxed.
- Used to compute a per-atom chemical potential:

.. math::

   \mu_i = \frac{E_{\mathrm{bulk}}}{N_{\mathrm{atoms}}}

where:

- :math:`E_{\mathrm{bulk}}` is the relaxed total energy of the bulk structure.
- :math:`N_{\mathrm{atoms}}` is the number of atoms in the bulk cell.


Reference Sources
-----------------

Two reference sources are supported.


Local bulk structures
~~~~~~~~~~~~~~~~~~~~~

Bulk structures are read from a user-specified directory.
Each element must have a corresponding structure file.


External database
~~~~~~~~~~~~~~~~~

Bulk structures may be retrieved programmatically from
an external materials database using provided identifiers
and API credentials.

Downloaded structures are cached locally to ensure reproducibility.


Relaxation Method
-----------------

All reference relaxations use:

- Interatomic potential: M3GNet
- Optimizer: FIRE
- Convergence criterion: maximum force below ``fmax``

The relaxations are fully unconstrained
(cell parameters and atomic positions).


Caching Strategy
----------------

If:

::

   skip_if_done = true

and the reference JSON already exists,
this stage is skipped.

This ensures deterministic behavior and avoids unnecessary recomputation.


Outputs
-------

The file ``reference_energies.json`` contains:

- Timestamp
- Host species
- Supercell definition
- Relaxed pristine supercell energy
- Chemical potentials per element
- Metadata describing bulk relaxation conditions
- Reference source information


Notes and Limitations
---------------------

- This stage does not evaluate doped structures.
- Energies are ML-predicted (not DFT-level total energies).
- Reference phase selection affects chemical potentials.
- No finite-size corrections are applied.
- No charge-state corrections are included.
- No entropy or temperature effects are considered.
- No competing phase stability analysis is performed.
