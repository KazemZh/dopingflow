Reference Energy Construction (Step 00)
========================================

Purpose
-------

This stage constructs the reference energies required for
formation energy evaluation of doped structures.

The following quantities are computed:

- The relaxed total energy of the pristine supercell
- The elemental chemical potentials of all species involved
  (host + dopants)

All reference energies are generated using a consistent
machine-learned interatomic potential and stored in:

::

    reference_structures/reference_energies.json


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

1. Reading a unit-cell structure defined in the input configuration.
2. Applying the workflow supercell expansion.
3. Performing full structural relaxation.
4. Extracting the final total energy.

Both atomic positions and lattice vectors are allowed to relax.


Elemental Chemical Potentials
-----------------------------

For each relevant element, a bulk structure is:

- Obtained either from a local file or external database.
- Fully relaxed.
- Used to compute a per-atom chemical potential:

.. math::

   \mu_i = \frac{E_{\mathrm{bulk}}}{N_{\mathrm{atoms}}}

where:

- :math:`E_{\mathrm{bulk}}` is the relaxed total energy of the bulk structure.
- :math:`N_{\mathrm{atoms}}` is the number of atoms in the bulk cell.


Relaxation Method
-----------------

All reference relaxations use:

- Interatomic potential: M3GNet
- Optimizer: FIRE
- Convergence criterion: maximum force below ``fmax``

The relaxations are fully unconstrained (cell + atomic positions).


Reference Sources
-----------------

Two reference sources are supported:

**Local bulk structures**

Bulk structures are read from a user-specified directory.

**External database**

Bulk structures may be retrieved programmatically from an external
materials database using provided identifiers and API credentials.

Downloaded structures are cached locally for reproducibility.


Caching Strategy
----------------

If reference energies have already been computed and
``skip_if_done`` is enabled, this stage is skipped.

This ensures deterministic behavior and avoids unnecessary recomputation.


Modeling Assumptions
--------------------

- Substitutional doping only
- Neutral defect configurations
- No explicit temperature contribution
- No vibrational free energies
- No competing phase stability analysis
- Chemical potentials derived from relaxed elemental bulk phases
- No electrochemical environment


Limitations
-----------

- Energies are ML-predicted (not DFT-level total energies)
- Reference phase selection affects chemical potentials
- No finite-size correction
- No charged defect corrections
- No entropy contributions
