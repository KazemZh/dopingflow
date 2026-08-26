0. Reference Energy Construction
================================

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

This stage prepares all thermodynamic reference quantities required for
formation energy evaluation of substitutionally doped structures.

The stage performs the following tasks:

- Relax the host oxide unit cell
- Build and relax the host supercell
- Relax reference structures according to the selected reference scheme
- Store all relevant energies and metadata in:

::

   reference_structures/reference_energies.json

The resulting reference data are later used by the formation-energy stage.

If backend-specific correction fitting is enabled, these same backend/model/task
settings and elemental terminal energies also define calculated calibration
formation enthalpies. Calibration compounds themselves come from a separate
manifest and are never inferred from ``oxides_ref``. See
:doc:`energy_corrections`.

For correction application, ``[relax]`` must use the same backend, model, and
task as ``[references]``. Runtime provenance is also exact: after a backend
package or local-checkpoint change, rebuild references and rerun stale
candidate relaxations before refitting and applying the model.


Inputs
------

This stage uses settings from:

- ``[references]``: reference mode, host structure, reference directories,
  backend settings, relaxation settings, oxygen settings, and caching behavior
- ``[doping]``: defines the host species and the dopant set used in later steps

The host supercell is defined in ``[references].supercell`` and is constructed
at this stage.


Execution Model
---------------

The references stage uses the same backend abstraction as the main relaxation
stage.

For each structure to be optimized, the workflow:

1. Loads the selected ML backend
2. Builds an ASE-compatible calculator
3. Applies the selected ASE optimizer
4. Relaxes the structure until convergence or until ``max_steps`` is reached

Supported backends include:

- ``m3gnet``
- ``uma``
- ``mace``
- ``grace``

Supported optimizers include:

- ``bfgs``
- ``lbfgs``
- ``fire``
- ``mdmin``
- ``quasinewton``

The execution device is controlled through:

- ``device`` (``cpu`` or ``cuda``)
- ``gpu_id``

This design keeps the references stage consistent with ``relax.py``.


Reference Modes
---------------

Two thermodynamic reference schemes are supported.


Metal reference mode
~~~~~~~~~~~~~~~~~~~~

In metal mode, elemental chemical potentials are taken from relaxed elemental
reference phases.

For each relevant element :math:`i`, the workflow relaxes the corresponding
metal structure and computes:

.. math::

   \mu_i = \frac{E_{\mathrm{metal}}}{N_{\mathrm{atoms}}}

where:

- :math:`E_{\mathrm{metal}}` is the relaxed total energy of the elemental
  reference structure
- :math:`N_{\mathrm{atoms}}` is the number of atoms in that structure

This mode corresponds to equilibrium with elemental reservoirs.


Oxide reference mode
~~~~~~~~~~~~~~~~~~~~

In oxide mode, dopant chemical potentials are derived from oxide reference
phases together with the oxygen chemical potential.

For a binary oxide :math:`M_xO_y`, the chemical potential satisfies:

.. math::

   x\mu_M + y\mu_O = E_{M_xO_y}

which gives:

.. math::

   \mu_M = \frac{E_{M_xO_y} - y\mu_O}{x}

The oxygen convention is resolved in two explicit steps. First, an optional
electronic/reference correction is applied to the raw same-backend gas energy:

.. math::

   E_{O_2}^{ref} = E_{O_2}^{raw} + 2\,\Delta E_O^{ref}.

Then the physical oxygen chemical potential is

.. math::

   \mu_O = \frac{1}{2}E_{O_2}^{ref} + \Delta\mu_O.

Here ``oxygen_reference_correction_ev`` is :math:`\Delta E_O^{ref}` in eV per
O atom, whereas ``delta_mu_O_ev`` is the physical thermodynamic
:math:`\Delta\mu_O` relative to O-rich conditions. ``oxygen_mode = "O-rich"``
requires :math:`\Delta\mu_O=0`; ``oxygen_mode = "O-poor"`` permits only zero or
negative values.

When ``[energy_correction].enabled = true``, ``oxygen_reference_correction_ev``
must be zero. The experimental correction model is fitted against the raw
same-backend oxygen reference, so an additional empirical oxygen-reference
shift would double count an oxygen-linear correction. ``delta_mu_O_ev`` remains
allowed because it describes the physical thermodynamic environment rather
than an electronic-structure correction.

The legacy ``muO_shift_ev`` key is retained only for migration; new inputs
should use the two explicit quantities above. See :doc:`oxygen_thermodynamics`
for the complete compatibility rules.


Method Summary
--------------

1. Read the host oxide unit-cell structure
2. Relax the host unit cell
3. Build and relax the host supercell
4. Determine the selected reference mode

Metal mode:
    a. Relax elemental metal references
    b. Compute per-atom elemental chemical potentials

Oxide mode:
    a. Relax oxide reference structures
    b. Relax the oxygen gas reference
    c. Compute :math:`\mu_O`
    d. Derive cation chemical potentials from oxide thermodynamics

5. Write all results and metadata to ``reference_energies.json``


Formation Energy Framework
--------------------------

The workflow assumes substitutional doping on host sites.

The formation energy is defined as:

.. math::

   E_{\mathrm{form}} =
   E_{\mathrm{doped}}
   - E_{\mathrm{pristine}}
   + \sum_i n_i \left( \mu_{\mathrm{host}} - \mu_i \right)

where:

- :math:`E_{\mathrm{doped}}` is the relaxed total energy of the doped supercell
- :math:`E_{\mathrm{pristine}}` is the relaxed total energy of the pristine host supercell
- :math:`\mu_i` is the chemical potential of dopant species :math:`i`
- :math:`\mu_{\mathrm{host}}` is the chemical potential of the substituted host species
- :math:`n_i` is the number of substituted atoms of species :math:`i`

This corresponds to removing host atoms and inserting dopant atoms while
keeping the total lattice size fixed.

The same formal expression is used in both reference modes; only the way
the chemical potentials are constructed differs.


Host Reference Energy
---------------------

The pristine host reference energy is computed by:

1. Reading the host oxide unit cell
2. Relaxing the unit cell
3. Building the requested supercell
4. Relaxing the supercell
5. Extracting the final total energy

The relaxed host supercell is reused by later workflow stages as the
starting point for structure generation.

Both atomic positions and lattice vectors are allowed to relax.


Metal Chemical Potentials
-------------------------

For each relevant elemental reference phase, the workflow computes:

.. math::

   \mu_i = \frac{E_{\mathrm{metal}}}{N_{\mathrm{atoms}}}

These values are used directly in formation-energy evaluation when
``reference_mode = "metal"``.


Oxide-Derived Chemical Potentials
---------------------------------

When ``reference_mode = "oxide"``, the workflow stores relaxed oxide
reference energies and the oxygen gas reference energy.

The oxygen chemical potential is computed from the explicitly resolved gas
reference and physical ``delta_mu_O_ev``. The cation chemical potentials are
then derived from the oxide stoichiometry.

For a reduced oxide composition :math:`M_xO_y`:

.. math::

   \mu_M = \frac{E_{M_xO_y}^{\mathrm{(f.u.)}} - y\mu_O}{x}

where :math:`E_{M_xO_y}^{\mathrm{(f.u.)}}` is the relaxed energy per
formula unit of the oxide reference.


Relaxation Method
-----------------

All reference relaxations use:

- a selected ML interatomic potential backend
- an ASE-compatible calculator
- a user-selected ASE optimizer
- a force-based convergence criterion defined by ``fmax``

The maximum number of optimization steps is controlled by ``max_steps``.

The backend is selected through:

- ``backend``
- ``model``
- ``task`` (required for UMA; optional MACE multi-head model head)

The runtime environment is controlled through:

- ``device``
- ``gpu_id``
- ``tf_threads``
- ``omp_threads``

The relaxations are fully unconstrained
(cell parameters and atomic positions).


Caching Strategy
----------------

If:

::

   skip_if_done = true

the stage validates each cached host/reference against its source-file hash,
relaxed-output hash, and relaxation signature. Compatible entries are reused;
a new, changed, tampered, or incompatible entry is relaxed.
``skip_if_done = false`` forces recomputation.

Correction models have a separate fit-input hash under
``reference_structures/corrections`` and are never treated as part of the oxide
reference inventory.


Outputs
-------

The file ``reference_energies.json`` contains metadata and energies needed
for later stages.

Typical top-level fields include:

- ``timestamp``
- ``reference_mode``
- ``backend``
- ``model``
- ``task``
- ``optimizer``
- ``device``
- ``gpu_id``
- ``host``
- ``references``
- ``oxide_mode`` (only relevant in oxide mode)
- ``supercell``
- ``config_path``

The ``host`` block typically contains:

- host formula
- source POSCAR path
- relaxed unit-cell POSCAR path
- relaxed supercell POSCAR path
- number of atoms in unit cell and supercell
- total and per-atom energies
- optimizer step counts
- final force values
- convergence status

The ``references`` block contains one entry per relaxed reference phase,
for example metals, oxides, or gas references.

Each reference entry may include:

- source POSCAR path
- relaxed POSCAR path
- total energy
- per-atom energy
- per-formula-unit energy (when applicable)
- per-molecule energy for O₂ (when applicable)
- optimizer steps
- final force value
- convergence status
- wall time
- backend and optimizer metadata

For oxide mode, the JSON also stores the oxygen reference settings such as:

- ``oxides_ref``
- ``gas_ref``
- ``oxygen_mode``
- ``muO_shift_ev``


Notes and Limitations
---------------------

- This stage does not evaluate doped structures
- Energies are ML-predicted, not DFT total energies
- Reference phase selection strongly affects the resulting chemical potentials
- The workflow currently assumes substitutional doping
- Oxide-mode derivation assumes simple oxide reference chemistry
- No finite-size corrections are applied
- No charge-state corrections are included
- No entropy or temperature effects are considered
- No competing phase stability analysis is performed
- Experimental calibration selection and fitting are performed by the optional
  ``corrections-fit`` stage, not by ``refs-build``
