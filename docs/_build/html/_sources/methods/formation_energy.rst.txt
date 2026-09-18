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
- ``[formation]``: controls skipping, normalization, and optional relative columns.
- ``[energy_correction]``: optionally selects a fitted, backend-compatible
  correction model. See :doc:`energy_corrections`.

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

For every species :math:`\alpha`, let :math:`\Delta n_\alpha` be its candidate
atom count minus its pristine-supercell atom count. The general atom-balanced
definition is:

.. math::

   E_{\mathrm{form}} =
   E_{\mathrm{doped}}
   - E_{\mathrm{pristine}}
   - \sum_\alpha \Delta n_\alpha \mu_\alpha.

where:

- :math:`E_{\mathrm{doped}}` is the relaxed total energy of the doped supercell
- :math:`E_{\mathrm{pristine}}` is the relaxed total energy of the pristine supercell
- :math:`\mu_{\mathrm{host}}` is the host chemical potential (per atom)
- :math:`\mu_i` is the dopant chemical potential (per atom)
- :math:`\Delta n_\alpha` is the signed stoichiometric change for species
  :math:`\alpha`

For a purely substitutional candidate at fixed oxygen content this reduces to

.. math::

   E_{\mathrm{form}} = E_{\mathrm{doped}}-E_{\mathrm{pristine}}
   +\sum_i n_i(\mu_{\mathrm{host}}-\mu_i).

Using the general form is essential for a doped structure that also contains
host or oxygen vacancies. In particular, an oxygen change contributes
:math:`-\Delta n_\mathrm{O}\mu_\mathrm{O}` rather than being silently treated as
stoichiometric SnO2.

In metal-reference mode, such an oxygen-nonstoichiometric reaction requires a
same-backend O2 entry in ``reference_energies.json``. If it is absent, the stage
fails rather than assuming an oxygen chemical potential.

Reference energies are taken from Step 00 and must be consistent with the
supercell size and host species used here.

Raw and corrected quantities
~~~~~~~~~~~~~~~~~~~~~~~~~~~~

The existing ``E_form_*`` fields remain the raw formation energies defined
above. They are never overwritten.

When ``[energy_correction].enabled = true``, the stage loads the fitted model
whose backend/model/task and fit terms match the current references. Candidate
relaxation metadata must use the same backend/model/task. The correction is
then evaluated for the complete balanced reaction rather than added only to the
final scalar.

Compatibility also includes resolved runtime provenance such as the installed
backend package version and a local checkpoint's content hash. Existing relax
outputs are not made current merely by editing ``input.toml``. After a package
or checkpoint change, set both ``[relax].skip_if_done`` and
``[relax].skip_candidate_if_done`` to ``false`` and rerun relaxation before
formation analysis.

For metal-reference mode, elemental chemical potentials are not corrected, so
the reaction correction is

.. math::

   \Delta C_{\mathrm{form}} = C_{\mathrm{doped}} - C_{\mathrm{pristine}}.

For each oxide-reference scenario, the corrected host and dopant chemical
potentials are derived from corrected host/oxide compound energies. If the
host formula contains :math:`a` host atoms and dopant oxide :math:`i` contains
:math:`d_i` dopant atoms, then

.. math::

   \Delta C_{\mathrm{form}} = C_{\mathrm{doped}}-C_{\mathrm{pristine}}
   -\Delta n_{\mathrm{host}}\frac{C_{\mathrm{host, FU}}}{a}
   -\sum_i n_i\frac{C_{\mathrm{oxide},i,\mathrm{FU}}}{d_i}.

Co-dopants are included in the same sum; no special Sb/Ti lookup is used. The
mixing reaction is corrected using its own atom-balanced reaction vector. Full
fit covariance is propagated through that combined vector, preserving exact
correlation and cancellation between constituents.

The default ordinary-oxide O term cancels for metal-reference substitution at
fixed oxygen content. Such an explicit zero is expected and is not silently
replaced by a dopant-specific parameter. See :doc:`energy_corrections` for the
model's scientific scope and limitations.


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
   d. If enabled, validate the active fit and evaluate the corrected balanced
      reaction and its coefficient uncertainty.
   e. Apply raw and corrected normalizations separately (see below).
   f. Write ``candidate_*/04_formation/meta.json``.

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


Multiple Oxide References and Relative Values
----------------------------------------------

In oxide mode, every binary oxide listed for a dopant is evaluated. Co-doped
candidates use the Cartesian product of the available oxide choices, producing
deterministic scenario labels such as ``Sb2O3__TiO2``.

Relative output is configured without adding another TOML section::

   [formation]
   relative_enabled = true
   endpoint_x = "auto"

The oxide chemical potentials and the atom-balanced mixing reaction already
reference the candidate to its host-oxide/dopant-oxide tie-line. Consequently,
the relative per-cation values equal their corresponding oxide-referenced
formation and mixing values; collection does not apply a second endpoint
subtraction. ``endpoint_x = "auto"`` records the pure oxide endmember
(:math:`x=1`).

For co-doping, endpoint provenance includes both the endpoint energy for each
dopant and the composition-weighted correction
:math:`\sum_i x_i E_{\mathrm{endpoint},i}`.


Outputs
-------

Per-folder summary
~~~~~~~~~~~~~~~~~~

For each structure folder, this stage writes:

::

   formation_energies.csv

Base columns:

- ``candidate``: candidate directory name
- ``E_doped_eV``: relaxed total energy of the doped candidate
- ``n_dopant_atoms``: total dopant atoms :math:`N_{\mathrm{dop}}`
- ``dopant_counts``: compact dopant count string (e.g. ``Sb:2;Zr:1``)
- ``x_dopant``: actual dopant fraction on the cation sublattice
- ``reference_mode``: ``metal`` or ``oxide``

Each reference scenario then contributes wide columns with a ``__<scenario>``
suffix, including formation energies, mixing energies, relative values, oxide
endpoint energies, reaction text, and JSON endpoint provenance. Rows remain one
candidate per row and are sorted by relaxed doped energy.

When correction is enabled, each scenario additionally includes explicit raw,
correction, corrected, and correction-uncertainty columns. Legacy unsuffixed and
``E_form_*__<scenario>`` values remain raw. Corrected fields use names such as
``E_form_corrected_eV_total__<scenario>`` and include method, fit ID, dataset,
and applicability provenance.

Per-candidate metadata
~~~~~~~~~~~~~~~~~~~~~~

For each evaluated candidate, this stage writes:

::

   candidate_XXX/04_formation/meta.json

This file includes:

- full formation-energy definition string from the reference JSON
- :math:`E_{\mathrm{doped}}`, :math:`E_{\mathrm{pristine}}`
- chemical potentials used for the involved species
- signed candidate-minus-pristine stoichiometric changes and their total
  chemical-potential contribution
- inferred dopant counts
- total formation energy and the reported normalized value
- the full ``reference_results`` mapping for all oxide scenarios
- the primary reference label used for backward-compatible top-level fields
- relative-energy and oxide-endpoint provenance
- when enabled, the correction reaction vector, raw/corrected values,
  coefficient uncertainty, backend signature, dataset, parameter set, and fit ID


Reproducibility and Skipping
----------------------------

If:

::

   [formation].skip_if_done = true

and ``formation_energies.csv`` already exists for a folder, that folder is
skipped on the legacy correction-disabled path. When correction is enabled, an
existing result is rebuilt from the current selected candidates, POSCARs,
relaxation metadata, references, configuration, and active fit. Formation is a
cheap analysis stage compared with relaxation, and this avoids a stale
fit-ID-only cache. Disabling correction also rebuilds output that still contains
corrected fields.

Given unchanged relaxed energies, POSCARs, reference JSON, and configuration,
this stage is deterministic.


Notes and Limitations
---------------------

- This stage requires at least one identified dopant and uses a simple
  species-based rule (host vs anions vs dopants). Signed host/anion deviations
  are nevertheless included in the balanced reaction, so mixed
  dopant--vacancy candidates are not forced to pristine stoichiometry.
- No charged-defect corrections, finite-size corrections, entropy terms, or
  competing-phase chemical potential bounds are included.
- The correction application uses the actual composition of every processed
  structure, but the separate vacancy workflow does not yet publish corrected
  vacancy thermodynamics automatically.
- The absolute values depend on the reference energies and the chosen bulk
  phases used to define :math:`\mu_i`.
