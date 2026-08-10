Phase Diagram and Energy Above Hull
===================================

The phase-diagram step evaluates relaxed candidates with pymatgen's
``PhaseDiagram`` class. It builds a separate diagram for every exact chemical
system represented in ``results_database.csv``. Examples include ``O-Sb-Sn``,
``Ce-O-Sb-Sn``, and ``In-O-Sb-Sn``.

For a given system, the hull includes:

- reference phases whose elements are a subset of that system;
- candidates from the exact system;
- lower-dimensional candidates whose elements are a subset of that system.

Only candidates belonging to the exact system are reported in that system's
CSV. This prevents unrelated dopants from being combined in one artificial
high-dimensional hull.

When ``[energy_correction].enabled = true``, the same system selection is used
to construct a second complete entry set. Every applicable non-elemental
reference, lower-dimensional candidate, and evaluated candidate is corrected.
Elemental terminals remain unchanged. A separate pymatgen ``PhaseDiagram`` is
then built from the corrected entries. The raw hull is never shifted after the
fact, and a mixture of corrected candidates with raw competing compounds is
never labeled a corrected hull. See :doc:`energy_corrections`.

Terminal References
-------------------

A closed phase diagram requires an elemental terminal entry for every element
in the system. ``O2`` supplies the oxygen terminal because its composition
contains only oxygen. Every metal must be listed in ``metal_ref`` and have a
corresponding POSCAR available when ``refs-build`` is run, including when
``reference_mode = "oxide"``.

The step stops with a descriptive error if a terminal is missing.

Configuration
-------------

All settings remain in the existing ``[phase_diagram]`` section::

   [phase_diagram]
   skip_if_done = true
   stable_threshold_eV_per_atom = 0.05

``skip_if_done`` returns the existing combined output without rebuilding the
diagrams on the correction-disabled path. With correction enabled, corrected
hulls are always rebuilt because a fit ID alone does not fingerprint candidate
structures, energies, membership, or the stability threshold. Disabling
correction also rebuilds an existing file that contains corrected columns.
``stable_threshold_eV_per_atom`` controls the boolean raw and corrected
stability columns and must be non-negative.

Energy Above Hull
-----------------

For each candidate:

.. math::

   E_\mathrm{above\ hull}
   = E_\mathrm{candidate} - E_\mathrm{hull}

where :math:`E_\mathrm{hull}` is the lowest-energy combination of the available
entries with the same overall composition.

Raw and corrected energy above hull are calculated from their respective
independently constructed hulls:

.. math::

   E_{\mathrm{above\ hull}}^{\mathrm{raw}}
   = E_{\mathrm{candidate}}^{\mathrm{raw}}-E_{\mathrm{hull}}^{\mathrm{raw}},

.. math::

   E_{\mathrm{above\ hull}}^{\mathrm{corrected}}
   = E_{\mathrm{candidate}}^{\mathrm{corrected}}
   -E_{\mathrm{hull}}^{\mathrm{corrected}}.

It is scientifically incorrect to add a phase correction directly to raw
energy above hull because correction can change the hull facets and
decomposition itself.

Outputs
-------

The combined output is:

``phase_diagram_results.csv``

Individual systems are also written under ``phase_diagrams/``::

   phase_diagrams/phase_diagram_O-Sb-Sn.csv
   phase_diagrams/phase_diagram_Ce-O-Sb-Sn.csv

Columns include:

- ``chemical_system``
- ``candidate`` and ``composition_tag``
- ``formula``
- ``energy_total_eV`` and ``energy_per_atom_eV``
- ``energy_above_hull_eV_per_atom``
- ``stable``
- ``decomposition``

These legacy columns remain raw. When correction is enabled, additional
columns include:

- ``energy_raw_eV``, ``energy_correction_eV``, and ``energy_corrected_eV``
- ``correction_uncertainty_eV``
- ``energy_above_hull_raw_eV_per_atom``
- ``energy_above_hull_corrected_eV_per_atom``
- ``energy_above_hull_correction_eV_per_atom`` (corrected minus raw hull
  distance, including any facet change)
- ``energy_above_hull_parameter_shift_eV_per_atom``
- ``energy_above_hull_correction_uncertainty_eV_per_atom`` and its combined
  reaction ``q`` vector
- ``stable_raw`` and ``stable_corrected``
- ``decomposition_raw`` and ``decomposition_corrected``
- applicability reason, method, fit ID, parameter set, experimental dataset,
  and backend/model/task provenance

If any required non-elemental entry has no structure, an incompatible
backend/model/task/settings provenance, lacks positive convergence, or has an
oxygen environment absent from the fitted basis, the corrected diagram for
that system fails explicitly. Raw results are not overwritten.

The correction uncertainty is evaluated as
:math:`\sqrt{\mathbf q^T C_\beta\mathbf q}` for the candidate minus the phases
on its corrected-hull decomposition. This is a fixed-corrected-facet
linearization: it retains coefficient correlations, but it does not yet sample
coefficient uncertainty to estimate the probability of a different hull facet.
