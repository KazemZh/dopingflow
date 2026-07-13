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
diagrams. ``stable_threshold_eV_per_atom`` controls the boolean ``stable``
column and must be non-negative.

Energy Above Hull
-----------------

For each candidate:

.. math::

   E_\mathrm{above\ hull}
   = E_\mathrm{candidate} - E_\mathrm{hull}

where :math:`E_\mathrm{hull}` is the lowest-energy combination of the available
entries with the same overall composition.

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
