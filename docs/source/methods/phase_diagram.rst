Phase Diagram and Energy Above Hull
===================================

The phase-diagram step evaluates the thermodynamic stability of relaxed doped
structures using pymatgen's ``PhaseDiagram`` class.

For each candidate, the workflow creates a ``PDEntry`` from the relaxed
composition and total relaxed energy. These entries are combined with reference
phases such as elemental Sn, Sb, O2, SnO2, Sb2O3, and Sb2O5.

The main output is the energy above hull:

.. math::

   E_\mathrm{above\ hull}
   =
   E_\mathrm{candidate}
   -
   E_\mathrm{hull}

where ``E_hull`` is the lowest-energy combination of available competing phases
with the same overall composition.

Output
------

The step writes:

``phase_diagram_results.csv``

Important columns include:

- ``energy_above_hull_eV_per_atom``
- ``stable``
- ``decomposition``