Vacancy Energy Above Hull
=========================

``dopingflow`` can optionally include the lowest-energy relaxed oxygen-vacancy
structure for each vacancy count in the existing phase-diagram calculation.
This makes it possible to compare energy above hull as a function of the number
of oxygen vacancies without building a separate convex-hull implementation.

Configuration
-------------

Add the following options to ``[phase_diagram]``::

   [phase_diagram]
   include_vacancy_minima = true
   vacancy_results_directory = "vacancy-selected"

``vacancy_results_directory`` must contain ``vacancy_static_minima.csv`` and the
corresponding relaxed vacancy structures.  If the option is omitted, the code
first uses ``[vacancies].parent_directory`` when ``parent_source = "directory"``;
otherwise it uses the workflow root.

Run the normal phase-diagram command::

   dopingflow phase-diagram -c input.toml

Method
------

For each actual cation composition and each investigated vacancy count, the
extension reads the minimum selected by ``vacancy_static_minima.csv``.  Only
positively converged relaxed minima are accepted.  These vacancy structures are
added to the same chemical-system entry set used by the standard phase diagram.
Consequently, vacancy structures can themselves become hull vertices and can
change the decomposition facet seen by other entries.

When ``[energy_correction].enabled = true``, the existing corrected-hull
machinery is reused.  Corrections are applied to the entries before a new
``pymatgen`` ``PhaseDiagram`` is constructed; corrected energy above hull is
therefore evaluated against a fully corrected hull rather than by shifting a
raw hull result after the fact.

Outputs
-------

The normal outputs are rebuilt and include the vacancy structures in their
chemical systems.  In addition, a compact vacancy-only table is written as::

   vacancy_energy_above_hull.csv

Important columns include:

- ``actual_composition_key``
- ``n_vacancies``
- ``vacancy_percent_of_parent_oxygen``
- ``source_configuration_id``
- ``energy_above_hull_eV_per_atom``
- ``energy_above_hull_raw_eV_per_atom`` when correction is enabled
- ``energy_above_hull_corrected_eV_per_atom`` when correction is enabled
- ``stable_raw`` and ``stable_corrected`` when correction is enabled
- ``decomposition_raw`` and ``decomposition_corrected`` when correction is enabled

The resulting table can be plotted directly as corrected energy above hull
versus ``n_vacancies`` for each composition.

Interpretation
--------------

A decrease in energy above hull with increasing vacancy count means that the
oxygen-deficient structure is closer to the lowest-energy decomposition
combination available in the phase-diagram entry set.  A value of zero means
that the structure lies on that hull.  This is distinct from vacancy formation
free energy, which asks whether removing oxygen from the same host is favorable
at a chosen oxygen chemical potential.

The result is only as complete as the competing phases supplied to the phase
diagram.  It should therefore be described as stability relative to the
available reference/candidate phase set unless the competing-phase set is known
to be comprehensive.
