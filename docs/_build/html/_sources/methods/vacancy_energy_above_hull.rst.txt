Vacancy Energy Above Hull
=========================

``dopingflow`` can optionally include the lowest-energy relaxed oxygen-vacancy
structure for each vacancy count in the existing phase-diagram calculation.
This makes it possible to compare energy above hull as a function of the number
of oxygen vacancies without building a separate convex-hull implementation.

Configuration
-------------

The complete input-key reference is also listed in
:doc:`../input_file_phase_diagram`.

Add the following options to ``[phase_diagram]``::

   [phase_diagram]
   include_vacancy_minima = true
   vacancy_results_directory = "vacancy-selected"

``vacancy_results_directory`` must contain ``vacancy_static_minima.csv`` and the
corresponding relaxed vacancy structures. If the option is omitted, the code
first uses ``[vacancies].parent_directory`` when ``parent_source = "directory"``;
otherwise it uses the workflow root.

Run the normal phase-diagram command::

   dopingflow phase-diagram -c input.toml

When vacancy minima are enabled, the phase diagram is rebuilt even if
``skip_if_done = true`` because a pre-existing cached phase diagram does not
contain the vacancy entries.

Method
------

For each actual cation composition and each investigated vacancy count, the
extension reads the minimum selected by ``vacancy_static_minima.csv``. Only
positively converged relaxed minima are accepted. These vacancy structures are
added to the same chemical-system entry set used by the standard phase diagram.
Consequently, vacancy structures can themselves become hull vertices and can
change the decomposition facet seen by other entries.

For ``n_vacancies = 0``, the energy and structure are taken from the vacancy
workflow's own parent reference. If that parent reference reused the original
candidate relaxation, the existing phase-diagram entry is reused rather than
duplicated. If the vacancy workflow performed a separate consistency
relaxation, that n=0 reference remains a distinct entry so its stored energy is
never paired with the wrong geometry. Copied calculations with stale absolute
paths are reconstructed from the local vacancy directory layout.

When ``[energy_correction].enabled = true``, the existing corrected-hull
machinery is reused. Corrections are applied to the entries before a new
``pymatgen`` ``PhaseDiagram`` is constructed; corrected energy above hull is
therefore evaluated against a fully corrected hull rather than by shifting a
raw hull result after the fact.

The corrected-hull provenance checks also apply to vacancy entries. Older
vacancy calculations that predate package-version and relaxed-POSCAR hashes can
be used only through the existing explicit legacy-provenance option::

   [energy_correction]
   allow_legacy_candidate_provenance = true

Known backend/model/task, optimizer, force threshold, maximum steps and positive
convergence must still be compatible with the fitted correction model. A
present but conflicting provenance value is never ignored.

Outputs
-------

The normal outputs are rebuilt and include the vacancy structures in their
chemical systems. In addition, a compact vacancy-only table is written to the
workflow project root as::

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
combination available in the phase-diagram entry set. A value of zero means
that the structure lies on that closed-system hull.

Each vacancy count has a different oxygen stoichiometry, so these points are not
raw total-energy differences. Each point is evaluated against the decomposition
appropriate to its own composition.

This result is distinct from vacancy formation free energy, which asks whether
removing oxygen from the same host is favorable at a chosen oxygen chemical
potential. It is also distinct from an oxygen-open grand-potential hull, which
would make decomposition stability itself a function of oxygen chemical
potential.

The result is only as complete as the competing phases supplied to the phase
diagram. It should therefore be described as stability relative to the
available reference/candidate phase set unless the competing-phase set is known
to be comprehensive.

GUI
---

Start the normal Streamlit application. The automatically discovered
``Phase Diagram`` page provides two views:

- composition-space raw/corrected energy-above-hull plots and decomposition tables;
- raw/corrected energy above hull versus oxygen-vacancy count from
  ``vacancy_energy_above_hull.csv``.

The page also exposes ``include_vacancy_minima`` and
``vacancy_results_directory`` and can save them directly into
``[phase_diagram]`` in ``input.toml``. The composition plotting helper matches
copied calculations by the final ``composition/candidate`` path components, so
absolute path prefixes from another machine do not by themselves break the
plot.
