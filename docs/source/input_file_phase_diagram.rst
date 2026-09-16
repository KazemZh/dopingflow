Phase-Diagram Input Options
===========================

This page supplements :doc:`input_file` with the vacancy-resolved
phase-diagram options. The full method is described in
:doc:`methods/phase_diagram` and :doc:`methods/vacancy_energy_above_hull`.

The relevant ``input.toml`` block is::

   [phase_diagram]
   skip_if_done = true
   stable_threshold_eV_per_atom = 1.0e-8
   include_vacancy_minima = false
   vacancy_results_directory = ""

``include_vacancy_minima`` (boolean, default: false)
----------------------------------------------------

When true, the phase-diagram stage reads the lowest-energy relaxed structure
for every investigated oxygen-vacancy count from ``vacancy_static_minima.csv``
and inserts those entries into the same raw/corrected phase diagram.

Because an existing cached phase diagram may not contain those vacancy entries,
the implementation rebuilds the phase diagram when this option is enabled.

``vacancy_results_directory`` (string, default: "")
----------------------------------------------------

Directory containing ``vacancy_static_minima.csv`` and the corresponding
relaxed vacancy structures. Relative paths are resolved from the directory
containing ``input.toml``.

When this field is blank, dopingflow first uses
``[vacancies].parent_directory`` when ``parent_source = "directory"``;
otherwise it uses the workflow root.

The additional vacancy-only output is written in the workflow project root as::

   vacancy_energy_above_hull.csv

This output is a closed-system energy-above-hull analysis. It is distinct from
vacancy formation free energy and from an oxygen-open grand-potential phase
diagram.
