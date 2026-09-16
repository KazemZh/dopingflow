Vacancy workflow example
========================

The complete example is in ``examples/vacancies``. It assumes the normal bulk
workflow has already produced selected relaxed candidates.

.. code-block:: bash

   dopingflow vacancies -c examples/vacancies/input.toml

The example uses one flat ``[vacancies]`` table and a single MACE calculator for
screening and relaxation. Adjust ``device`` and ``model`` to match the installed
environment.

It explicitly selects ``search_method = "enumeration"`` and ``supercell =
[1, 1, 1]``. The checked-in TOML also contains a complete commented Monte Carlo
configuration. Monte Carlo supports constant-temperature sampling with
``mc_annealing = false`` and an optional high-temperature hold plus linear
cooling schedule when enabled. Both methods feed the same top-k relaxation and
reranking stages.

Optional M0/M1 vacancy correction
---------------------------------

The checked-in TOML also documents the complete opt-in path for reusing the
same fitted backend-specific correction model in bulk phase-diagram analysis
and vacancy thermodynamics. The correction block is disabled by default so the
vacancy example remains runnable without fitting a correction first::

   [energy_correction]
   enabled = false
   experimental_source = "kingsbury"
   model_family = "auto"
   correction_terms = ["oxide"]
   m1_elements = "workflow"
   calibration_selection = "phase_resolved"
   auto_fetch_phase_structures = true
   reuse_fitted = true

To activate it, set ``enabled = true``, run ``corrections-fit``, and then set::

   [vacancies]
   static_thermodynamic_analysis = true
   apply_fitted_energy_correction = true
   allow_legacy_energy_correction_provenance = false
   oxygen_reference_mode = "reference_file"
   oxygen_reference_file = "reference_structures/reference_energies.json"

Run the relevant stages in order::

   dopingflow refs-build -c input.toml
   dopingflow corrections-fit -c input.toml
   dopingflow vacancies -c input.toml

The correction/provenance check requires compatible backend, model, task,
optimizer, force tolerance, and maximum relaxation steps. The legacy-provenance
switch only accepts historical fields that were never recorded; it does not
waive a known mismatch.

The M0/M1 vacancy correction is mutually exclusive with the experimental
``global`` or ``chemistry-specific`` vacancy oxygen-reference calibration. Use a
raw same-backend reservoir such as ``reference_file`` or ``same_calculator``
when ``apply_fitted_energy_correction = true``.

Alternative calibrated oxygen reference
---------------------------------------

When M0/M1 vacancy correction is off, the checked-in file can switch from the
backward-compatible ``reference_file`` oxygen reference to ``global`` or
``chemistry-specific`` experimental oxygen calibration. Automatic calibration
uses only real ordinary binary oxides already calculated by ``refs-build`` and
requires same-backend bulk-metal references plus matching experimental 298 K
formation enthalpies.

See :doc:`../methods/oxygen_calibration` for the calibrated oxygen-reference
method and :doc:`../methods/vacancy_energy_correction` for the M0/M1 vacancy
correction equations, outputs, double-counting guard, and provenance rules.
