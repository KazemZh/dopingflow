Vacancy M0/M1 Input Options
===========================

This page supplements :doc:`input_file`. The scientific method and output
interpretation are described in :doc:`methods/vacancy_energy_correction`.

The optional settings are part of the existing flat ``[vacancies]`` section::

   [vacancies]
   apply_fitted_energy_correction = false
   allow_legacy_energy_correction_provenance = false

``apply_fitted_energy_correction`` (boolean, default: false)
------------------------------------------------------------

When true, the selected relaxed parent and vacancy minima are corrected with the
already fitted backend-specific M0/M1 model before vacancy counts are compared.
Raw energies are retained in the output.

The correction family is configured and fitted through ``[energy_correction]``::

   [energy_correction]
   enabled = true
   model_family = "m0"       # "m1" or "auto" are also supported

A fitted model must already exist. Run::

   dopingflow corrections-fit -c input.toml

before the vacancy analysis.

The option cannot be combined with ``oxygen_reference_mode = "global"`` or
``"chemistry-specific"`` because both paths use experimental formation-
enthalpy information for oxygen-related calibration. Use a raw same-backend
oxygen reference such as ``reference_file`` or ``same_calculator`` instead.

``allow_legacy_energy_correction_provenance`` (boolean, default: false)
------------------------------------------------------------------------

Explicitly accepts older vacancy relaxation metadata that lacks package-version
or relaxed-POSCAR hash fields. Known backend/model/task, relaxation-setting,
convergence, energy, or structure mismatches remain errors.

This option exists for migration of older vacancy datasets. Keep it false for
new calculations whenever the full provenance fields are available.
