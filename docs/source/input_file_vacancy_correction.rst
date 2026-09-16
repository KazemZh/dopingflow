Vacancy M0/M1 Input Options
===========================

This page supplements :doc:`input_file`. The scientific method and output
interpretation are described in :doc:`methods/vacancy_energy_correction`.

A complete practical opt-in configuration has two parts. First configure and fit
the same backend-specific correction model used by corrected bulk formation and
phase-diagram analysis::

   [energy_correction]
   enabled = true
   experimental_source = "kingsbury"
   model_family = "auto"        # or force "m0" / "m1"
   correction_terms = ["oxide"]
   m1_elements = "workflow"
   calibration_selection = "phase_resolved"
   auto_fetch_phase_structures = true
   reuse_fitted = true

Then enable application of that already fitted model inside the existing flat
``[vacancies]`` section and use a raw same-backend oxygen reservoir::

   [vacancies]
   static_thermodynamic_analysis = true
   apply_fitted_energy_correction = true
   allow_legacy_energy_correction_provenance = false
   oxygen_reference_mode = "reference_file"
   oxygen_reference_file = "reference_structures/reference_energies.json"

Run the relevant stages in this order::

   dopingflow refs-build -c input.toml
   dopingflow corrections-fit -c input.toml
   dopingflow vacancies -c input.toml

``apply_fitted_energy_correction`` (boolean, default: false)
------------------------------------------------------------

When true, the selected relaxed parent and vacancy minima are corrected with the
already fitted backend-specific M0/M1 model before vacancy counts are compared.
Raw energies are retained in the output. ``model_family = "auto"`` reuses the
M0 or M1 family selected by the established correction-fitting procedure; the
vacancy stage does not fit a second model.

The option cannot be combined with ``oxygen_reference_mode = "global"`` or
``"chemistry-specific"`` because both paths use experimental formation-
enthalpy information for oxygen-related calibration. Use a raw same-backend
oxygen reference such as ``reference_file`` or ``same_calculator`` instead.

Correction/candidate provenance
-------------------------------

The fitted model is tied to its backend signature. Corrected vacancy minima must
be compatible with the fit in backend, model, task, optimizer, force tolerance
(``fmax``), maximum relaxation steps, positive convergence, and available
structure/energy provenance. For example, a correction fit built with
``fmax = 0.02`` is not silently applied to a vacancy relaxation known to have
used ``fmax = 0.05``. Rebuild/refit the correction with the intended compatible
settings rather than relabeling an existing calculation.

``allow_legacy_energy_correction_provenance`` (boolean, default: false)
------------------------------------------------------------------------

Explicitly accepts older vacancy relaxation metadata that lacks package-version
or relaxed-POSCAR hash fields. Known backend/model/task, optimizer, force-
tolerance, maximum-step, convergence, energy, or structure mismatches remain
errors.

This option exists for migration of older vacancy datasets. Keep it false for
new calculations whenever the full provenance fields are available.

Oxygen-reference note
---------------------

The ``[energy_correction]`` source (for example Kingsbury) and the vacancy
``oxygen_calibration_experimental_source`` setting serve different purposes.
When M0/M1 vacancy correction is active, the former supplies the experimental
data used to fit the common bulk/vacancy correction model, while the latter is
not used because the vacancy oxygen reservoir must remain a raw same-backend
reference. This keeps corrected phase-diagram and corrected vacancy energetics
on one correction model without double counting the experimental oxygen
calibration.
