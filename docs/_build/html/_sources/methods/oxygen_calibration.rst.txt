Oxygen Reference Calibration
============================

Purpose
-------

Oxygen-vacancy thermodynamics require an oxygen chemical potential on the same
energy scale as the ML calculator used for the parent and vacancy structures.
A raw isolated-``O2`` energy from a solid-state foundation model can carry a
large model-dependent error.  ``dopingflow`` therefore provides optional
experimental calibration of the oxygen reference without hard-coding a
universal correction constant.

The calibration uses only **real ordinary binary oxides that were already
calculated by ``refs-build``** and have a matching experimental standard
formation enthalpy at approximately 298 K.  Missing oxide stoichiometries are
never generated or invented for the purpose of fitting.

Theory
------

For an eligible binary oxide :math:`M_xO_y`, the workflow combines the
same-backend ML energies with the experimental 298 K formation enthalpy:

.. math::

   \mu_{O,i}^{0,\mathrm{cal}} =
   \frac{E^{\mathrm{ML}}(M_xO_y)
   - x E^{\mathrm{ML}}(M)
   - \Delta H_{f,i}^{\mathrm{exp}}(298\,\mathrm{K})}{y}.

Every accepted oxide therefore gives one inferred oxygen reference per oxygen
atom.  The calibrated value is the arithmetic mean of the accepted per-O
values.  The individual values are retained and reported together with their
spread and the resulting formation-enthalpy residuals.

This calibration is specific to the ``backend / model / task`` identity stored
by ``refs-build``.  If that identity differs from the vacancy calculator, the
calibration is rejected rather than reused.

Calibration scopes
------------------

Two calibrated ``oxygen_reference_mode`` values are available.

``global``
   Use every eligible ordinary binary oxide in the current reference inventory,
   including the host oxide when requested, provided that the corresponding
   bulk metal reference and experimental 298 K formation enthalpy are available.
   This gives one backend-wide oxygen reference for the current reference set.

``chemistry-specific``
   Refit the oxygen reference for each vacancy composition.  Only oxides whose
   cation belongs to the actual vacancy chemistry are eligible.  For an
   Sn-Sb-O parent, for example, Sn and Sb oxides may contribute; Ti or Zr oxides
   are excluded even if they are present in ``oxides_ref``.  Only oxides that
   actually exist in the reference inventory and experimental dataset are used.

The chemistry-specific mode is useful when the spread of inferred oxygen
references shows a significant chemistry dependence in a universal ML model.
The global mode is useful when one internally consistent backend reference is
preferred across a broad screening set.

Experimental data
-----------------

The default source is the curated Kingsbury experimental formation-enthalpy
dataset already used by the optional correction infrastructure.  It is loaded
through ``matminer`` and normalized internally to eV per reduced formula unit.
Install the optional support with:

.. code-block:: bash

   pip install -e ".[corrections]"

The available settings are:

.. code-block:: toml

   [vacancies]
   oxygen_calibration_experimental_source = "kingsbury"
   # alternatives: "custom", "kingsbury+custom"

   oxygen_calibration_experimental_data = ""
   oxygen_calibration_dataset_cache_dir = ""

A custom CSV uses the same explicit schema as the existing experimental
correction loader and must contain at least:

``formula, formation_enthalpy, uncertainty, phase, temperature, units, source``.

The temperature must represent the standard 298 K value.  Supported energy
units are the explicit eV/atom or eV/formula-unit forms accepted by the loader.

Reference eligibility
---------------------

An oxide is included only when all of the following are true:

- it is an ordinary binary M-O oxide rather than a peroxide, superoxide, or a
  multication/polyanion composition;
- its relaxed structure and energy are present in
  ``reference_structures/reference_energies.json`` (or it is the calculated
  host oxide when host inclusion is enabled);
- the matching elemental bulk-metal reference is present;
- an experimental 298 K formation enthalpy with the same reduced formula is
  available;
- the reference backend, model and task match the vacancy calculation.

The minimum number of accepted oxides is controlled by:

.. code-block:: toml

   oxygen_calibration_min_references = 2

The default of two prevents a one-point value from being presented as a fit.
The user may deliberately lower it to one for a chemistry where only one
validated oxide is available, but the resulting calibration has no independent
spread estimate.

The host oxide is considered by default:

.. code-block:: toml

   oxygen_calibration_include_host_oxide = true

If the same reduced formula is also present in ``oxides_ref``, it is used only
once.

Finite-temperature oxygen chemical potential
---------------------------------------------

For calibrated modes, the fitted reference is combined with gas-phase
thermochemistry as

.. math::

   \mu_O(T,p) =
   \mu_O^{0,\mathrm{cal}}
   + \frac{1}{2}
     \left[H_{O_2}(T)-H_{O_2}(298)-T S_{O_2}(T)\right]
   + \frac{1}{2}k_B T\ln\left(\frac{p_{O_2}}{p^\circ}\right).

With

.. code-block:: toml

   oxygen_standard_state_mode = "nist_shomate"

``dopingflow`` evaluates the O2 enthalpy and entropy continuously from the NIST
Shomate representation (100--6000 K) at the 1 bar standard state.  The 298 K
enthalpy origin is used for calibrated oxygen references because the fit itself
uses experimental 298 K formation enthalpies.

This gas-phase temperature/entropy correction is independent of whether the
zero-temperature oxygen reference is global or chemistry-specific.

Optional solid configurational entropy
---------------------------------------

The finite-temperature vacancy analysis supports three choices:

.. code-block:: toml

   solid_configurational_entropy = "none"
   # or "ideal"
   # or "configurational"

``ideal`` uses the occupied/vacant oxygen-site mixing entropy

.. math::

   S_{\mathrm{config}} = -k_B N_O
   \left[x_v\ln x_v + (1-x_v)\ln(1-x_v)\right],

with :math:`x_v=N_v/N_O`.

``configurational`` uses the actual symmetry-distinct vacancy configurations
generated by the workflow. For exact orbit degeneracy :math:`g_i`,

.. math::

   Z_n(T) = \sum_i g_i
   \exp\left[-rac{E_i-E_{min}}{k_B T}ight],

.. math::

   \Delta F_{config}(n,T) = -k_B T \ln Z_n(T).

Exact enumeration is required. If all exact configurations were relaxed, their
relaxed energies are used. Otherwise the full exact single-point spectrum is
used to obtain the configurational correction relative to its minimum, while
the static baseline remains the relaxed minimum. Sampled configurations are not
assigned guessed degeneracies.

The finite-T output combines this solid term with the calibrated oxygen chemical
potential and writes ``vacancy_formation_free_energy.csv/json`` for every vacancy
count, temperature and oxygen pressure. Direct ``delta_mu_O`` intervals remain
static-lattice quantities. Solid vibrational, zero-point, magnetic, thermal-
electronic, anharmonic, thermal-expansion and solid-pV terms remain neglected.

Recommended configuration
-------------------------

For a backend-calibrated finite-temperature vacancy screen:

.. code-block:: toml

   [vacancies]
   static_thermodynamic_analysis = true

   oxygen_reference_mode = "chemistry-specific"
   oxygen_reference_file = "reference_structures/reference_energies.json"
   oxygen_calibration_experimental_source = "kingsbury"
   oxygen_calibration_min_references = 2
   oxygen_calibration_include_host_oxide = true

   oxygen_standard_state_mode = "nist_shomate"
   pressure_mapping = true
   temperatures_K = [300.0, 600.0, 900.0]
   standard_oxygen_pressure_bar = 1.0

   solid_configurational_entropy = "none"

Use ``global`` instead when one common backend reference is desired for all
compositions.

Legacy/raw reference modes
--------------------------

The existing modes remain available for diagnostics and backward
compatibility:

``reference_file``
   Read the O2 reference from ``reference_energies.json`` (including the
   existing ``muO_shift_ev`` if configured).

``same_calculator``
   Evaluate an isolated O2 structure with the vacancy calculator.  This is
   calculator-consistent but can be inaccurate for a solid-trained model.

``explicit``
   Use a user-supplied per-O reference.

``none``
   Produce within-count minima but make no cross-vacancy-count stability claim.

Outputs
-------

Calibrated runs add:

``oxygen_calibration_report.json``
   Full calibration audit trail: scope, target elements, individual oxide
   references, experimental values and provenance, fitted per-O reference,
   excluded references, spread and formation-enthalpy residuals.

The standard vacancy thermodynamic outputs are rewritten with the calibrated
reference and include calibration metadata.  For chemistry-specific mode the
per-composition oxygen reference is recorded directly in each output row.

Scientific interpretation
-------------------------

The calibrated value is not a universal constant.  It belongs to the selected
ML backend/model/task and calibration set.  A large spread between the
individual oxide-derived references is a diagnostic that one scalar oxygen
reference does not remove all chemistry-dependent errors of the model.  The
workflow reports that spread explicitly so global and chemistry-specific
calibrations can be compared rather than silently assuming transferability.
