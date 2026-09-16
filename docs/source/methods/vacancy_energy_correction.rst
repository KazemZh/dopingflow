M0/M1 Correction for Vacancy Thermodynamics
===========================================

Purpose
-------

The oxygen-vacancy workflow can optionally apply the already fitted
backend-specific M0/M1 formation-energy correction to the relaxed parent and
oxygen-deficient minima before different vacancy counts are compared.

The feature is disabled by default. Enable it with::

   [vacancies]
   apply_fitted_energy_correction = true

The correction family itself is still configured in the existing
``[energy_correction]`` section. For example::

   [energy_correction]
   enabled = true
   model_family = "m0"       # or "m1" / "auto"

Run ``dopingflow corrections-fit -c input.toml`` before the vacancy analysis.
If ``model_family = "auto"``, the M0 or M1 model selected by the established
cross-validation/admission procedure is used.

Method
------

For a vacancy structure containing ``n`` removed oxygen atoms, dopingflow keeps
the raw relaxed ML energies but changes the cross-count reaction energy from

.. math::

   \Delta E_{\mathrm{raw}}(n)=E_{\mathrm{def}}^{\mathrm{raw}}(n)
   -E_{\mathrm{parent}}^{\mathrm{raw}}

to

.. math::

   \Delta E_{\mathrm{corr}}(n)=\Delta E_{\mathrm{raw}}(n)
   +C_{\mathrm{def}}(n)-C_{\mathrm{parent}}.

For M0,

.. math::

   C_{M0}=\beta_O N_O,

and for M1,

.. math::

   C_{M1}=\beta_O N_O+\sum_M \beta_M N_M.

For a vacancy series with fixed cation composition and structures classified as
ordinary oxides, the M1 cation terms cancel between parent and defect. The
oxygen feature changes with the number of removed oxygen atoms. The actual
structure is nevertheless classified for every selected minimum; an unsupported
oxygen environment fails explicitly rather than silently receiving the ordinary
oxide correction.

Correction uncertainty
----------------------

Parent and defect corrections use the same fitted coefficients, so their
uncertainties are correlated. Dopingflow therefore forms the reaction feature
vector

.. math::

   \mathbf q_{vac}=\mathbf x_{def}-\mathbf x_{parent}

and evaluates

.. math::

   \sigma_{vac}=\sqrt{\mathbf q_{vac}^{T}C_\beta\mathbf q_{vac}}.

It does not add the parent and defect correction uncertainties independently in
quadrature.

Oxygen reference and double counting
------------------------------------

The fitted M0/M1 correction and the vacancy ``global`` /
``chemistry-specific`` oxygen-reference calibration both use experimental
formation-enthalpy information to correct oxygen-related systematic error.
They must not be applied together.

Therefore ``apply_fitted_energy_correction = true`` is rejected when::

   oxygen_reference_mode = "global"

or::

   oxygen_reference_mode = "chemistry-specific"

Use a raw same-backend oxygen reference, normally ``reference_file`` or
``same_calculator``, when applying M0/M1 to the vacancy solid energies.

Finite-temperature convention
-----------------------------

The M0/M1 coefficients are fitted to standard formation enthalpies near 298 K.
When the fitted vacancy correction is active, the O2 gas thermochemistry uses a
298 K enthalpy origin:

.. math::

   \Delta\mu_O(T,p)=\frac12[H_{O_2}(T)-H_{O_2}(298)-TS_{O_2}(T)]
   +\frac12 k_BT\ln(p/p^\circ).

Solid vibrational, zero-point, magnetic, electronic, anharmonic, thermal-
expansion and pV terms remain outside this screening model unless separately
stated by the existing configurational-entropy option.

Configurational entropy
-----------------------

M0/M1 is composition-linear. For configurations with the same vacancy count
and the same ordinary-oxide classification, the correction is identical and
therefore does not change their relative configurational energy spectrum. The
existing ``none``, ``ideal`` and exact ``configurational`` vacancy entropy
options can therefore continue to operate on the selected count minima.

Outputs
-------

The existing vacancy tables remain the main outputs. When the option is active,
``vacancy_static_minima.csv`` includes both raw and corrected information,
including:

- ``delta_energy_to_parent_raw_eV``
- ``delta_energy_to_parent_eV`` (active corrected value)
- ``energy_correction_eV``
- ``parent_energy_correction_eV``
- ``vacancy_reaction_correction_eV``
- ``vacancy_reaction_correction_uncertainty_eV``
- ``vacancy_reaction_feature_vector``
- ``correction_model_family`` and ``correction_fit_id``
- ``grand_potential_intercept_raw_eV``
- ``grand_potential_intercept_eV`` (active corrected value)

``vacancy_formation_free_energy.csv`` additionally reports the raw and corrected
finite-temperature vacancy free energies side by side. The existing active
``vacancy_formation_free_energy_eV`` column is the corrected result when this
option is enabled.

Provenance
----------

The fitted model and vacancy energies must use compatible backend/model/task and
relaxation provenance. Older vacancy calculations may predate package-version
and relaxed-POSCAR hashes. They can be explicitly adopted with::

   [vacancies]
   allow_legacy_energy_correction_provenance = true

Known mismatches are still rejected. Keep this false for newly provenance-rich
data whenever possible.
