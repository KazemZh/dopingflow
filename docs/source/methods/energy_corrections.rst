Backend-Specific Energy Corrections
===================================

Purpose and scope
-----------------

``dopingflow`` can optionally fit systematic energy corrections using measured
solid-state formation enthalpies and total energies evaluated with the same ML
backend used by the workflow.  The implementation follows the simultaneous,
uncertainty-weighted linear framework described by Wang, Kingsbury *et al.*,
``A framework for quantifying uncertainty in DFT energy corrections``.

The feature is disabled by default.  It does **not** apply the pre-fitted
MP2020 numerical coefficients to UMA, MACE, GRACE, M3GNet, or another ML
potential.  MP2020 coefficients belong to a particular Materials Project
GGA/GGA+U methodology and are not transferable model parameters.

When enabled, the data flow is::

   experimental formation enthalpies
       + phase-matched structures evaluated by one ML backend/model/task
       -> uncertainty-weighted correction fit
       -> corrected compound and balanced-reaction energies
       -> a second, independently rebuilt corrected phase diagram

Raw energies and raw hull results are always retained.

Installation
------------

The curated Kingsbury source is provided by matminer.  Install the optional
dependency with::

   pip install "dopingflow[corrections]"

Matminer 0.10.1 or newer in the 0.10 series is selected on Python 3.11 and
newer.  The compatible 0.9 series is selected on Python 3.10.  Matminer owns
the download and local cache of the dataset.  The fitting stage also writes a
project-local snapshot of the exact accepted records, so a completed fit does
not depend on a later network lookup for provenance.

Configuration
-------------

The conservative default uses one ordinary-oxide oxygen term::

   [energy_correction]
   enabled = true
   experimental_source = "kingsbury"
   calibration_manifest = "reference_structures/corrections/calibration_manifest.csv"
   correction_terms = ["oxide"]
   exclude_polyanions = [
       "SO4", "SO3", "CO3", "NO3", "NO2", "OCl3", "ClO3", "ClO4",
       "HO", "ClO", "SeO3", "TiO3", "TiO4", "WO4", "SiO3", "SiO4",
       "Si2O5", "PO3", "PO4", "P2O7",
   ]

   max_relative_experimental_uncertainty = 0.10
   max_calculated_e_above_hull_eV_per_atom = 0.10
   allow_phase_mismatch = false
   allow_legacy_candidate_provenance = false
   allow_element_terms = false
   reuse_fitted = true

   min_degrees_of_freedom = 1
   min_term_support = 2
   max_condition_number = 1.0e8
   poor_fit_rmse_warning_eV_per_atom = 0.20

The legacy-compatible default is ``model_family = "manual"`` with
``calibration_selection = "manifest"``. Automatic family selection and
phase-resolved calibration expansion are explicit opt-ins, for example::

   [energy_correction]
   enabled = true
   experimental_source = "kingsbury"
   correction_terms = ["oxide"]

   model_family = "auto"
   m1_elements = "workflow"
   calibration_selection = "phase_resolved"
   auto_fetch_phase_structures = true
   optimade_base_url = "https://optimade.materialsproject.org/v1"

   min_element_compounds = 3
   min_element_stoichiometries = 3
   min_cv_improvement_eV_per_atom = 0.01
   require_cv_one_standard_error = true

Run the optional fit after reference construction::

   dopingflow refs-build -c input.toml
   dopingflow corrections-fit -c input.toml

``dopingflow run-all`` places the optional ``corrections`` stage immediately
after ``refs``.  When ``enabled = false`` or the table is absent, this stage is
a no-op and the legacy raw workflow is unchanged.

Correction fitting uses the ``[references]`` backend/model/task. Candidate
energies used later by formation and phase-diagram analysis must come from the
same ``[relax]`` backend/model/task and the same resolved package/checkpoint
provenance. Align those sections before running. A package upgrade or local
checkpoint-content change requires rebuilding references, refitting/reusing the
new signature-specific correction, and rerunning stale candidate relaxations.
For registry-hosted model names whose resolved weight file is not exposed by
the backend, provenance is limited to the model identifier plus package
version; pin the backend artifact externally when byte-level reproducibility is
required.

Existing candidates from older dopingflow versions
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

Correction application normally requires the package version and a stored hash
binding each relaxed POSCAR to its evaluated energy. Older candidate metadata
may predate those fields even when it records the correct backend, model, task,
optimizer, force tolerance, step limit, convergence, and energy. Repeating the
physical relaxation is not required merely to evaluate a composition-linear
correction. An explicit compatibility mode is available::

   allow_legacy_candidate_provenance = true

This mode still rejects every known mismatch, an unconverged result, a missing
energy or POSCAR, and a changed POSCAR when an original hash exists. For fields
that were never recorded, it adopts the current POSCAR and existing energy
without claiming that their original byte-level binding or historical package
version has been proven. Formation databases and phase-diagram CSVs label the
result ``legacy_explicitly_accepted`` and serialize every assumption. CPU versus
GPU execution is recorded separately and is not treated as a different
correction Hamiltonian. Keep this option false for newly generated data.

Experimental sources
--------------------

Curated Kingsbury data
~~~~~~~~~~~~~~~~~~~~~~

The default source is matminer's
``expt_formation_enthalpy_kingsbury`` dataset.  The verified fields are:

``formula``
   Chemical formula.

``expt_form_e``
   Experimental standard formation enthalpy at 298 K, in eV/atom.

``uncertainty``
   Reported experimental uncertainty, in eV/atom.

``phaseinfo``
   Reported crystal/phase information.

``reference``
   Original experimental source.

``likely_mpid``
   A likely Materials Project association.  It is provenance, not proof of
   polymorph identity.

The inspected matminer 0.10.1 artifact contains 2,135 rows and has compressed
dataset SHA-256
``a2d2ced98d40349abd2041f169d9ed9c7f49453e86a77f82cab8c61c70dcb7ca``.
Matminer validates its cached artifact; dopingflow additionally fingerprints
the normalized experimental records entering each fit and saves the exact
accepted snapshot.

The normal user does not need to create an experimental CSV.

Kingsbury plus custom records
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

Use::

   experimental_source = "kingsbury+custom"
   experimental_data = "path/to/custom_formation_enthalpies.csv"

Custom records override only an unambiguous reduced-formula, phase, and
temperature match; when a custom ``likely_mpid`` is supplied it must match as
well. A blank-phase record supplements rather than silently replacing a
phase-specific curated row. Ambiguous polymorph matches are rejected instead
of selecting a row arbitrarily.

Fully custom records
~~~~~~~~~~~~~~~~~~~~

Use::

   experimental_source = "custom"
   experimental_data = "path/to/custom_formation_enthalpies.csv"

The CSV requires these columns:

.. code-block:: text

   formula,formation_enthalpy,uncertainty,phase,temperature,units,source

Optional columns are ``likely_mpid``, ``doi``, ``reference_id``, and ``notes``.
Accepted units are explicit variants of ``eV/atom`` and
``eV/formula_unit``.  A bare ``eV`` value is rejected as ambiguous.  Missing
uncertainty is an empty value, never zero statistical error. The current fit
targets standard formation enthalpy near 298 K; nonstandard or ambiguous custom
temperatures are rejected instead of mixed silently.

Units and uncertainty
---------------------

The curated values are converted from eV/atom to eV/reduced-formula unit.  If
the reduced formula has :math:`N` atoms,

.. math::

   H_{f,i}^{\mathrm{FU}} = N_i H_{f,i}^{\mathrm{atom}},\qquad
   \sigma_i^{\mathrm{FU}} = N_i \sigma_i^{\mathrm{atom}}.

For example, SnO2 has three atoms per reduced formula unit.  Both its measured
value and uncertainty are multiplied by three before fitting.

The backend values are relaxed static ML energies, whereas the measurements
are standard formation enthalpies near 298 K, following the empirical
calibration convention of the cited framework. The fitted coefficient can
therefore absorb systematic static-energy and standard-state differences; it
is not an explicit phonon, configurational-entropy, or finite-temperature Gibbs
free-energy calculation and must not be interpreted as one.

Missing, zero, or non-finite uncertainties do not receive infinite weight.
After all formula, phase, stability, elemental-reference, and applicability
filters have run, missing uncertainties are imputed using the mean positive
per-atom uncertainty of the records that actually enter the fit.  The saved
snapshot marks each value as ``reported`` or ``imputed_mean``.

Calibration data are separate from competing phases
-----------------------------------------------------

Three inventories have different roles:

* the experimental database supplies measured formation enthalpies;
* the calibration manifest identifies phase-matched structures/energies used
  to fit coefficients;
* ``[references].oxides_ref`` supplies chemical-potential references and/or
  competing phase-diagram entries.

An oxide in ``oxides_ref`` is never silently added to the fit.  Conversely, a
calibration compound need not be a competing phase in the final diagram.

Calibration manifest
--------------------

The manifest contains one calculated polymorph per reduced formula.  Its
minimum practical schema is:

.. code-block:: text

   formula,phase,space_group,structure_path,include,
   energy_total_eV,e_above_hull_eV_per_atom,e_above_hull_provenance,
   e_above_hull_backend,e_above_hull_model,e_above_hull_task,
   e_above_hull_backend_version,e_above_hull_calculation_settings_hash,
   oxide_type,structure_id,backend,model,task,backend_version,
   calculation_settings,calculation_settings_hash,converged

``structure_path`` is always required.  The structure composition is checked
against ``formula`` and its detected space group is recorded.  If
``space_group`` is supplied it must equal the source structure's detected
space group.  Formula or ``likely_mpid`` alone is insufficient to establish a
phase match.

Phase labels are intentionally strict. Empty or generic labels such as
``solid``, ``crystalline``, ``unknown``, or ``N/A`` do not match an
experimental polymorph. A crystal-system or space-group label must also agree
with the supplied structure. ``allow_phase_mismatch = true`` can admit only an
otherwise unambiguous formula match and records that expert override in the
fit artifacts.

If ``energy_total_eV`` is empty, the structure is relaxed and evaluated using
the ``[references]`` backend/model/task and relaxation infrastructure.  Source
and relaxed space groups are compared; reconstructive phase changes require
review and are not admitted automatically.

If ``energy_total_eV`` is supplied, ``backend``, ``model``, ``task``,
``backend_version``, and ``calculation_settings`` are mandatory provenance.
The row must also declare ``converged = true``; an unconverged or unspecified
pre-relaxed geometry is not accepted as calibration evidence.
The backend/model/task and backend package version must match the active
``[references]`` calculation exactly. ``calculation_settings_hash`` must equal
the hash of the active resolved reference-relaxation signature; descriptive
``calculation_settings`` text alone is insufficient. A supplied
energy-above-hull value also requires non-empty ``e_above_hull_provenance``.
Its backend/model/task, installed backend version, and calculation-settings
hash columns must exactly match ``[references]``. Materials Project hull values
must not be presented as same-backend ML hull values.

The default 0.10 eV/atom hull limit is active when the configuration key is
omitted. Set ``max_calculated_e_above_hull_eV_per_atom = false`` explicitly to
disable this filter; doing so removes an instability safeguard and should be a
deliberate expert choice.

Phase mismatches are excluded by default.  ``allow_phase_mismatch = true`` is
an explicit expert override and the mismatch remains in the calibration
snapshot/report.

Phase-resolved calibration expansion
-------------------------------------

``calibration_selection = "phase_resolved"`` derives a project-wide target
scope from the non-oxygen host and every configured dopant. It considers all
experimental records in that scope that are strict, non-generic, ordinary
oxides: each record must contain O, contain no element outside the target
host/dopant chemistry, and carry
specific phase information, and have a valid curated ``likely_mpid``. Known
peroxide/superoxide formulas and structures classified with a non-ordinary
oxygen environment are excluded. The accepted and rejected records, reasons,
coverage, and target scope are written to
``calibration_expansion_snapshot.json``.

An exact phase-matched structure in ``calibration_manifest`` is used when
available. Otherwise, ``auto_fetch_phase_structures = true`` retrieves the exact
``likely_mpid`` structure from ``optimade_base_url``. The default is the
Materials Project OPTIMADE endpoint. Both the canonical response and generated
POSCAR are stored under ``phase_structures/`` with SHA-256 hashes. This cache is
immutable: a complete matching pair is validated and reused, while a partial,
changed, composition-mismatched, or identifier-mismatched entry stops the fit
instead of being overwritten. If automatic fetching is false, every selected
record must have an exact phase-verified manifest structure. Failure to
materialize any selected record stops the expansion rather than silently
shrinking it.

OPTIMADE supplies structure geometry only. It does not supply a correction
energy or a hull value. Each newly materialized calibration structure is
relaxed and evaluated with the active ``[references]`` backend/model/task.
Missing calibration hull values are then constructed from same-backend
chemical-system energies, compatible entries in ``reference_energies.json``, and the
elemental terminals before the configured hull filter is applied. Materials
Project DFT energies or hull distances are never presented as ML-backend
evidence.

This expansion affects calibration structures only. ``corrections-fit`` does
not fetch, alter, or rerelax a doped workflow candidate; downstream stages apply
the selected composition-linear model to existing, provenance-compatible raw
energies. A real backend/package/checkpoint change still requires the normal
reference and candidate recalculation described above.

Correction basis
----------------

The default ``["oxide"]`` basis applies a coefficient per O atom classified
as an ordinary oxide.  Oxygen environments are detected from the actual
structure with pymatgen's oxide classifier and a recorded relative cutoff of
1.05.  A declared manifest ``oxide_type`` must agree with the structure-derived
classification.

``peroxide`` and ``superoxide`` terms are available only when named explicitly
and require their own phase-matched calibration coverage.  If a target requires
an oxygen-environment term absent from the fitted model, correction fails
instead of silently applying an ordinary-oxide value.

Model families
~~~~~~~~~~~~~~

The nested families requested by the workflow are

.. math::

   C_{M0}=\beta_{O}N_{O},

and

.. math::

   C_{M1}=\beta_{O}N_{O}+\sum_{M\in\mathcal{S}}\beta_M N_M,

where :math:`\mathcal{S}` contains only workflow cations that pass the
independent coverage and validation gates below. The :math:`N_M` features are
evaluated only for ordinary oxides.

``model_family`` accepts four values:

``manual``
   Fits exactly the user-provided ``correction_terms`` and preserves the
   previous behavior. This is the default.

``m0``
   Forces the conservative M0 family: one ordinary-oxide correction per O atom.
   The automatic-family path currently requires ``correction_terms = ["oxide"]``.

``m1``
   Forces an admissible M1 family: M0 plus one or more independently supported
   ``oxide_cation:<Element>`` terms. It fails if no valid M1 candidate remains.

``auto``
   Compares the admissible M0 and M1 candidates on identical leave-one-out
   observations and publishes M1 only when it clears every configured gate.
   Otherwise it deterministically falls back to M0; equality also goes to M0.

With ``m1_elements = "workflow"``, candidate M1 elements are the non-oxygen
host and dopants inferred from ``[references].host`` and ``[doping]``. An array
such as ``m1_elements = ["Sn", "Sb"]`` explicitly sets the candidate M1
elements, but phase-resolved discovery remains scoped to the complete workflow
host-and-dopant inventory. Elements without enough independent calibration
support are excluded individually and reported; they are never assigned a
guessed or zero coefficient.

An ``oxide_cation:X`` term is deliberately narrower than the manual
``element:X`` term. It counts X only when the actual target structure is
classified as an ordinary oxide, and is zero for elemental and non-oxide
entries. It is not a local coordination, oxidation-state, or dopant-site
descriptor. M1 therefore remains a composition-linear oxide model, not a model
of a particular doped supercell.

Automatic-family coverage requires at least ``min_element_compounds``
independent formula/oxygen-ratio pairs for each admitted cation (minimum 3;
default 3) and ``min_element_stoichiometries`` distinct O/cation ratios (minimum
2; default 3). M1 must also remain full-rank and well-conditioned and must not
worsen the cation-specific leave-one-out RMSE. For ``auto`` to publish M1, its
overall leave-one-out RMSE must improve on M0 by strictly more than
``min_cv_improvement_eV_per_atom`` (default 0.01 eV/atom). When
``require_cv_one_standard_error = true`` (default), the paired improvement in
squared leave-one-out loss must additionally exceed one standard error.
Automatic selection requires at least eight independent ordinary-oxide
formulas and caps the effective condition-number threshold at ``1e4`` even if
the broader manual-fit limit is larger. Forced ``m0`` and ``m1`` modes still run and record these admission
diagnostics. Forced ``m1`` may override the final automatic RMSE or
one-standard-error preference, but it cannot create an M1 candidate that fails
coverage, rank, conditioning, or cation-specific non-worsening requirements.

Advanced ``element:<symbol>`` terms use a FERE-like composition term in every
non-elemental compound containing that element.  They are disabled unless
``allow_element_terms = true`` is also set.  This is an explicit scientific
assumption, not an automatically inferred dopant coefficient.  Such a basis
must be full-rank, well-conditioned, independently validated, and justified for
the ML backend.  In particular:

* the paper's transition-metal MP terms repair GGA/GGA+U mixing and are not a
  default for a uniform ML backend;
* the paper's ``Sb`` term describes Sb as an anion in antimonides, not Sb as a
  cation dopant in an oxide;
* no Sn, Sb, Ti, or other coefficient is created merely because that element
  appears in a doping request.

The default ``exclude_polyanions`` list rejects sulfate, sulfite, carbonate,
nitrate/nitrite, chlorate/perchlorate, hydroxyl-like, selenite, titanate,
tungstate, silicate, and phosphate formula tokens before fitting. These oxygen
chemistries are outside the conservative ordinary-oxide basis. Clearing or
changing the list is supported, but it is a scientific model change and
requires calibration coverage and validation appropriate to the added terms.
The policy is serialized with the model and the same excluded formula tokens
are rejected at application time, so a competing polyanion phase cannot enter a
hull with an ordinary-oxide correction silently extrapolated onto it.

Fit and validation
------------------

For calibration compound :math:`i`,

.. math::

   h_i^{\mathrm{raw}} = E_i^{\mathrm{ML}} - \sum_e n_{ie}\mu_e^{\mathrm{ML}},

.. math::

   d_i = h_i^{\mathrm{expt}} - h_i^{\mathrm{raw}}.

The coefficient vector :math:`\beta` minimizes

.. math::

   \sum_i \left(\frac{\mathbf{x}_i^T\beta-d_i}{\sigma_i}\right)^2.

All coefficients are fitted simultaneously.  The implementation requires full
weighted-design rank, at least the configured residual degrees of freedom, at
least ``min_term_support`` accepted compounds with nonzero support for every
term, and a condition number below ``max_condition_number`` (default
``1e8``).  It stores the full
parameter covariance

.. math::

   C_\beta=(X^T W X)^{-1},

including off-diagonal correlations.  The fit report contains residual RMSE,
MAE, maximum residual, weighted metrics, leave-one-out diagnostics, and
leave-element-family-out diagnostics where the remaining design stays
identifiable.

For M0/M1 modes, ``correction_model_selection.json`` additionally records both
candidate scores, coverage and exclusion reasons, thresholds, the automatic and
published families, and a selection-run hash. Full candidate parameters and fit
reports are retained in ``candidate_models/``. The selected model alone is
published as ``correction_parameters.json`` and is the only model consumed by
formation and phase-diagram stages.

Residual and cross-validation errors are model-validation diagnostics.  They
are not treated as experimental uncertainty or automatically combined with the
coefficient covariance.

When any fit or cross-validation RMSE exceeds
``poor_fit_rmse_warning_eV_per_atom``, the model is still saved so that its
diagnostics remain reproducible, but ``correction_fit_report.json`` records a
``quality_warning``.  Treat that warning as a stop condition for production
formation energies and phase diagrams: broaden or revise the scientifically
justified calibration basis instead of discarding inconvenient compounds or
raising the warning threshold merely to accept the fit.

Application and balanced reactions
----------------------------------

For a target structure,

.. math::

   C_s=\mathbf{x}_s^T\hat\beta,\qquad
   E_s^{\mathrm{corrected}}=E_s^{\mathrm{raw}}+C_s.

For a balanced reaction, the combined feature vector is

.. math::

   \mathbf{q}=\sum_j \nu_j\mathbf{x}_j,

and therefore

.. math::

   \Delta E^{\mathrm{corrected}} = \Delta E^{\mathrm{raw}}
   +\mathbf{q}^T\hat\beta,

.. math::

   u(\Delta C)=\sqrt{\mathbf{q}^T C_\beta\mathbf{q}}.

This reaction-vector calculation retains the exact parameter correlations.
Adding separate phase uncertainties in quadrature would incorrectly discard
correlation and cancellation.

Elemental terminal entries, including the O2 reference, receive no compound
correction.  This is consistent with the fitted formation-energy convention
and avoids double counting.

Scientific consequences of the conservative oxide-only model
--------------------------------------------------------------

An ordinary-oxide correction is composition-linear.  It does not represent a
local dopant or vacancy environment.  Consequently:

* a metal-reference substitution reaction with unchanged oxygen count has a
  zero reaction correction;
* isocompositional candidate ordering cannot change;
* oxide-to-oxide decompositions that retain all oxygen in compounds cancel;
* redox reactions exchanging oxygen with uncorrected elemental O2 can shift;
* an oxygen-vacancy reaction has a nonzero oxygen reaction vector, but the
  current vacancy workflow does not yet publish corrected vacancy
  thermodynamics automatically.

For SnO2:Sb or SnO2:Sb+Ti, the default correction may therefore cancel in the
substitution energy.  This is a scientifically meaningful zero, not a missing
calculation.  A dopant-specific shift requires a separately justified,
identifiable, and validated correction basis; it must not be invented merely
to force a numerical change.

An admitted M1 oxide-cation term can make that balanced substitution correction
nonzero, but only because independent ordinary-oxide calibration data
passed the coverage and cross-validation gates. It remains composition-linear
and cannot distinguish dopant arrangements with the same composition.

Corrected phase diagrams
------------------------

The full ``phase-diagram`` stage builds two complete entry sets:

#. raw references and candidates, followed by a raw ``PhaseDiagram``;
#. corrected copies of every applicable non-elemental reference and candidate,
   followed by a new corrected ``PhaseDiagram``.

Corrected energy above hull is evaluated only on the second hull.  It is never
computed by adding a correction to raw energy above hull.  If any participating
non-elemental entry lacks a structure, has an incompatible backend, or needs an
unfitted environment term, the corrected diagram for that system is rejected.
There is no partially corrected/raw hull.

The specialized one-dimensional ``alloy-hull`` stage remains a raw diagnostic;
corrected multicomponent stability is reported by ``phase-diagram``.

Artifacts and cache compatibility
---------------------------------

Artifacts are stored under a backend/model/task-specific directory whose final
10 hexadecimal characters hash the complete resolved backend signature,
including package/checkpoint provenance::

   reference_structures/corrections/<backend-model-task>-<signature-hash>/
       correction_parameters.json
       correction_fit_report.json
       correction_metadata.json
       experimental_calibration_used.json
       calibration_calculated_energies.json
       calibration_rejected.json
       relaxed_calibration/
       correction_model_selection.json        # M0/M1 modes
       candidate_models/                       # M0/M1 modes
           m0.json
           m0_fit_report.json
           m1.json                             # when admissible
           m1_fit_report.json                  # when admissible
       calibration_expansion_snapshot.json    # phase_resolved mode
       phase_structures/                       # fetched OPTIMADE inputs
           mp-<id>.optimade.json
           mp-<id>.POSCAR

The model records the backend, model, task, relaxation settings, terms,
calibration records, units, dataset/version, fit-input hash, coefficients, full
covariance, and a fit ID. M0/M1 models additionally bind the selected family,
target elements, selection-run hash, selection-report hash, and expansion
snapshot hash. Reuse requires these files and hashes to agree exactly.
Formation and phase-diagram analyses rebuild whenever correction is enabled,
while collection verifies the current fit and formation-input fingerprint
before publishing corrected rows.

Reference reuse and correction application also require positive convergence,
an identical relaxation signature for the host and every elemental, gas, and
compound reference, and matching hashes for every stored relaxed structure.
Editing a relaxed host, metal, O2, or oxide POSCAR after its energy was evaluated
invalidates the reference set and requires ``refs-build`` followed by
``corrections-fit``.

For example, a MACE ``small`` fit with no task is stored under a slug shaped
like ``mace-small-1a2b3c4d5e``. The hash, not the illustrative characters in
this example, distinguishes signatures that share the same visible
backend/model/task label but differ in resolved settings, package version, or
checkpoint content.

Output fields
-------------

Formation metadata and CSV output retain legacy ``E_form_*`` fields as raw and
add fields including:

* ``formation_energy_raw_eV_total``;
* ``energy_correction_eV_total``;
* ``correction_uncertainty_eV_total``;
* ``E_form_corrected_eV_total`` and normalized variants;
* method, dataset, backend, parameter set, fit ID, and selected model family.

Phase-diagram output retains the legacy raw aliases and adds:

* ``energy_raw_eV`` and ``energy_corrected_eV``;
* ``energy_above_hull_raw_eV_per_atom``;
* ``energy_above_hull_corrected_eV_per_atom``;
* corrected-minus-raw hull change, fixed-facet parameter shift, combined
  reaction vector, and full-covariance uncertainty;
* raw/corrected stability and decomposition;
* correction applicability and provenance.

Uncertainty categories remain separate.  Experimental measurement uncertainty,
coefficient/correction uncertainty, ML ensemble uncertainty (if a backend ever
provides it), and residual validation error are not combined without an
explicit statistical model.

Reference
---------

A. Wang, R. Kingsbury, M. McDermott, M. Horton, A. Jain, S. P. Ong,
S. Dwaraknath, and K. A. Persson, *A framework for quantifying uncertainty in
DFT energy corrections*, Scientific Reports **11**, 15496 (2021).
