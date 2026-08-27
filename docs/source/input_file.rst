.. _input_file_spec:

Input File Specification (input.toml)
=====================================

The workflow is fully controlled through a single TOML configuration file.

Each stage reads only the parameters relevant to it.

Execution Model
---------------

Each stage defines both:

- **what it does** (physics / workflow logic)
- **how it runs** (CPU / GPU / parallelization)

There is **no global hardware section**. Instead:

- ``[scan]`` controls screening
- ``[relax]`` controls structural relaxation
- ``[bandgap]`` controls band gap prediction

Each stage independently defines:

- ``device`` (cpu or cuda)
- ``gpu_id`` (for GPU execution)

This design gives full flexibility and avoids cross-stage conflicts.

Execution Behavior
------------------

+-----------+-------------------+--------------------------+
| Stage     | Device            | Strategy                 |
+===========+===================+==========================+
| scan      | CPU               | multiprocessing          |
+-----------+-------------------+--------------------------+
| relax     | CPU / GPU         | workers or single GPU    |
+-----------+-------------------+--------------------------+
| bandgap   | CPU               | multiprocessing          |
+-----------+-------------------+--------------------------+
| bandgap   | GPU               | batched inference        |
+-----------+-------------------+--------------------------+

This design allows each stage to control its own parallelization strategy
while sharing a consistent hardware configuration.

All paths are interpreted relative to the directory containing ``input.toml``.

The following sections are supported:

- ``[references]``
- ``[structure]``
- ``[doping]``
- ``[sequential]``
- ``[generate]``
- ``[scan]``
- ``[relax]``
- ``[filter]``
- ``[bandgap]``
- ``[energy_correction]`` (optional)
- ``[formation]``
- ``[database]`` (optional)
- ``[vacancies]`` (required only by the vacancy workflow)

Not all sections are required for every stage. Each stage reads only what it needs.

---------------------------------------------------------------------

[vacancies]
-----------

The complete oxygen-vacancy workflow is configured in exactly one flat section
and run with ``dopingflow vacancies -c input.toml``. Parallel arrays represent
mixed formal oxidation states without creating TOML subsections.

``host_species``, ``oxidation_state_elements``, and
``oxidation_state_values`` are required chemistry-specific inputs.
``parent_directory`` is additionally required when
``parent_source = "directory"``. All other vacancy parameters shown below have
runtime defaults; this example intentionally overrides the default
M3GNet/CPU calculator with MACE/CUDA.

::

   [vacancies]
   enabled = true
   parent_source = "selected_candidates"
   include_parent_reference = true
   skip_if_done = true
   resume = true
   count_mode = "all_reachable"
   host_species = "Sn"
   host_oxidation_state = 4
   vacancy_species = "O"
   vacancy_compensation_charge = 2
   oxidation_state_elements = ["Sb", "Nb"]
   oxidation_state_values = [[3, 5], [5]]
   extra_vacancies = 0
   max_vacancies_cap = 8
   symprec = 1.0e-3
   angle_tolerance = 5.0
   mapping_tolerance = 1.0
   enumeration_mode = "auto"
   max_exact_raw_configs = 300000
   max_exact_unique_configs = 100000
   sample_budget = 20000
   sample_batch_size = 256
   sample_patience = 4000
   sample_seed = 42
   sample_max_saved = 50000
   minimum_vacancy_distance = 0.0
   backend = "mace"
   model = "medium-mpa-0"
   task = ""
   device = "cuda"
   gpu_id = 0
   n_workers = 1
   tf_threads = 1
   omp_threads = 1
   chunksize = 25
   topk_per_vacancy_count = 15
   energy_normalization = "per_vacancy"
   optimizer = "bfgs"
   fmax = 0.05
   max_steps = 300
   relax_mode = "atoms"
   cell_filter = "frechet"
   static_thermodynamic_analysis = true
   oxygen_reference_mode = "reference_file"
   oxygen_reference_file = "reference_structures/reference_energies.json"
   oxygen_reference_structure = "reference_structures/gas/O2.POSCAR"
   oxygen_reference_relax = false
   allow_unverified_oxygen_reference = false
   delta_mu_O_min_eV = -3.0
   delta_mu_O_max_eV = 0.0
   delta_mu_O_points_eV = [0.0, -0.5, -1.0, -1.5, -2.0, -2.5, -3.0]
   thermodynamic_tolerance_eV = 1.0e-8
   static_energy_source = "relaxed_only"
   exclude_unconverged = true
   pressure_mapping = true
   temperatures_K = [300.0, 600.0, 900.0, 1200.0, 1500.0]
   standard_oxygen_pressure_bar = 1.0
   log10_pO2_min_bar = -30.0
   log10_pO2_max_bar = 1.0
   log10_pO2_step = 0.5
   oxygen_standard_state_mode = "nist_shomate"
   oxygen_standard_state_temperatures_K = []
   oxygen_standard_state_delta_mu_eV_per_O = []

The same resolved backend/model/task is used for parent energies, defective
single points, and every relaxation. See :doc:`methods/vacancies` for the full
algorithm, output layout, resume rules, and interpretation limits.

For a directory that already contains many composition subdirectories, use::

   parent_source = "directory"
   parent_directory = "Mn-Nb-Ta-Ce-Ru-In-Sn-Sb"

Each subdirectory is discovered through its own ``selected_candidates.txt``.
The directory path remains a flat key in ``[vacancies]``.

Static-lattice thermodynamic analysis is disabled by default for compatibility. When
enabled, all keys remain in this same flat table. ``oxygen_reference_mode`` is
``reference_file``, ``same_calculator``, ``explicit``, or ``none``. Explicit
mode requires ``mu_O_reference_eV`` per oxygen atom. Reference-file mode rejects
incompatible or unverifiable backend/model/task metadata unless the latter is
deliberately acknowledged with ``allow_unverified_oxygen_reference = true``.
``relaxed_only`` plus ``exclude_unconverged = true`` is the safe default energy
policy. ``oxygen_standard_state_mode = "nist_shomate"`` evaluates the published
piecewise NIST O2 equations continuously at any requested temperature from 100
to 6000 K. ``user_table`` accepts another convention; ``none`` is qualitative
relative to the standard pressure at the same temperature. See
:doc:`methods/vacancies` for equations, exact intervals, outputs, and limits.

---------------------------------------------------------------------

[references]
------------

Step 00 — Reference construction and relaxation.

This stage prepares **all thermodynamic reference structures** and writes:

::

   reference_structures/reference_energies.json

It performs:

- Relaxation of the host oxide unit cell
- Construction and relaxation of the host supercell
- Relaxation of metal reference phases (metal mode)
- Relaxation of oxide reference phases (oxide mode)
- Relaxation of O₂ gas (oxide mode)
- Storage of all relaxed POSCAR files for reuse

Common Parameters
~~~~~~~~~~~~~~~~~

The required chemistry inputs are ``host`` and ``supercell``. ``metal_ref`` is
required in metal mode; ``oxides_ref`` and a non-empty ``gas_ref`` are required
in oxide mode. Omitted execution settings safely default to
``backend = "m3gnet"``, ``model = "default"``, ``task = ""``,
``optimizer = "bfgs"``, ``device = "cpu"``, ``gpu_id = 0``,
``max_steps = 300``, and one TensorFlow/OpenMP thread. Backend-specific blank
model/task values are normalized to the compatible choices listed under
``[scan]``.

reference_mode (string, default: "metal")
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

Choose reference scheme:

- ``"metal"``
- ``"oxide"``

skip_if_done (boolean, default: true)
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

Skip reconstruction if JSON cache exists.

fmax (float, default: 0.02)
^^^^^^^^^^^^^^^^^^^^^^^^^^^

Force convergence criterion used in relaxation (eV/Å).

supercell (array of 3 integers, required)
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

Supercell used later for doping.
The host supercell is constructed and relaxed at this stage.

host (string, required)
^^^^^^^^^^^^^^^^^^^^^^^

Chemical formula of the host oxide (e.g. ``"SnO2"``).

host_dir (string, default: "reference_structures/oxides")
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

Directory containing ``<host>.POSCAR``.

Metal Reference Mode
~~~~~~~~~~~~~~~~~~~~

Used when ``reference_mode = "metal"``.

metal_ref (array of strings, required in metal mode)
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

List of metal element symbols used as reference phases.

metals_dir (string, default: "reference_structures/metals")
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

Directory containing ``<Element>.POSCAR`` files.

Example::

   reference_structures/metals/Sn.POSCAR
   reference_structures/metals/Sb.POSCAR

Oxide Reference Mode
~~~~~~~~~~~~~~~~~~~~

Used when ``reference_mode = "oxide"``.

oxides_ref (array of strings; required in oxide mode, optional in metal mode)
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

List of oxide formulas (e.g. ``"Sb2O5"``, ``"SnO"``, or ``"Sn3O4"``).
These structures are also relaxed when ``reference_mode = "metal"`` so they
can serve as competing phases in the phase diagram without changing the
formation-energy reference scheme.

With ``skip_if_done = true``, reference construction checks each source
POSCAR and the relaxation settings independently. Compatible cached results
are reused, while newly added or changed references are relaxed. Setting
``skip_if_done = false`` forces all host and reference relaxations to run.

oxides_dir (string, default: "reference_structures/oxides")
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

Directory containing oxide POSCAR files.

gas_ref (string, default: "O2")
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

Gas reference formula (typically ``"O2"``).

gas_dir (string, default: "reference_structures/gas")
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

Directory containing gas POSCAR file.

oxygen_mode (string, default: "O-rich")
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

Selects the oxygen thermodynamic convention:

- ``"O-rich"`` requires ``delta_mu_O_ev = 0.0``.
- ``"O-poor"`` permits ``delta_mu_O_ev <= 0.0``.

oxygen_reference_correction_ev (float, default: 0.0)
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

Empirical/electronic correction to the raw same-backend oxygen reference, in
eV per O atom. If the raw molecular energy is :math:`E_{O_2}^{raw}`, then

.. math::

   E_{O_2}^{ref} = E_{O_2}^{raw} + 2\,\Delta E_O^{ref}.

When ``[energy_correction].enabled = true``, this value must be zero because the
fitted formation-energy correction is already calibrated against the raw
same-backend oxygen reference. A second empirical oxygen shift would double
count an oxygen-linear correction.

delta_mu_O_ev (float, default: 0.0)
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

Physical thermodynamic oxygen chemical-potential shift relative to the O-rich
reference, in eV per O atom:

.. math::

   \mu_O = \frac{1}{2} E_{O_2}^{ref} + \Delta\mu_O.

This quantity may be used together with a fitted formation-energy correction.
It must be exactly zero in ``O-rich`` mode and zero or negative in ``O-poor``
mode.

Legacy ``muO_shift_ev``
^^^^^^^^^^^^^^^^^^^^^^^^

``muO_shift_ev`` is retained only as a migration aid. A zero value is harmless.
A non-zero legacy value is interpreted conservatively as the historical
empirical oxygen-reference shift when energy correction is disabled, and is
rejected when ``[energy_correction].enabled = true``. A non-zero legacy value
must not be mixed with either explicit new oxygen key. New input files should
use ``oxygen_reference_correction_ev`` and ``delta_mu_O_ev`` instead.

These ``[references]`` settings are distinct from the independent oxygen
chemical-potential grid and temperature/pressure controls in ``[vacancies]``.

---------------------------------------------------------------------

[energy_correction]
-------------------

Optional backend-specific formation-energy correction fitting. The section may
be omitted. ``enabled = false`` is a hard no-op and preserves the legacy raw
workflow. No pre-fitted MP2020 coefficient is applied.

Minimal curated configuration::

   [energy_correction]
   enabled = true
   experimental_source = "kingsbury"
   calibration_manifest = "reference_structures/corrections/calibration_manifest.csv"
   correction_terms = ["oxide"]

Automatic M0/M1 selection with complete phase-resolved calibration is an
explicit opt-in::

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

enabled (boolean, default: false)
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

Enables fitting/reuse, corrected balanced formation reactions, and an
independently rebuilt corrected full phase diagram. Raw values remain present.
The ``[references]`` and ``[relax]`` backend, model, and task must agree. The
stored package version and local-checkpoint hash are also part of the energy
provenance; rerun references, correction fitting, and stale candidate
relaxations after either changes.

experimental_source (string, default: "kingsbury")
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

One of ``"kingsbury"``, ``"kingsbury+custom"``, or ``"custom"``. The curated
source is matminer's ``expt_formation_enthalpy_kingsbury`` dataset.

experimental_data (string, conditionally required)
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

Custom experimental CSV used by ``custom`` or ``kingsbury+custom``. It requires
``formula``, ``formation_enthalpy``, ``uncertainty``, ``phase``, ``temperature``,
``units``, and ``source``. Units must explicitly be eV/atom or eV/formula unit.
The current model accepts standard-temperature values near 298 K and rejects a
custom table that would mix another temperature into that population.

dataset_cache_dir (string, optional)
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

Optional matminer dataset cache directory. Matminer's normal local cache is
used when omitted. Every fit also saves the exact accepted records in the
project correction directory.

model_family (string, default: "manual")
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

One of ``"manual"``, ``"m0"``, ``"m1"``, or ``"auto"``:

- ``manual`` fits exactly ``correction_terms`` and preserves the previous
  behavior.
- ``m0`` forces one ordinary-oxide coefficient per O atom.
- ``m1`` forces an admissible M0-plus-``oxide_cation:<Element>`` model and fails
  when no M1 candidate passes the support and validation gates.
- ``auto`` compares admissible M0 and M1 fits on identical leave-one-out
  observations and publishes M1 only when it clears every configured gate;
  otherwise it falls back deterministically to M0.

The non-manual modes require ``correction_terms = ["oxide"]``. They record the
automatic decision even when ``m0`` or ``m1`` forces the published family.

m1_elements (string or array, default: "workflow")
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

``"workflow"`` derives candidate M1 cations from the non-oxygen host and all
configured dopants. An explicit array such as ``["Sn", "Sb"]`` overrides the
candidate M1 elements. Elements without independent support are excluded and
reported rather than assigned guessed coefficients. The array does not narrow
the complete host-and-dopant scope used by phase-resolved discovery.

calibration_selection (string, default: "manifest")
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

``"manifest"`` uses only the explicitly listed calibration rows.
``"phase_resolved"`` considers every strict non-generic ordinary oxide
experimental record whose non-oxygen elements are all in the complete
host-and-dopant target scope. A selected
record must have a specific phase and valid curated ``likely_mpid``; known or
structure-classified peroxide/superoxide records are excluded. All selections,
rejections, coverage, and materialized structures are snapshotted.

auto_fetch_phase_structures (boolean, default: false)
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

Only valid with ``calibration_selection = "phase_resolved"``. When true, a
selected record without an exact phase-matched manifest structure is retrieved
by ``likely_mpid`` from OPTIMADE. The response and POSCAR form an immutable,
hash-validated cache under the backend-specific correction directory. When
false, all selected records must be materialized by the manifest. Missing,
partial, mismatched, or changed inputs fail explicitly. Fetching and relaxation
apply only to calibration structures; doped workflow candidates are not
rerelaxed.

optimade_base_url (string, default: Materials Project OPTIMADE v1)
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

Base endpoint used for automatic structure acquisition. The default is
``https://optimade.materialsproject.org/v1``; ``/structures/<likely_mpid>`` is
appended as needed. OPTIMADE contributes geometry only, never a DFT correction
coefficient, calculated formation energy, or hull distance.

calibration_manifest (string, default shown above)
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

CSV containing the phase-matched calibration structures and same-backend
energies. This is deliberately separate from ``[references].oxides_ref``.
Formula, actual structure composition, phase/space-group information, energy
provenance, and same-backend energy-above-hull provenance are validated.
Generic labels such as ``solid``, ``crystalline``, or an empty phase do not
constitute a phase match unless the explicit ``allow_phase_mismatch`` expert
override is used for an unambiguous formula. The file is required in
``manifest`` mode. It may be absent in ``phase_resolved`` mode only when every
selected structure can be acquired automatically; otherwise it supplies exact
phase-verified local structures.

correction_terms (array of strings, default: ["oxide"])
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

Supported oxygen-environment terms are ``oxide``, ``peroxide``, and
``superoxide``. Each named term must be identifiable from the accepted
calibration set. Advanced ``element:<symbol>`` composition terms are rejected
unless ``allow_element_terms = true`` is also set. Explicit element or
``oxide_cation`` terms are manual-mode controls; M0/M1 derives scoped
``oxide_cation`` terms from admitted calibration evidence.

allow_element_terms (boolean, default: false)
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

Explicit expert acceptance of a FERE-like term applied to every non-elemental
compound containing the named element. This is not an automatically inferred
dopant correction and requires separate scientific justification/validation.

exclude_polyanions (array of strings, conservative list by default)
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

Formula tokens excluded from calibration before fitting because sulfate,
carbonate, nitrate, phosphate, silicate, and related oxygen chemistries are not
represented reliably by the simple ordinary-oxide basis. Set ``[]`` or change
the list only after validating a correction basis that covers those compounds.

max_relative_experimental_uncertainty (float, default: 0.10)
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

Rejects measurements whose reported relative uncertainty exceeds the limit.
Missing uncertainty is imputed from the final accepted calibration population,
never assigned zero.

max_calculated_e_above_hull_eV_per_atom (float, default: 0.10)
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

Rejects unstable calibration polymorphs. The manifest must give explicit
same-backend provenance plus matching ``e_above_hull_backend``,
``e_above_hull_model``, ``e_above_hull_task``,
``e_above_hull_backend_version``, and
``e_above_hull_calculation_settings_hash`` fields. Set the option
explicitly to ``false`` only after deliberately choosing not to apply this
filter. Omitting the key does **not** disable it; omission selects the default
limit of 0.10 eV/atom. For materialized phase-resolved rows without a supplied
hull value, dopingflow builds the chemical-system hull from same-backend relaxed
calibration energies, compatible ``reference_energies.json`` entries, and the
elemental terminals before applying this filter. Materials Project hull values
are not treated as same-backend ML values.

allow_phase_mismatch (boolean, default: false)
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

Expert override for a reviewed experimental/calculated phase mismatch. The
mismatch remains recorded. Formula and ``likely_mpid`` alone do not establish
polymorph identity.

allow_legacy_candidate_provenance (boolean, default: false)
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

Explicitly accepts existing candidate energies from older dopingflow metadata
that lacks the original backend-package and relaxed-POSCAR hash fields. Known
backend/model/task, optimizer, force tolerance, maximum steps, positive
convergence, energy, and POSCAR availability are still required and checked.
All unprovable assumptions are written to corrected formation and phase-diagram
outputs. This option does not alter or rerelax the structure.

reuse_fitted (boolean, default: true)
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

Reuses a model only when its backend signature, terms, calibration inputs,
dataset/filter settings, and fit-input hash match exactly.

min_degrees_of_freedom (integer, default: 1)
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

Minimum number of accepted compounds beyond the number of fitted terms.

min_term_support (integer, default: 2)
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

Minimum number of accepted calibration compounds with nonzero support for
each fitted term. This is enforced in addition to full-rank and residual-
degrees-of-freedom checks.

min_element_compounds (integer, default: 3; minimum: 3)
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

Minimum independent formula/oxygen-ratio support required before an M1
``oxide_cation:<Element>`` term is admitted.

min_element_stoichiometries (integer, default: 3; minimum: 2)
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

Minimum number of distinct O/cation ratios required for each admitted M1
element.

min_cv_improvement_eV_per_atom (float, default: 0.01)
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

M1 must improve identical-observation leave-one-out RMSE over M0 by strictly
more than this non-negative value. An exact threshold tie selects M0.

require_cv_one_standard_error (boolean, default: true)
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

When true, M1's paired improvement in squared leave-one-out loss must also
exceed one standard error. Failure selects the parsimonious M0 model in
``auto`` mode; forced ``m1`` still cannot bypass M1 admission and identifiability
requirements.

max_condition_number (float, default: 1e8)
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

Maximum allowed weighted-design condition number. A rank-deficient or
ill-conditioned correction basis fails rather than overfitting. Automatic
M0/M1 selection uses the stricter of this value and ``1e4``.

poor_fit_rmse_warning_eV_per_atom (float, default: 0.20)
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

Adds a prominent fit-quality warning without conflating residual error with
coefficient uncertainty. The diagnostic model is saved, but a
``quality_warning`` in ``correction_fit_report.json`` should be treated as a
stop condition for production correction application.

See :doc:`methods/energy_corrections` for the manifest schema, equations,
outputs, phase-diagram consistency rules, and scientific limitations.

---------------------------------------------------------------------

[structure]
-----------

Defines workflow I/O only.

outdir (string, default: "random_structures")
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

Directory where generated composition folders are written.

This directory becomes the root for:

- Step 01 outputs
- Step 02–06 per-composition subfolders

---------------------------------------------------------------------

[doping]
--------

Defines substitutional doping behavior.

host_species (string, required)
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

Element symbol of the host species to be substituted.

mode (string, default: "explicit")
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

Defines doping mode:

- ``"explicit"`` — user provides exact compositions.
- ``"enumerate"`` — workflow constructs compositions combinatorially.

Explicit Mode
~~~~~~~~~~~~~

compositions (array of tables, required in explicit mode)
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

List of dictionaries:

::

   compositions = [
       { Sb = 5 },
       { Sb = 5, Zr = 5 }
   ]

Percentages are defined relative to host sites.

Enumerate Mode
~~~~~~~~~~~~~~

dopants (array of strings, required in enumerate mode)
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

Allowed dopant elements.

must_include (array of strings, default: [])
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

Dopants that must appear in each composition.

max_dopants_total (integer, default: 2)
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

Maximum number of distinct dopants per structure.

allowed_totals (array of floats, required in enumerate mode)
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

Allowed total dopant percentages.

levels (array of floats, required in enumerate mode)
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

Discrete concentration values per dopant.

---------------------------------------------------------------------

[sequential]
------------

Controls the gradual sequential doping workflow executed with:

::

   dopingflow sequential-run -c input.toml

When ``[energy_correction].enabled = true``, ``sequential-run`` fits or reuses
the correction model once before entering the composition loop. It does not
refit separately for every sequential step. The reference cache and calibration
inputs must therefore already be valid for the active backend signature.

outdir (string, default: "sequential_structures")
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

Directory where sequential step folders are written.

mode (string, default: "full")
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

Controls the sequential execution mode:

- ``"full"`` — run the full stepwise workflow:
  generate → scan → relax → filter → optional bandgap → formation → collect.

- ``"recompute_energies"`` — reuse existing relaxed sequential structures and
  rerun only formation energy and database collection. This is useful when the
  user wants to change the thermodynamic reference, for example from ``Sb2O3``
  to ``Sb2O5``, without regenerating or relaxing structures.

Example
~~~~~~~

::

   [sequential]
   outdir = "sequential_structures"
   mode = "full"

---------------------------------------------------------------------

[generate]
----------

Step 01 — Structure generation.

This step generates one doped structure per composition by substituting
host atoms inside the **relaxed host supercell** produced by ``refs-build``.

Important
~~~~~~~~~

- ``refs-build`` must be executed first.
- The relaxed host supercell is loaded from:

  ``reference_structures/reference_energies.json``

- No supercell is constructed in this step.

Parameters
~~~~~~~~~~

seed_base (integer, default: 0)
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

Base seed used for deterministic random substitution.

Each composition generates a stable hash-based seed.

poscar_order (array of strings, default: [])
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

Defines element ordering in written POSCAR files.

If empty, pymatgen default ordering is used.

Example::

   poscar_order = ["Zr", "Ti", "Sb", "Sn", "O"]

clean_outdir (boolean, default: true)
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

If true, existing output directory is deleted before writing new structures.

Output
~~~~~~

For each composition:

::

   <outdir>/<composition_tag>/
       POSCAR
       metadata.json

---------------------------------------------------------------------

[scan]
------

Step 02 — Dopant configuration prescreening using machine-learning interatomic potentials.

For each generated structure folder inside ``[structure].outdir``:

1. Generates doped configurations on the cation sublattice
2. Identifies symmetry-unique configurations
3. Evaluates single-point energies using the selected ML backend
4. Ranks configurations by energy
5. Keeps the lowest-energy ``topk`` candidates
6. Writes candidate folders and ranking files

Depending on the scan mode, configurations are obtained either by:

- exact symmetry-unique enumeration
- random symmetry-unique sampling

This step operates only on subfolders created in Step 01.

Parameters
~~~~~~~~~~

backend (string, default: "m3gnet")
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

Selects the ML model used for energy prediction.

Available options:

- ``"m3gnet"`` — TensorFlow-based universal interatomic potential
- ``"uma"`` — FAIR-Chem universal model (requires Hugging Face access)
- ``"mace"`` — MACE foundation models (MP, MPA, OMAT, MATPES, and MH)
- ``"grace"`` — GRACE graph neural network models (if installed)

Each backend has its own supported models and execution characteristics.

model (string, default: "default")
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

Specifies the pretrained model variant used by the selected backend.

An omitted, empty, or ``"default"`` value is normalized to a compatible model:
``default`` for M3GNet, ``uma-s-1p2`` for UMA, ``small`` for MACE, and
``GRACE-1L-OMAT`` for GRACE.

Behavior depends on backend:

- ``m3gnet``:
  - ``"default"`` → loads the standard M3GNet universal model

- ``uma``:
  - ``"uma-s-1p2"``
  - ``"uma-s-1p1"``
  - ``"uma-m-1p1"``

- ``mace``:

  - accepts every alias exposed by the installed MACE ``mace_mp`` calculator
  - common aliases include ``"small"``, ``"medium"``, ``"large"``,
    ``"medium-mpa-0"``, ``"small-omat-0"``, ``"medium-omat-0"``,
    ``"mace-matpes-pbe-0"``, ``"mace-matpes-r2scan-0"``, ``"mh-0"``,
    and ``"mh-1"``
  - a local ``.model``, ``.pt``, or ``.pth`` checkpoint path is also accepted
  - named model weights are downloaded and cached by MACE on first use

- ``grace``:
  - ``GRACE-1L-OMAT``
  - ``GRACE-1L-OMAT-M-base``
  - ``GRACE-1L-OMAT-M``
  - ``GRACE-1L-OMAT-L-base``
  - ``GRACE-1L-OMAT-L``
  - ``GRACE-2L-OMAT``
  - ``GRACE-2L-OMAT-M-base``
  - ``GRACE-2L-OMAT-M``
  - ``GRACE-2L-OMAT-L-base``
  - ``GRACE-2L-OMAT-L``
  - ``GRACE-1L-OAM``
  - ``GRACE-1L-OAM-M``
  - ``GRACE-1L-OAM-L``
  - ``GRACE-2L-OAM``
  - ``GRACE-2L-OAM-M``
  - ``GRACE-2L-OAM-L``
  - ``GRACE-1L-SMAX-L``
  - ``GRACE-1L-SMAX-OMAT-L``
  - ``GRACE-2L-SMAX-M``
  - ``GRACE-2L-SMAX-L``
  - ``GRACE-2L-SMAX-OMAT-M``
  - ``GRACE-2L-SMAX-OMAT-L``

task (string, default: "")
~~~~~~~~~~~~~~~~~~~~~~~~~~

Optional task specification (used only for certain backends).

- ``uma`` uses ``"omat"`` when ``task`` is omitted or empty. Supported values are:
  - ``"omat"``
  - ``"oc20"``
  - ``"oc22"``
  - ``"oc25"``
  - ``"omol"``
  - ``"odac"``
  - ``"omc"``

- ``mace`` uses ``task`` to select the training domain (the ``head``) of a
  multi-head model. For ``model = "mh-1"``, an omitted or empty value is
  resolved by dopingflow to the compatible ``omat_pbe`` head. Set a supported
  head explicitly, such as ``task = "matpes_r2scan"``, when that training
  domain and level of theory is desired.

- ``m3gnet``, ``grace``:
  - Not used → keep empty (``""``)

poscar_in (string, default: "POSCAR")
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

Filename inside each composition folder used as input.

topk (integer, default: 15)
~~~~~~~~~~~~~~~~~~~~~~~~~~~

Number of lowest-energy configurations retained.

symprec (float, default: 1e-3)
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

Tolerance used for symmetry detection in ``SpacegroupAnalyzer``.

mode (string, default: "auto")
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

Scan strategy.

Possible values:

- ``auto``  
  Automatically chooses exact enumeration for manageable problems and switches
  to sampling when the configuration space becomes too large.

- ``exact``  
  Forces full symmetry-unique enumeration of all configurations.

- ``sample``  
  Uses random symmetry-unique sampling instead of full enumeration.

max_enum (integer, default: 300000)
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

Maximum allowed number of raw combinatorial configurations in exact mode.

If this limit is exceeded and ``mode = "auto"``, the scan automatically switches
to sampling mode.

max_unique (integer, default: 100000)
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

Maximum allowed number of symmetry-unique configurations in exact mode.

Prevents excessive memory usage for very large configuration spaces.

device (string, default: "cpu")
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

Execution device:

- ``"cpu"``
- ``"cuda"``

Behavior depends on backend:

- ``m3gnet``:
  - GPU mode uses a single worker (TensorFlow limitation)

- ``uma``:
  - GPU supported via PyTorch

- ``mace``:
  - GPU supported and recommended for performance

- ``grace``:
  - GPU support depends on model implementation

n_workers (integer, default: 12)
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

Number of parallel worker processes.

- Used only when ``device = "cpu"``
- Some backends may internally limit parallelism:

  - ``m3gnet`` (GPU): forces single worker
  - ``mace``: typically runs efficiently in single-process mode
  
- Ignored when ``device = "cuda"`` (GPU mode uses a single worker)

chunksize (integer, default: 50)
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

Chunk size used in the multiprocessing pool.

Relevant only when ``device = "cpu"``.

gpu_id (integer, default: 0)
~~~~~~~~~~~~~~~~~~~~~~~~~~~~

GPU index used when ``device = "cuda"``.

anion_species (array of strings, default: ["O"])
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

Species excluded from substitutional enumeration.
Typically contains oxygen:

::

   anion_species = ["O"]

host_species (from [doping])
~~~~~~~~~~~~~~~~~~~~~~~~~~~~

Used to define the cation sublattice.
Must match the host element defined in ``[doping]``.

skip_if_done (boolean, default: true)
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

If true, skip a structure folder if ``ranking_scan.csv`` already exists.

Sampling parameters
~~~~~~~~~~~~~~~~~~~

Used when:

- ``mode = "sample"``
- or ``mode = "auto"`` selects sampling for large configuration spaces

sample_budget (integer, default: 20000)
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

Maximum number of random sampling attempts.

sample_batch_size (integer, default: 256)
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

Number of new symmetry-unique sampled configurations evaluated per batch.

sample_patience (integer, default: 4000)
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

Sampling stops after this many sampled configurations fail to improve the
current best candidate.

sample_seed (integer, default: 42)
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

Random seed used for reproducible sampling.

sample_max_saved (integer, default: 50000)
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

Maximum number of sampled canonical configurations stored to avoid duplicates.

Notes
~~~~~

- If ``device = "cuda"``, scan runs on a single GPU and ``n_workers`` is ignored.
- If ``device = "cpu"``, parallelization is controlled via ``n_workers`` and ``chunksize``.
- GPU mode is recommended for faster single-structure evaluation, while CPU mode scales better across many configurations.
- The scan backend can be selected via ``backend``.
- Different backends have different accuracy/speed trade-offs:

  - ``m3gnet``: stable, general-purpose
  - ``uma``: high-quality FAIR-Chem models (requires authentication)
  - ``mace``: fast and scalable foundation models
  - ``grace``: advanced GNN-based models (experimental)

- For large configuration spaces, sampling mode is recommended.

- GPU acceleration is backend-dependent and may not always scale with multiprocessing.

Output
~~~~~~

For each composition folder:

::

   ranking_scan.csv
   scan_summary.txt
   candidate_001/01_scan/POSCAR
   candidate_001/01_scan/meta.json
   ...

Each candidate folder contains:

- symmetry-unique configuration
- single-point energy (from selected backend)
- dopant site signature
- scan metadata

The following metadata is recorded:

- backend
- model
- task
- energy_sp_eV
- configuration details

---------------------------------------------------------------------

[relax]
-------

Step 03 — Structural relaxation using machine-learning interatomic potentials.

This stage relaxes the low-energy candidates selected in Step 02.
For each structure folder in ``[structure].outdir``, the workflow reads:

::

   candidate_*/01_scan/POSCAR

and writes relaxed structures and metadata to:

::

   candidate_*/02_relax/

Relaxation is performed using the selected ML backend together with an ASE optimizer.

Supported backends:

- ``"m3gnet"``
- ``"uma"``
- ``"mace"``
- ``"grace"``

Supported optimizers:

- ``"bfgs"``
- ``"lbfgs"``
- ``"fire"``
- ``"mdmin"``
- ``"quasinewton"``

Relaxation stops when either:

- the maximum atomic force drops below ``fmax``
- the number of optimizer steps reaches ``max_steps``

Parameters
~~~~~~~~~~

backend (string, default: "m3gnet")
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

Selects the ML backend used for structural relaxation.

Available options:

- ``"m3gnet"`` — TensorFlow-based universal interatomic potential
- ``"uma"`` — FAIR-Chem universal model (requires Hugging Face access)
- ``"mace"`` — MACE foundation models
- ``"grace"`` — GRACE graph neural network models (if installed)

model (string, default: "default")
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

Specifies the pretrained model variant used by the selected relaxation backend.

An omitted, empty, or ``"default"`` value is normalized to a compatible model:
``default`` for M3GNet, ``uma-s-1p2`` for UMA, ``small`` for MACE, and
``GRACE-1L-OMAT`` for GRACE.

Behavior depends on backend:

- ``m3gnet``:
  - ``"default"`` → loads the standard pretrained M3GNet model

- ``uma``:
  - ``"uma-s-1p2"``
  - ``"uma-s-1p1"``
  - ``"uma-m-1p1"``

- ``mace``:

  - accepts every alias exposed by the installed MACE ``mace_mp`` calculator
  - common aliases include ``"small"``, ``"medium"``, ``"large"``,
    ``"medium-mpa-0"``, ``"small-omat-0"``, ``"medium-omat-0"``,
    ``"mace-matpes-pbe-0"``, ``"mace-matpes-r2scan-0"``, ``"mh-0"``,
    and ``"mh-1"``
  - local ``.model``, ``.pt``, and ``.pth`` checkpoint paths are accepted

- ``grace``:
  - ``GRACE-1L-OMAT``
  - ``GRACE-1L-OMAT-M-base``
  - ``GRACE-1L-OMAT-M``
  - ``GRACE-1L-OMAT-L-base``
  - ``GRACE-1L-OMAT-L``
  - ``GRACE-2L-OMAT``
  - ``GRACE-2L-OMAT-M-base``
  - ``GRACE-2L-OMAT-M``
  - ``GRACE-2L-OMAT-L-base``
  - ``GRACE-2L-OMAT-L``
  - ``GRACE-1L-OAM``
  - ``GRACE-1L-OAM-M``
  - ``GRACE-1L-OAM-L``
  - ``GRACE-2L-OAM``
  - ``GRACE-2L-OAM-M``
  - ``GRACE-2L-OAM-L``
  - ``GRACE-1L-SMAX-L``
  - ``GRACE-1L-SMAX-OMAT-L``
  - ``GRACE-2L-SMAX-M``
  - ``GRACE-2L-SMAX-L``
  - ``GRACE-2L-SMAX-OMAT-M``
  - ``GRACE-2L-SMAX-OMAT-L``

task (string, default: "")
~~~~~~~~~~~~~~~~~~~~~~~~~~

Optional task specification used for ``uma`` and multi-head MACE models.

For UMA, an omitted or empty value resolves to ``"omat"``. Allowed values are:

- ``"omat"``
- ``"oc20"``
- ``"oc22"``
- ``"oc25"``
- ``"omol"``
- ``"odac"``
- ``"omc"``

For MACE, ``task`` is passed as the optional model ``head``. An omitted or empty
value uses a compatible dopingflow default; MACE-MH-1 resolves to
``omat_pbe``. For
``m3gnet`` and ``grace``, this parameter is ignored.

relax_mode (string, default: "atoms")
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

Controls which degrees of freedom are optimized.

Available options:

- ``"atoms"`` — relax atomic positions only.
- ``"full"`` — relax atomic positions, cell shape, and volume.
- ``"isotropic"`` — relax atomic positions with uniform cell scaling.
- ``"volume"`` — relax volume only while keeping fractional atomic positions fixed.
- ``"shape"`` — relax cell shape at nearly constant volume.
- ``"xy"`` — relax only the in-plane ``a`` and ``b`` cell lengths, keeping ``c`` and angles fixed.
- ``"cell_only"`` — relax the cell while keeping fractional atomic positions fixed.

cell_filter (string, default: "frechet")
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

ASE cell filter used when ``relax_mode`` involves cell relaxation.

Available options:

- ``"frechet"`` — recommended default.
- ``"unit"`` — simpler direct cell filter.
- ``"exp"`` — older exponential cell filter, mainly for compatibility.

optimizer (string, default: "bfgs")
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

ASE optimizer used during relaxation.

Available options:

- ``"bfgs"``
- ``"lbfgs"``
- ``"fire"``
- ``"mdmin"``
- ``"quasinewton"``

fmax (float, default: 0.05)
~~~~~~~~~~~~~~~~~~~~~~~~~~~

Maximum force convergence criterion (eV/Å).

Relaxation is considered converged when the maximum atomic force falls below this threshold.

max_steps (integer, default: 300)
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

Maximum number of optimizer steps.

If this limit is reached before the force threshold is satisfied, the relaxation stops and the final structure is still written to disk.

device (string, default: "cpu")
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

Execution device:

- ``"cpu"``
- ``"cuda"``

Behavior depends on backend and runtime environment.
GPU execution uses a single effective worker.

gpu_id (integer, default: 0)
~~~~~~~~~~~~~~~~~~~~~~~~~~~~

GPU index used when ``device = "cuda"``.

n_workers (integer, default: 6)
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

Number of parallel relaxation workers (one candidate per worker process).

Relevant mainly when ``device = "cpu"``.

tf_threads (integer, default: 1)
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

TensorFlow thread count per worker.

Mainly relevant for the ``m3gnet`` backend.
Keep small (typically 1) when using multiple workers.

omp_threads (integer, default: 1)
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

OpenMP thread count per worker.

Keep small to avoid CPU oversubscription.

skip_if_done (boolean, default: true)
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

Skip an entire composition folder if ``ranking_relax.csv`` already exists.

skip_candidate_if_done (boolean, default: true)
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

Skip an individual candidate if both of the following already exist:

::

   candidate_*/02_relax/meta.json
   candidate_*/02_relax/POSCAR

Notes
~~~~~

- If ``device = "cuda"``, relaxation uses a single effective worker and ``n_workers`` is ignored.
- If ``device = "cpu"``, parallelization is controlled via ``n_workers``.
- The relaxed ``POSCAR`` follows the species ordering defined by ``[generate].poscar_order``.
- If ``poscar_order`` is empty, the default pymatgen ordering is used.

Output
~~~~~~

For each candidate:

::

   candidate_###/02_relax/POSCAR
   candidate_###/02_relax/meta.json

For each structure folder:

::

   ranking_relax.csv

The metadata records:

- backend
- model
- task
- optimizer
- relaxed energy
- convergence target
- final maximum force
- optimizer step count
- convergence status
- walltime
- link to the original scan metadata

[filter]
--------

Step 04 — Candidate selection.

mode (string, default: "window")
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

- ``"window"``
- ``"topn"``

window_meV (float, default: 50.0)
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

Energy window above the lowest relaxed energy (in meV).

Candidates with:

    E_relaxed <= E_min + window_meV

are retained.

A value of 0 keeps only the lowest-energy structure.

If no candidate satisfies the filtering criteria, the workflow raises an error.

max_candidates (integer, default: 12)
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

Number of candidates kept when mode = ``"topn"``.

skip_if_done (boolean, default: true)
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

Skip filtering if output exists.

---------------------------------------------------------------------

[bandgap]
---------

Step 05 — Bandgap prediction using a local ALIGNN model.

Requires environment variable ``ALIGNN_MODEL_DIR`` pointing to
a local ALIGNN model directory.

enabled (boolean, default: true)
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

If false, ``sequential-run`` skips the bandgap stage. The final database is still
written, but ``bandgap_eV`` is left empty.

skip_if_done (boolean, default: true)
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

If true, previously computed bandgap results are reused.

Behavior:
- If ``candidate_*/03_band/meta.json`` already exists, the stored bandgap value is reused and prediction is skipped for that candidate.
- The summary CSV is rebuilt from existing metadata.
- This allows safe re-running of the workflow without recomputing already processed candidates.

If a candidate prediction fails:
- The error is recorded in ``candidate_*/03_band/meta.json``.
- The workflow continues with remaining candidates.
- Failed candidates appear in the summary CSV with ``NaN`` bandgap.

cutoff (float, default: 8.0)
~~~~~~~~~~~~~~~~~~~~~~~~~~~~

Neighbor cutoff radius (Å) used to construct the atomic graph
for ALIGNN inference. Must be > 0.

max_neighbors (integer, default: 12)
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

Maximum number of neighbors retained per atom when building the graph.
Must be > 0.

device (string, default: "cpu")
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

Execution device:

- ``"cpu"``
- ``"cuda"``

gpu_id (integer, default: 0)
~~~~~~~~~~~~~~~~~~~~~~~~~~~~

GPU index used when ``device = "cuda"``.

batch_size (integer, default: 32)
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

Batch size used for GPU inference.

n_workers (integer, default: 1)
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

Number of CPU workers used for parallel bandgap prediction.

Notes
~~~~~

- CPU mode → multiprocessing over structures using ``n_workers``
- GPU mode → batched inference using ``batch_size``
- ``n_workers`` is ignored when using GPU

---------------------------------------------------------------------

[formation]
-----------

Step 06 — Formation energy calculation.

Formation energies are computed using the chemical potentials written by
``refs-build`` in ``reference_structures/reference_energies.json``.

The reference scheme (metal or oxide) is automatically determined from
``[references].reference_mode`` and no additional user input is required here.

skip_if_done (boolean, default: true)
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

If true, skip formation calculation if ``formation_energies.csv`` already exists
in a composition folder.

normalize (string, default: "per_dopant")
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

Defines how the reported formation energy is normalized:

- ``"total"`` — total formation energy (eV)
- ``"per_dopant"`` — eV per substituted dopant atom
- ``"per_host"`` — eV per atom in the supercell

relative_enabled (boolean, default: false)
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

If true, write reference-specific relative formation and mixing-energy columns.
In oxide mode these values preserve the already tie-line-referenced per-cation
energies and include explicit oxide-endmember provenance.

endpoint_x (string or float, default: "auto")
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

Controls endpoint metadata and the compatibility calculation for older result
databases. ``"auto"`` uses the pure oxide endmember for new oxide-reference
results. A numeric value must lie in ``(0, 1]``.

Formation energies require:

- Successful execution of ``refs-build``
- ``reference_structures/reference_energies.json``
- Relaxed candidate structures in ``candidate_*/02_relax/``

Notes
~~~~~

- The same substitution formula is used for both reference schemes.
- In ``metal`` mode, elemental metal references define the chemical potentials.
- In ``oxide`` mode, chemical potentials are derived from oxide references
  and the chosen oxygen condition (``O-rich`` or ``O-poor``).
- The selected reference mode is stored in
  ``candidate_*/04_formation/meta.json`` for reproducibility.

---------------------------------------------------------------------

[database]
----------

Step 07 — Final database collection.

skip_if_done (boolean, default: true)
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

If true, do not overwrite existing ``results_database.csv``.

---------------------------------------------------------------------

[phase_diagram]
---------------

Step 09 — Per-chemical-system phase-diagram analysis.

skip_if_done (boolean, default: true)
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

If true and ``phase_diagram_results.csv`` exists, return the existing result
without rebuilding the per-system diagrams.

stable_threshold_eV_per_atom (float, default: 1e-8)
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

Non-negative energy-above-hull threshold used for the boolean ``stable`` field.
The step writes a combined CSV plus one CSV per exact chemical system under
``phase_diagrams/``.

---------------------------------------------------------------------

[vacancies]
-----------

The vacancy stage uses one flat ``[vacancies]`` table. Important controls are:

- ``host_species``, ``host_oxidation_state``, ``vacancy_species`` and
  ``vacancy_compensation_charge`` for the formal-charge search space.
- ``oxidation_state_elements`` / ``oxidation_state_values`` for dopants.
- ``enumeration_mode = "auto" | "exact" | "sample"`` and the exact/sampling
  limits for symmetry-distinct vacancy generation.
- ``backend``, ``model``, ``task`` and ``device`` for ML screening/relaxation.
- ``topk_per_vacancy_count``, ``optimizer``, ``fmax`` and ``max_steps`` for
  relaxation.
- ``static_thermodynamic_analysis = true`` to compare different oxygen contents.
- ``oxygen_reference_mode = "global" | "chemistry-specific" |
  "reference_file" | "same_calculator" | "explicit" | "none"``.
- ``oxygen_standard_state_mode = "nist_shomate" | "user_table" | "none"``
  together with ``temperatures_K`` and the ``pO2`` grid.
- ``solid_configurational_entropy = "none" | "ideal" | "configurational"``.
  The ``configurational`` option evaluates an explicit canonical partition
  function using exact symmetry-orbit degeneracies; it therefore requires exact
  vacancy enumeration.

Finite-temperature runs write ``vacancy_formation_free_energy.csv/json`` with
:math:`\Delta G_{vac}(T,p_{O_2})` for every vacancy count. See
:doc:`methods/vacancies` and :doc:`methods/oxygen_calibration` for the equations,
calibration rules and approximations.

---------------------------------------------------------------------

[surface]
---------

Optional surface generation and surface relaxation.

This stage constructs slab models from selected relaxed bulk candidates
and optionally relaxes those slabs using machine-learning interatomic potentials.

The stage reads candidate data from a database file:

::

   results_database.csv

For each selected candidate, the workflow loads the relaxed bulk structure from:

::

   candidate_*/02_relax/POSCAR

and generates surface slabs for the requested orientations and terminations.

Generated slabs are written to:

::

   [surface].outdir / composition_tag / candidate_### / hkl_* / term_*/

If enabled, slab relaxation is performed using the same backend abstraction
as in Step 03.

Candidate Selection
~~~~~~~~~~~~~~~~~~~

Selection is performed in two stages:

1. Restrict dataset by composition:

   - ``composition_tag``
   - or ``composition_tags``

2. Apply ``selection_mode`` inside the selected composition subset.

Available selection modes:

- ``"id"``
- ``"ids"``
- ``"rank_range"``
- ``"top_n"``
- ``"filters"``


Orientation Modes
~~~~~~~~~~~~~~~~~

orientation_mode (string)
~~~~~~~~~~~~~~~~~~~~~~~~~

- ``"explicit"`` — use Miller indices from ``miller_list``
- ``"automatic"`` — generate indices up to ``max_miller``

miller_list (list)
~~~~~~~~~~~~~~~~~~

List of Miller indices when using explicit mode.

Example:

::

   [[1, 0, 0], [1, 1, 0], [1, 1, 1]]

max_miller (integer)
~~~~~~~~~~~~~~~~~~~~

Maximum Miller index used in automatic mode.

max_orientations (integer)
~~~~~~~~~~~~~~~~~~~~~~~~~~

Maximum number of orientations kept in automatic mode.


Slab Construction
~~~~~~~~~~~~~~~~~

min_slab_size (float)
~~~~~~~~~~~~~~~~~~~~~

Minimum slab thickness.

min_vacuum_size (float)
~~~~~~~~~~~~~~~~~~~~~~~

Minimum vacuum thickness.

center_slab (boolean)
~~~~~~~~~~~~~~~~~~~~~

If true, centers the slab in the simulation cell.

in_unit_planes (boolean)
~~~~~~~~~~~~~~~~~~~~~~~~

Controls units of slab and vacuum thickness:

- ``true`` → values interpreted in crystallographic planes
- ``false`` → values interpreted in Å (recommended)

lll_reduce (boolean)
~~~~~~~~~~~~~~~~~~~~

Apply LLL lattice reduction.

primitive (boolean)
~~~~~~~~~~~~~~~~~~~

Build primitive slab if possible.

reorient_lattice (boolean)
~~~~~~~~~~~~~~~~~~~~~~~~~~

Align slab normal with the out-of-plane axis.

orthogonal_c (boolean)
~~~~~~~~~~~~~~~~~~~~~~

If true, enforces:

- ``c perpendicular to (a, b)``

This is strongly recommended for:

- clean POSCAR output
- stable relaxation
- correct slab thickness interpretation


Surface Terminations
~~~~~~~~~~~~~~~~~~~~

termination_mode (string)
~~~~~~~~~~~~~~~~~~~~~~~~~

- ``"all"`` — keep all generated terminations
- ``"first"`` — keep only the first termination

max_terminations_per_orientation (integer)
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

Maximum number of terminations retained per orientation.


Fixed Atoms
~~~~~~~~~~~

The surface stage can optionally fix atoms in the slab.

Fixed atoms are written in the same ``POSCAR`` file using selective dynamics:

- fixed → ``F F F``
- free → ``T T T``

fix_atoms (boolean)
~~~~~~~~~~~~~~~~~~~

Enable or disable fixed atoms.

fix_region (string)
~~~~~~~~~~~~~~~~~~~

- ``"bottom"`` — fix bottom part of slab
- ``"middle"`` — fix central region

fix_method (string)
~~~~~~~~~~~~~~~~~~~

- ``"layers"`` — fix a number of atomic layers
- ``"thickness"`` — fix a region of given thickness (Å)

fix_n_layers (integer)
~~~~~~~~~~~~~~~~~~~~~~

Number of layers to fix when using ``layers`` mode.

fix_thickness_A (float)
~~~~~~~~~~~~~~~~~~~~~~~

Thickness of fixed region (Å) when using ``thickness`` mode.

fix_layer_tolerance_A (float)
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

Tolerance used to group atoms into layers along the z-direction.

Notes
~~~~~

- Layer grouping is based on Cartesian z-coordinates.
- Fixed atoms are enforced during relaxation using ASE constraints,
  not only written to the POSCAR.

Surface Relaxation
~~~~~~~~~~~~~~~~~~

relax_surface (boolean)
~~~~~~~~~~~~~~~~~~~~~~~

Enable or disable slab relaxation.

Relaxation is performed using the same backend system as in Step 03.

surface_backend (string, default: "m3gnet")
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

Available options:

- ``"m3gnet"``
- ``"uma"``
- ``"mace"``
- ``"grace"``

surface_model (string, default: "default")
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

Model selection depends on backend (same behavior as Step 03).

surface_task (string, default: "")
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

Task/domain for UMA, or an optional MACE model ``head``. An omitted or empty
value uses a compatible dopingflow default (MACE-MH-1 resolves to
``omat_pbe``).

surface_optimizer (string, default: "bfgs")
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

ASE optimizer:

- ``"bfgs"``
- ``"lbfgs"``
- ``"fire"``
- ``"mdmin"``
- ``"quasinewton"``

surface_fmax (float)
~~~~~~~~~~~~~~~~~~~~

Maximum force convergence criterion (eV/Å).

surface_max_steps (integer, default: 300)
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

Maximum number of optimizer steps.

surface_device (string, default: "cpu")
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

- ``"cpu"``
- ``"cuda"``

surface_gpu_id (integer, default: 0)
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

GPU index used when ``device = "cuda"``.

surface_tf_threads (integer)
~~~~~~~~~~~~~~~~~~~~~~~~~~~~

TensorFlow thread count (for ``m3gnet``).

surface_omp_threads (integer)
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

OpenMP thread count.

Notes
~~~~~

- GPU mode uses a single effective worker.
- CPU mode runs sequentially per slab (no multiprocessing currently).
- Fixed atoms are enforced via ASE ``FixAtoms``.


Surface energy
~~~~~~~~~~~~~~

When enabled, the surface stage also evaluates the surface energy of each generated slab.

The surface energy is computed using the standard symmetric slab expression:

.. math::

   \gamma = \frac{E_{\mathrm{slab}} - n \, E_{\mathrm{bulk}}}{2A}

where:

- :math:`E_{\mathrm{slab}}` is the total energy of the slab (typically from surface relaxation)
- :math:`E_{\mathrm{bulk}}` is the relaxed bulk energy of the parent candidate
- :math:`n` is the number of bulk-equivalent units contained in the slab
- :math:`A` is the surface area
- the factor of 2 accounts for the two surfaces of the slab

Requirements
^^^^^^^^^^^^

Surface energy is computed only when all of the following conditions are satisfied:

- A slab energy is available (typically when ``relax_surface = true``)
- A bulk reference energy is available from the workflow database
- The slab composition is proportional to the parent bulk composition

If any of these conditions are not met, the surface energy is not computed and a status flag is recorded instead.

Output fields
^^^^^^^^^^^^^

The following fields are written to:

- ``meta.json`` (per surface)
- ``surface_summary.csv`` (global summary)

surface_energy_status (string)
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

Indicates whether surface energy was successfully computed.

Possible values include:

- ``"ok"`` — surface energy successfully computed
- ``"missing_slab_energy"`` — slab energy not available
- ``"missing_bulk_energy"`` — bulk reference energy missing
- ``"not_computable_not_proportional"`` — slab composition not proportional to bulk
- ``"not_computable_missing_species"``
- ``"not_computable_extra_species"``
- ``"failed: <error>"`` — unexpected numerical or runtime failure

surface_energy_eV_A2 (float)
~~~~~~~~~~~~~~~~~~~~~~~~~~~~

Surface energy expressed in eV/Å².

surface_energy_J_m2 (float)
~~~~~~~~~~~~~~~~~~~~~~~~~~~

Surface energy expressed in J/m².

Conversion used:

.. math::

   1 \, \mathrm{eV/Å^2} = 16.02176634 \, \mathrm{J/m^2}

surface_energy_reference_bulk_eV (float)
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

Bulk reference energy used in the calculation.

surface_energy_n_bulk_equiv (float)
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

Number of bulk-equivalent units contained in the slab.

This value is determined automatically from the ratio of slab and bulk compositions.

Notes
^^^^^

- Surface energies are only well-defined for slabs whose composition is proportional to the parent bulk.
- Non-stoichiometric terminations are not assigned a surface energy in the current implementation.
- Future extensions may include chemical-potential-based surface energies for non-stoichiometric slabs.

Output
~~~~~~

For each generated slab:

::

   composition_tag/
       candidate_###/
           hkl_h_k_l/
               term_###/
                   POSCAR
                   CONTCAR
                   meta.json
                   surface_relax.log
                   surface_relax.traj
                   surface_relax.json

Files:

``POSCAR``
   Generated slab structure (with optional selective dynamics)

``CONTCAR``
   Relaxed slab structure (if relaxation is enabled)

``meta.json``
   Slab metadata

``surface_relax.log``
   Relaxation log

``surface_relax.traj``
   ASE trajectory

``surface_relax.json``
   Relaxation metadata

Global output:

::

   surface_summary.csv

Notes
~~~~~

- ``min_slab_size`` and ``min_vacuum_size`` should be interpreted in Å
  when ``in_unit_planes = false``.
- ``orthogonal_c = true`` is recommended for correct slab geometry.
- Relaxed structures are saved separately and do not overwrite the initial slab.

Design Principles
-----------------

- All stages are deterministic given fixed input.
- All randomness is seed-controlled.
- Each stage can be skipped independently.
- Each stage writes metadata for full reproducibility.
- The relaxed host supercell is constructed once and reused.
