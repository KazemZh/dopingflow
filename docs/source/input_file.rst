Warning: truncated output (original token count: 15615)
Total output lines: 2077

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
   search_method = "enumeration"
   supercell = [1, 1, 1]
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

   # Monte Carlo settings are used only with search_method = "monte-carlo".
   mc_temperature_K = 300.0
   mc_annealing = false
   mc_initial_temperature_K = 1200.0
   mc_annealing_hold_steps = 500
   mc_annealing_steps = 2000
   mc_run_mode = "combined"
   mc_max_steps = 10000
   mc_patience = 2000
   mc_improvement_tolerance_eV = 1.0e-5
   mc_energy_window_eV = 0.5
   mc_cation_move_weight = 0.5
   mc_vacancy_move_weight = 0.5
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

``search_method`` accepts ``"enumeration"`` (default) or ``"monte-carlo"``.
``supercell`` is a three-positive-integer expansion applied consistently to the
scan and relaxed parent before either search. Enumeration uses the symmetry,
exact/sample, sampling-budget, and minimum-distance fields. Monte Carlo uses
``sample_seed`` and ``sample_max_saved`` plus the ``mc_*`` fields; enumeration
limits are ignored by that search. With ``mc_annealing = false``, all steps use
the constant ``mc_temperature_K``. With annealing enabled, the search holds
``mc_initial_temperature_K``, cools linearly during ``mc_annealing_steps``, and
then remains at ``mc_temperature_K``. The initial temperature must not be lower
than the target.

For a directory that already contains many composition subdirectories, use::

   parent_source = "directory"
   parent_directory = "Mn-Nb-Ta-Ce-Ru-In-Sn-Sb"

Each subdirectory is discovered through its own ``selected_candidates.txt``.
The directory path remains a flat key in ``[vacancies]``.

Static-lattice thermodynamic analysis is disabled by default for compatibility. When
enabled, all keys remain in this same flat table. ``oxygen_reference_mode`` is
``reference_file``, ``same_calculator``, ``explicit``, ``none``, ``global``, or
``chemistry-specific``. The last two modes fit a backend/model/task-specific
per-O reference from eligible calculated binary oxides and experimental 298 K
formation enthalpies; they use the ``oxygen_calibration_*`` fields. Explicit
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
``oxide_cation`` terms from admitted ca…5615 tokens truncated…zer used during relaxation.

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

Set ``bulk_structure`` to a periodic POSCAR or CIF to generate facets directly
from a bulk unit cell. Set ``conventional_millers`` to interpret the indices
in the conventional cell and ``inplane_repeat`` to repeat the slab laterally.
To compare coverages on the screened facets, configure ``state_families`` and
coverage percentages in ``[surface_pourbaix]``. Custom molecules use nested
``[surface_pourbaix.adsorbates.NAME]`` tables. See :doc:`methods/surface_pourbaix`.

For candidate-based screening the stage reads a database file:

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
