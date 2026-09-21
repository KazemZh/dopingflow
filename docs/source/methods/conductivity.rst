Electronic conductivity
=======================

The optional ``conductivity`` stage calculates the band-transport conductivity
**divided by relaxation time**, using GPAW eigenvalues and BoltzTraP2. It supports
vacancy-free and oxygen-deficient relaxed structures and shares verified GPAW
single points with oxidation analysis in either execution order.

This first backend assumes band-like transport. It does not calculate scattering
lifetimes, AMSET mobilities, defect ionization, small-polaron hopping, or an
automatic band-versus-polaron classification. Check localization before applying
it to vacancy states or transition-metal co-dopants. The existing oxidation
band-edge analysis is currently restricted to single-k-point wavefunctions;
its availability must not be assumed for a transport mesh.

Installation and execution
--------------------------

Install GPAW and its datasets in the scientific environment described in the
oxidation documentation.

For Linux/Conda environments, the recommended installation order is to install
the compiled BoltzTraP2 package from conda-forge first, and only then install the
DopingFlow conductivity extra::

    conda activate dopingflow_gpaw
    conda install -c conda-forge boltztrap2=26.3.1
    pip install -e ".[conductivity]"

If the Streamlit GUI is also required in the same environment::

    pip install -e ".[gui,conductivity]"

This order is intentional. When no compatible wheel is available, pip may try to
build BoltzTraP2 from source. That build compiles native extensions and its spglib
backend, so it may fail when development tools such as CMake are not installed.
A typical failure ends with::

    error: [Errno 2] No such file or directory: 'cmake'

Rather than adding build tools only to satisfy this source build, the recommended
DopingFlow setup is to use the precompiled conda-forge BoltzTraP2 package. The
subsequent `pip install -e ".[conductivity]"` recognizes the already installed
compatible package and installs DopingFlow's remaining optional dependencies.

GPAW is likewise intentionally not installed by the conductivity pip extra.
Install GPAW and its PAW datasets from conda-forge in the dedicated GPAW
environment.

Useful verification commands are::

    python -c "import gpaw; print('GPAW:', gpaw.__version__)"
    python -c "import BoltzTraP2; print('BoltzTraP2 OK')"
    python -c "import ase; print('ASE:', ase.__version__)"
    python -c "from dopingflow.conductivity import integrate_transport; print('DopingFlow conductivity OK')"

The adapter uses BoltzTraP2's Python interpolation/integration APIs and explicitly
loads both GPAW spin channels, with occupation weights 2 for unpolarized bands and
1 for each collinear-spin band. Noncollinear/SOC transport is not supported.

Copy the conductivity tables from ``examples/conductivity/input.toml`` into your
project input. Preview the structures without DFT execution::

    dopingflow conductivity -c input.toml --dry-run
    dopingflow conductivity -c input.toml

``python -m dopingflow`` is equivalent. The Streamlit **Electronic Conductivity**
page provides selection preview, settings, execution and JSON/CSV results.
``run-all`` includes this stage after oxidation, but it is disabled unless
``[conductivity].enabled = true``. A standalone conductivity command exits with
status 1 if any target fails or is unavailable; the JSON contains each reason.

Selection
---------

Conductivity intentionally uses the **same structure-selection model as the
oxidation-state stage**. The relevant parameters are::

    [conductivity]
    source_root = "vacancy-selected"
    include_vacancy_free = true
    include_oxygen_vacancies = true
    target_include = []

``source_root`` has the same meaning as ``oxidation.source_root``: it is the root
containing the selected relaxed parent structures and, when vacancy analysis is
enabled, ``vacancies_database.json``. If omitted, it first inherits
``oxidation.source_root`` when that is set; otherwise it falls back to
``[structure].outdir``.

``target_include`` is optional and behaves exactly like the oxidation selector.
Leave it empty to analyze every discovered target. Otherwise use exact target IDs,
safe IDs, or shell-style wildcards such as ``Sb5_Ti2p5/*``. Vacancy-free and
oxygen-vacancy structures can be enabled or disabled independently.

There is deliberately no second conductivity-specific ``favorable/manual/all``
selector, ``top_k_per_group``, or arbitrary ``structure_paths`` interface. Parent
and vacancy selection should happen once upstream; conductivity and oxidation then
consume the same target namespace. This keeps the TOML and Streamlit interfaces
consistent and prevents the two stages from silently choosing different structures.

Shared DFT calculations
-----------------------

``conductivity.dft`` inherits physical settings from ``oxidation.dft_electronic``
and the physical overrides of an active ``oxidation.dft_auto``. Explicit
conductivity settings take precedence. Set ``kpts`` explicitly: inheriting a
Gamma-only oxidation mesh is rejected. At least two points in each grid
direction are required, and this minimum is NOT a convergence criterion.

Both stages use ``<source_root>/dft_cache`` (or the same explicit
``cache_root``). Cache identity includes geometry, site order/species and electronic
settings: functional, cutoff, k mesh and offset convention, charge, spin,
initial moments, smearing, density convergence, iteration limit and band count.
The GPAW package version and GPAW_SETUP_PATH are also recorded in cache identity;
if datasets are replaced in place, force recalculation with reuse_existing=false.
A wavefunction-capable result satisfies a request without wavefunctions; the
reverse requires a new calculation. Files are checked for size/mtime changes.

Only identical electronic settings are reused: a denser transport mesh does not
silently replace a different oxidation calculation. No fixed-density mesh
refinement is implemented yet. Setting ``execute=false`` permits reuse but
never launches missing DFT. Setting ``reuse_existing=false`` and ``execute=true``
forces recalculation. Compatible old oxidation results are adopted only when
run metadata and the actual GPAW geometry agree; unverified files are refused.

A process lock serializes identical single points. Run the orchestrator in one
process; multi-rank MPI orchestration is explicitly rejected to avoid deadlocks.
GPAW runs inside that process. The cache retains a full GPW copy, plus a stage-local
copy, so allow disk space for wavefunctions. Removing a cache copy does not
remove the stage copy. GPAW outputs are reused, while inexpensive DOS/transport
postprocessing can run again. Automatic oxidation binds derived Bader and
Wannier artifacts to their source GPW and refuses stale derived data.

Transport meaning and outputs
-----------------------------

``temperatures_K`` controls finite-temperature integration. For each temperature,
the chemical potential is solved for the explicit structure's electron count
plus ``excess_electrons_cm3 * cell_volume_cm3``. Positive excess adds electrons;
negative removes them. Zero retains the native electron count, including the
electrons already present due to substitution and vacancies. This rigid-band
scan does not equate nominal dopants or vacancies with mobile carriers and is
not a defect-equilibrium calculation.

The output layout intentionally follows the oxidation stage where practical:

* ``selected_structures.json``: exact IDs, paths and shared target-selection provenance.
* ``conductivity_structure_index.csv`` and
  ``conductivity_structure_index.json``: one row/record per analyzed structure for
  the GUI structure browser.
* ``conductivity_results.json``: settings, warnings, per-target outcomes,
  GPAW paths/reuse and full Cartesian 3x3 conductivity/tau tensors.
* ``conductivity.csv``: one row per target, temperature and excess density.
* ``structures/<target_id>/conductivity.json``: individual target result.
* ``structures/<target_id>/summary.json``: compact per-structure status/provenance.

The Streamlit page likewise mirrors the oxidation page: stage/target controls,
method-specific settings, a TOML preview, the same two-column save/run area,
command preview and last-run output, followed by a per-structure results browser.
Common GPAW fields use the same labels and configuration keys in both stages.

For human-readable reporting, the primary conductivity-per-relaxation-time
quantity is written as ``sigma_over_tau_S_per_cm_per_fs``, in
``S cm^-1 fs^-1``. This is exactly the same quantity as the retained raw SI
field ``sigma_over_tau_S_per_m_per_s``; the numerical conversion is::

    (sigma/tau)[S cm^-1 fs^-1] = 1e-17 * (sigma/tau)[S m^-1 s^-1]

The trace divided by three is the directional average used for screening.
The ``S cm^-1 fs^-1`` representation is convenient because an assumed
relaxation time in femtoseconds converts directly to an estimated conductivity
in ``S/cm``. For example, ``282 S cm^-1 fs^-1`` with an assumed
``tau = 5 fs`` gives ``1410 S/cm``. The assumed-tau result remains
conditional: the present backend does not calculate the scattering lifetime.

ATO-normalized co-dopant comparison
-----------------------------------

For this project, the default benchmark is vacancy-free **ATO with 5% Sb**.
The reference is independent of the current co-dopant target selection: it can
live in another DopingFlow structure tree or be supplied as an explicit
POSCAR/CIF path.

A typical configuration is::

    [conductivity.comparison]
    enabled = true
    reference_label = "ATO 5% Sb"
    reference_sb_percent = 5.0
    reference_source_root = "../random_structures"
    reference_target = "Sb5/candidate_001"
    reference_structure_path = ""
    basis = "ato-5pct-sb-benchmark"

``reference_source_root`` defaults to the conductivity source root when left
empty. ``reference_target`` accepts the same exact/safe/wildcard syntax as the
normal target selector, but it must resolve to **exactly one vacancy-free
structure**. For production use an exact target ID rather than a broad wildcard.
If the ATO structure is outside a DopingFlow tree,
``reference_structure_path`` can point directly to a POSCAR or CIF and bypasses
reference discovery.

The first time a compatible ATO benchmark is needed, DopingFlow runs or reuses
the GPAW single point using the same ``conductivity.dft`` settings and then runs
BoltzTraP2. A successful reference is stored below the conductivity output
directory under ``references/<reference-label>/reference.json`` and summarized
in ``conductivity_reference.json``.

On later co-dopant runs, the stored ATO transport result is reused without
recalculating GPAW or BoltzTraP2 when its fingerprint still matches. The
fingerprint includes the reference geometry, GPAW electronic settings, GPAW
version/setup identity, temperatures, excess-carrier conditions, interpolation
factor, DOS grid, and transport regime. If only a transport setting changes,
the existing compatible GPAW cache can still be reused while BoltzTraP2 is
recomputed. If a DFT-defining setting changes, a compatible GPAW result is
required or ``conductivity.dft.execute = true`` must be enabled.

The benchmark is intentionally applied to **all screened structures**, including
oxygen-vacancy structures, because it answers the project-level question:
does a candidate preserve or improve the band-transport descriptor relative to
the 5% Sb ATO baseline? The output records both the candidate and reference
vacancy counts so this provenance remains explicit.

For each target and transport condition, DopingFlow reports::

    relative_to_reference = (sigma/tau)_target / (sigma/tau)_ATO

and::

    percent_change_vs_reference = 100 * (relative_to_reference - 1)

Thus a value of 1.0 (0%) preserves the ATO band-transport descriptor, values
above 1.0 indicate larger ``sigma/tau``, and values below 1.0 indicate a
smaller value. Comparisons require the same temperature and rigid-band
excess-electron concentration as the stored reference. They remain comparisons
of the **band-structure contribution** only; different dopants or vacancies may
also change the real scattering time.

The comparison table is written to
``conductivity_comparison.csv`` and ``conductivity_comparison.json`` and is
displayed prominently in the Streamlit page above the per-structure browser.

The table is **cumulative across separate conductivity runs**. DopingFlow scans
the per-structure ``conductivity.json`` files already present under the current
conductivity output directory and includes prior targets only when their saved
transport-settings fingerprint matches the current GPAW/BoltzTraP2 settings.
This makes it possible to run Ce, Ti, Mn, Nb, and other co-dopants one at a time
while maintaining one growing comparison against the same 5% Sb ATO benchmark.
Results calculated with a different k mesh, XC functional, cutoff, smearing,
spin setup, temperature/carrier grid, interpolation factor, DOS grid, or other
fingerprinted settings are not silently mixed into the table.

Converge k sampling, interpolation factor, integration grid, cutoff and empty
bands. Chemical potentials within 10 kBT of the sampled energy limits and
unconverged electron-count integration are rejected. Periodically repeated
vacancies do not capture random-defect scattering, grain boundaries, particle
contacts or all localization physics. MLFF force/energy calculations alone
cannot supply the electronic bands.

References
----------

* `BoltzTraP2 methodology <https://arxiv.org/abs/1712.07946>`_
* `AMSET scattering approach (future extension) <https://arxiv.org/abs/2008.09734>`_
* `Polaron hopping and kinetic Monte Carlo (separate method)
  <https://arxiv.org/abs/1808.02507>`_
