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
oxidation documentation. Then install the optional transport package::

    pip install -e '.[conductivity]'

BoltzTraP2 may need a C++ compiler and CMake when built from source. The adapter
uses its Python interpolation/integration APIs and explicitly loads both GPAW
spin channels, with occupation weights 2 for unpolarized bands and 1 for each
collinear-spin band. Noncollinear/SOC transport is not supported.

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
enabled, ``vacancies_database.json``. If omitted, it inherits
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

The results include:

* ``selected_structures.json``: exact IDs, paths and shared target-selection provenance.
* ``conductivity_results.json``: settings, warnings, per-target outcomes,
  GPAW paths/reuse and full Cartesian 3x3 conductivity/tau tensors.
* ``conductivity.csv``: one row per target, temperature and excess density.
* ``structures/<id>/conductivity.json``: individual target result.

``sigma_over_tau_S_per_m_per_s`` has units S m^-1 s^-1. The trace divided by
three is a directional average, not a prediction for a porous electrode.
Only if ``relaxation_time_fs`` is supplied are conditional absolute tensors
(S/m) and averages (S/cm) emitted. Those quantities remain assumptions about
the scattering time, not first-principles lifetime predictions.

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
