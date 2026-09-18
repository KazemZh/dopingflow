Oxidation-state analysis
========================

The ``oxidation`` stage analyzes the *relaxed structures already produced by
dopingflow*.  It is deliberately independent of the ML force field used for
relaxation.  A structure relaxed with MACE, UMA, GRACE, or M3GNet can therefore
be analyzed with TOSS-GNN, bond valence, CHGNet, BERTOS, DFT post-processing,
or any explicitly selected combination without replacing the relaxation
calculator.

By default the stage discovers both:

* vacancy-free selected relaxed parents; and
* relaxed oxygen-vacancy candidates from ``vacancies_database.json``.

This discovery is composition agnostic, so it applies to singly doped and
co-doped structures.  Every requested compatible method is run independently
on every discovered target.  Failure or unavailability of one optional method
does not invalidate the results from other requested methods.

Strategy and method selection
-----------------------------

Four strategy groups are exposed:

``structural``
   ``bond-valence`` (pymatgen ``BVAnalyzer``).

``ml``
   ``toss-gnn``, ``chgnet``, and ``bertos``.

``dft``
   ``dft-electronic``, ``bader``, ``wannier``, and ``eos``.

``combined``
   An explicitly selected mixture of methods from any groups.

Individual methods can always be selected.  For a single strategy group the
selected methods must belong to that group.  Use ``combined`` to mix groups.
For example::

   dopingflow oxidation -c input.toml --strategy ml --methods toss-gnn
   dopingflow oxidation -c input.toml --strategy combined --methods bond-valence,toss-gnn

The configuration entry point is::

   [oxidation]
   enabled = true
   strategy = "combined"
   methods = ["bond-valence", "toss-gnn"]
   include_vacancy_free = true
   include_oxygen_vacancies = true
   # Optional exact IDs, safe IDs, or shell-style glob patterns:
   # target_include = ["Sb5_Ti2p5/candidate_014"]
   output_dir = "06_oxidation"
   mapping_tolerance = 1.2
   fail_fast = false

``include_vacancy_free`` and ``include_oxygen_vacancies`` default to ``true``.
The normal use for vacancy chemistry is to leave both enabled so parent-relative
changes can be reported when atom mapping is valid.  ``target_include`` is optional;
when provided, only matching discovered targets are analyzed.  A selector can be an
exact target ID (for example ``Sb5_Ti2p5/candidate_014``), its safe-ID form
(``Sb5_Ti2p5__candidate_014``), or a shell-style glob such as
``Sb5_Ti2p5/*``.  This is the recommended way to constrain expensive DFT smoke
tests to one structure before scaling up.

TOSS-GNN
--------

The adapter is tied to the verified upstream TOSS inference interface at
commit ``c45582a3cd3088480b5d83b1440d360bd4577b80``.  The upstream pretrained
path uses:

* ``toss_GNN.Predict.Get_OS_by_models``;
* ``pyg_Hetero_GCNPredictor`` for link prediction;
* ``pyg_GCNPredictor`` for node classification;
* ``models/pyg_Hetero_GCN_s_0608.pth`` and
  ``models/pyg_GCN_s_0609.pth`` by default.

The result is site-resolved oxidation state plus coordination number.  Site
order is checked against the original relaxed structure.  The verified
upstream ``NC_predict`` interface does not expose calibrated probabilities, so
dopingflow does not invent confidence values.

Install the optional Python requirements with::

   pip install -e ".[oxidation-toss]"

and provide a local checkout of ``https://github.com/yueyin19960520/TOSS``.
The upstream TOSS repository itself is not installed silently by dopingflow.
Model objects are cached once per Python process for reuse.

The verified upstream predictor builds CPU graph tensors and converts outputs
with ``.detach().numpy()``.  Consequently the adapter currently accepts
``device = "cpu"`` for the verified interface; requesting CUDA returns an
explicit unavailable status rather than pretending CUDA was used.

CHGNet magnetic-moment evidence
-------------------------------

CHGNet is used first as a predictor of *local magnetic moments*.  These moments
are reported independently of any oxidation-state inference.  The upstream
CHGNet atom embedding is defined for up to 94 elements (atomic-number based),
and actual model inference may impose additional applicability limitations.

Oxidation-state inference is performed only where an explicit magnetic-moment
mapping exists.  By default dopingflow reproduces only the Mn mapping from
CHGNet's upstream ``solve_charge_by_mag`` helper, including its use of the raw
magnetic moment.  Other elements can be added through ``moment_oxidation_ranges``.
No universal mapping is assumed.  In particular, near-zero moments are not used
to distinguish Sn2+/Sn4+ or Sb3+/Sb5+.  An explicit
``use_absolute_moment = true`` option is available for user-defined workflows,
but it is recorded as a departure from the upstream helper semantics.

BERTOS
------

BERTOS is retained at its actual composition-token granularity.  The upstream
``getOS.py`` expands a reduced composition into repeated element tokens and
runs a pretrained token-classification model.  Those tokens are *not*
crystallographic site identifiers, so dopingflow labels the result
``composition-level`` and never maps it onto atom indices.

As a consequence, two structures with the same composition but different
vacancy arrangements are indistinguishable to BERTOS, and BERTOS cannot say
which specific atom was reduced near a vacancy.  The upstream softmax maximum
is recorded as a method-provided, uncalibrated score rather than validated
physical confidence.

The repository currently distributes pretrained model ZIP archives.  Extract
the selected archive (for example ``trained_models/ICSD_CN.zip``) and point the
configuration to the extracted model directory.

DFT interpretation
------------------

The DFT strategy uses **GPAW** as its electronic-structure backend.  GPAW is
open source and is driven directly from Python/ASE, so dopingflow does not
require per-structure ``INCAR``, ``KPOINTS``, ``POTCAR``, or equivalent input
files.

GPAW is intentionally not a ``pyproject.toml`` pip extra.  A pip installation
can build GPAW locally and therefore require MPI/compiler development headers.
The recommended GPAW + oxidation-GUI setup is a dedicated Conda environment::

   conda create -n dopingflow_gpaw python=3.11 pip -y
   conda activate dopingflow_gpaw
   conda install -c conda-forge gpaw gpaw-data wannier90
   pip install -e ".[gui]"
   gpaw info
   command -v wannier90.x
   python -m streamlit run gui/app.py

Run these commands from the dopingflow repository root for the editable
``pip install`` and GUI launch.  ``gpaw info`` should complete successfully
before production GPAW calculations are attempted.  Wannier90 is installed in the same environment for the optional Wannier route; ``command -v wannier90.x`` should resolve the executable.  Reuse the same
``dopingflow_gpaw`` environment for subsequent GPAW-backed oxidation runs.

``dft-electronic``
   Runs or reopens a GPAW ``.gpw`` ground-state calculation.  The current
   direct-execution path uses periodic plane-wave mode and records total energy,
   Fermi level, site magnetic moments, total DOS, and atom/angular-momentum
   projected DOS integrals.  ``dos.csv``, ``pdos_integrals.json``,
   ``magnetic_moments.csv``, and ``electronic_summary.json`` are written beside
   the ``oxidation.gpw`` restart file.  These quantities are descriptors only.

``bader``
   When ``execute=true``, dopingflow reopens the GPAW restart, reconstructs the
   all-electron density with ``get_all_electron_density()``, writes a cube file
   in the units expected by the Bader program, and invokes the free external
   ``bader`` executable.  Because the density contains all electrons, the
   reported continuous partial charge is ``Z - N_Bader``; no POTCAR or
   user-supplied valence-electron table is required.  Bader never emits an
   integer formal oxidation state by itself.  The GPAW all-electron-density
   grid refinement defaults to ``2`` to limit memory use; ``1`` and ``4`` remain
   available for convergence testing.

``wannier``
   Parses Wannier-center information and can generate it natively from an
   existing GPAW restart when ``execute=true``.  The first native mode is
   deliberately conservative: it supports an isolated, non-spin-polarized,
   Gamma-only occupied manifold, requires wavefunctions stored in
   ``oxidation.gpw``, detects the fully occupied bands and finite HOMO-LUMO gap,
   writes a Wannier90 input using Bloch phases as the initial gauge and
   ``write_xyz=true``, lets GPAW write ``.eig``/``.mmn``, and runs
   ``wannier90.x``.  Because the occupied manifold is isolated, no
   disentanglement is used and no arbitrary atomic projector set is invented.

   Completed Wannier results are also analyzed without rerunning Wannier90 when
   ``execute=false``.  Periodic minimum-image distances classify centers as
   atom-centered, bond-centered, or multicenter/ambiguous using configurable
   geometric thresholds.  A separate spread threshold flags anomalous or
   delocalized Wannier functions.  The defaults are ``0.45 Å`` for the
   atom-center cutoff, ``1.35 Å`` for the bond-center cutoff, ``0.30 Å`` for the
   two-neighbor distance balance, and ``3.0 Å²`` for the spread-outlier
   threshold.  These are screening heuristics, not universal chemical
   boundaries, and should be inspected for the material class.

   The post-processing writes ``wannier_centres_analysis.csv``,
   ``wannier_site_summary.csv``, and ``wannier_analysis.json`` beside the
   Wannier90 files.  Per-site counts and paired-electron equivalents are
   geometric bookkeeping descriptors; they are not atomic populations, Bader
   charges, or formal oxidation states.  Vacancy-free structures can be
   analyzed completely on their own.  If a matched oxygen-vacancy structure
   and its parent are both analyzed later, dopingflow additionally reports
   parent-relative changes in these per-site Wannier descriptors.  No vacancy
   structure is required for the standalone analysis.

   Metallic, spin-polarized, multi-k, or entangled cases are rejected by this
   native mode and should use an explicit projection/disentanglement workflow.
   Static centers remain supporting descriptors and are not converted into
   formal oxidation states automatically.

``eos``
   The formal DFT-based assignment adapter.  It accepts assignments only from
   an explicitly documented external EOS/charge-pumping workflow result marked
   ``validated=true``.  It does not infer EOS labels from Bader charge, DOS, or
   static Wannier centers.

The GPAW single-point path is opt-in.  ``execute=false`` only post-processes an
existing ``oxidation.gpw``.  ``execute=true`` runs GPAW directly with the shared
settings in ``[oxidation.dft_electronic]``.  Energy cutoff, k-point sampling,
spin initialization, smearing, and SCF convergence remain scientific convergence
parameters and must be validated for the target chemistry.  Bader has its own
``execute`` gate because running the external Bader executable is a separate
post-processing action.

The EOS exchange file is intentionally simple and auditable::

   {
     "validated": true,
     "assignment_procedure": "document the EOS / charge-pumping procedure here",
     "sites": [
       {"site_index": 0, "element": "Sn", "formal_oxidation_state": 4},
       {"site_index": 1, "element": "Sb", "formal_oxidation_state": 5}
     ],
     "provenance": {"code": "...", "calculation_id": "..."}
   }

Combined comparison and optional DFT follow-up
----------------------------------------------

Combined mode preserves each method result separately.  It reports sitewise
agreement, disagreement, missing assignments, and unavailable/failed methods.
Oxidation labels are never averaged and majority voting is never treated as
physical truth.

Conflicting or unresolved cases can be listed for DFT follow-up::

   [oxidation.dft_followup]
   enabled = true
   candidate_limit = 8
   execute = false
   methods = ["dft-electronic", "bader", "eos"]

With ``execute=false`` this only writes candidates.  Set ``execute=true`` only
when the selected method tables contain the GPAW/post-processing settings and work directories
needed for those calculations.

Outputs
-------

The stage writes aggregate files under ``[oxidation].output_dir``:

* ``oxidation_results.json``: complete per-method records;
* ``oxidation_sites.csv``: flattened site/composition-token records;
* ``oxidation_structure_index.csv`` / ``.json``: one row/object per analyzed structure;
* ``oxidation_comparison.json``: descriptive method comparison;
* ``dft_followup_candidates.json``: optional follow-up selection;
* ``meta.json``: resolved settings, target list, and provenance.

In addition, every structure gets a dedicated hierarchical result package under
``structures/<target_id>/``. For example, ``Sb5_In10/candidate_001`` is written
to ``structures/Sb5_In10/candidate_001/``. Each package contains
``summary.json``, ``oxidation_sites.csv``, ``oxidation_results.json``, and one
JSON file per requested method below ``methods/``. This layout is intended for
structure-by-structure inspection while the aggregate tables remain available
for screening and statistics.

A method that cannot assign a particular structure is recorded as
``unsupported``, ``unavailable``, or ``failed`` without interrupting other
structures. These cases are summarized once at the end of the run instead of
emitting a traceback for every ordinary non-assignment.

Results keep separate fields for method/strategy, prediction scope, formal
oxidation state, Bader partial charge, orbital populations, magnetic moments,
coordination, method-provided scores, provenance, limitations, and
parent-relative changes.  DFT DOS and Wannier descriptors are also retained in
the method JSON record.

Complete configuration examples
-------------------------------

1. TOSS-GNN only
^^^^^^^^^^^^^^^^

::

   [oxidation]
   enabled = true
   strategy = "ml"
   methods = ["toss-gnn"]
   include_vacancy_free = true
   include_oxygen_vacancies = true

   [oxidation.toss_gnn]
   repo_path = "/path/to/TOSS"
   lp_checkpoint = "models/pyg_Hetero_GCN_s_0608.pth"
   nc_checkpoint = "models/pyg_GCN_s_0609.pth"
   device = "cpu"

CLI::

   dopingflow oxidation -c input.toml --strategy ml --methods toss-gnn

2. Bond valence plus TOSS-GNN
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

::

   [oxidation]
   enabled = true
   strategy = "combined"
   methods = ["bond-valence", "toss-gnn"]

   [oxidation.bond_valence]
   symm_tol = 0.1
   max_radius = 4.0

   [oxidation.toss_gnn]
   repo_path = "/path/to/TOSS"
   device = "cpu"

CLI::

   dopingflow oxidation -c input.toml --strategy combined --methods bond-valence,toss-gnn

3. CHGNet analysis for compatible transition-metal dopants
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

::

   [oxidation]
   enabled = true
   strategy = "ml"
   methods = ["chgnet"]

   [oxidation.chgnet]
   model_name = "0.3.0"
   device = "cpu"
   use_upstream_mn_mapping = true
   use_absolute_moment = false
   # Optional additional element-specific ranges may be supplied as
   # moment_oxidation_ranges = { Element = [[min_moment, max_moment, OS], ...] }
   # only when a chemically justified mapping is available for that element/model.

CLI::

   dopingflow oxidation -c input.toml --strategy ml --methods chgnet

4. BERTOS composition screening
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

::

   [oxidation]
   enabled = true
   strategy = "ml"
   methods = ["bertos"]

   [oxidation.bertos]
   repo_path = "/path/to/BERTOS"
   model_path = "trained_models/ICSD_CN"  # extracted directory
   tokenizer_path = "tokenizer"
   device = "cpu"

CLI::

   dopingflow oxidation -c input.toml --strategy ml --methods bertos

5. GPAW plus Bader and orbital analysis
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

The two methods intentionally share the same per-target output root.  GPAW can
run the single point directly from the relaxed structure.  Bader can then
reconstruct GPAW's all-electron density and run the free ``bader`` executable.

::

   [oxidation]
   enabled = true
   strategy = "dft"
   methods = ["dft-electronic", "bader"]

   [oxidation.dft_electronic]
   code = "gpaw"
   output_root = "dft_oxidation"
   execute = true
   mode = "pw"
   ecut_eV = 500.0
   xc = "PBE"
   kpts = [1, 1, 1]
   gamma = true
   smearing_eV = 0.05
   convergence_density = 1e-5
   maxiter = 333
   spinpol = "auto"
   initial_magmoms = {}
   gpw_file = "oxidation.gpw"

   [oxidation.bader]
   output_root = "dft_oxidation"
   gpw_file = "oxidation.gpw"
   execute = true
   gridrefinement = 4
   density_file = "density.cube"
   acf_file = "ACF.dat"
   command = ["bader", "density.cube"]

CLI::

   dopingflow oxidation -c input.toml --strategy dft --methods dft-electronic,bader

Selecting only ``bader`` produces Bader descriptors and *no invented integer
formal oxidation-state assignment*.  If ``execute=true`` for Bader, an existing
GPAW restart must already be present in the same per-target directory.

6. DFT plus Wannier/EOS where supported
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

::

   [oxidation]
   enabled = true
   strategy = "dft"
   methods = ["dft-electronic", "wannier", "eos"]

   [oxidation.dft_electronic]
   output_root = "dft_oxidation"
   execute = false

   [oxidation.wannier]
   output_root = "dft_oxidation"
   centres_file = "wannier90_centres.xyz"
   execute = false

   [oxidation.eos]
   output_root = "dft_oxidation"
   results_file = "eos_results.json"
   execute = false

CLI::

   dopingflow oxidation -c input.toml --strategy dft --methods dft-electronic,wannier,eos

7. Combined comparison on matched vacancy-free and vacancy structures
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

::

   [oxidation]
   enabled = true
   strategy = "combined"
   methods = ["bond-valence", "toss-gnn", "chgnet", "bader", "eos"]
   include_vacancy_free = true
   include_oxygen_vacancies = true
   mapping_tolerance = 1.2
   fail_fast = false

   [oxidation.toss_gnn]
   repo_path = "/path/to/TOSS"
   device = "cpu"

   [oxidation.chgnet]
   model_name = "0.3.0"
   device = "cpu"
   use_upstream_mn_mapping = true

   [oxidation.bader]
   output_root = "dft_oxidation"
   execute = false

   [oxidation.eos]
   output_root = "dft_oxidation"
   execute = false
   results_file = "eos_results.json"

   [oxidation.dft_followup]
   enabled = true
   candidate_limit = 5
   execute = false
   methods = ["dft-electronic", "bader", "eos"]

CLI::

   dopingflow oxidation -c input.toml --strategy combined --methods bond-valence,toss-gnn,chgnet,bader,eos

The comparison remains descriptive.  A disagreement is a trigger for inspection
or an explicitly enabled DFT follow-up, not a vote to select a winner.

The native GPAW/Wannier90 adapter supports both the GPAW 25.7 `gpaw.wannier90` interface and the newer `gpaw.wannier.wannier90` interface.
