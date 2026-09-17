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
   ``bond-valence`` (pymatgen ``BVAnalyzer``) and ``toss-bayesian``
   (conventional MAP/Bayesian TOSS).

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
   output_dir = "06_oxidation"
   mapping_tolerance = 1.2
   fail_fast = false

``include_vacancy_free`` and ``include_oxygen_vacancies`` default to ``true``.
The normal use for vacancy chemistry is to leave both enabled so parent-relative
changes can be reported when atom mapping is valid.

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
CHGNet's upstream ``solve_charge_by_mag`` helper.  Other elements can be added
through ``moment_oxidation_ranges``.  No universal mapping is assumed.
In particular, near-zero moments are not used to distinguish Sn2+/Sn4+ or
Sb3+/Sb5+.

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

The DFT strategy deliberately separates *formal assignment* from supporting
electronic descriptors:

``dft-electronic``
   VASP ``vasprun.xml``/``OUTCAR`` post-processing for total/projected DOS,
   approximate integrated projected populations below the Fermi level, final
   energy, Fermi level, and site magnetic moments.  These are descriptors only.

``bader``
   Parses ``ACF.dat`` and reports Bader electron populations and partial charge
   when a valence-electron reference is available from ``POTCAR`` or the
   configuration.  Bader never emits an integer formal oxidation state by
   itself.

``wannier``
   Parses ``wannier90_centres.xyz`` and records Wannier-center information.
   Static centers are supporting descriptors and are not converted into formal
   oxidation states automatically.

``eos``
   The formal DFT-based assignment adapter.  It accepts assignments only from
   an explicitly documented external EOS/charge-pumping workflow result marked
   ``validated=true``.  It does not infer EOS labels from Bader charge, DOS, or
   static Wannier centers.

DFT mode can post-process existing outputs or explicitly execute a configured
external command.  Expensive work is never started merely because another
method failed.  For each DFT submethod, ``execute = false`` is the safe default.
When ``execute = true`` an explicit ``command`` is required.

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
when the selected DFT method tables also contain the commands/work directories
needed for those calculations.

Outputs
-------

The stage writes under ``[oxidation].output_dir``:

* ``oxidation_results.json``: complete per-method records;
* ``oxidation_sites.csv``: flattened site/composition-token records;
* ``oxidation_comparison.json``: descriptive method comparison;
* ``dft_followup_candidates.json``: optional follow-up selection;
* ``meta.json``: resolved settings, target list, and provenance.

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
   # Optional additional element-specific ranges may be supplied as
   # moment_oxidation_ranges = { Element = [[min_abs_moment, max_abs_moment, OS], ...] }
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

5. DFT plus Bader and orbital analysis
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

The two methods intentionally share the same per-target output root.  Run the
DFT command first; Bader can either post-process an existing ``ACF.dat`` or run
an explicit Bader command.

::

   [oxidation]
   enabled = true
   strategy = "dft"
   methods = ["dft-electronic", "bader"]

   [oxidation.dft_electronic]
   code = "vasp"
   output_root = "dft_oxidation"
   execute = true
   command = ["mpirun", "-np", "8", "vasp_std"]

   [oxidation.bader]
   output_root = "dft_oxidation"
   execute = true
   command = ["bader", "CHGCAR"]
   # Optional if POTCAR is unavailable:
   valence_electrons = { Sn = 4, Sb = 5, O = 6 }

CLI::

   dopingflow oxidation -c input.toml --strategy dft --methods dft-electronic,bader

Selecting only ``bader`` produces Bader descriptors and *no invented integer
formal oxidation-state assignment*.

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
