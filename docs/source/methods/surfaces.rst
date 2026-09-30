Staged Surface Screening and Refinement
=======================================

Overview
--------

The surface workflow converts selected relaxed source structures into slab
models, scans orientations, terminations, and representative co-dopant depth
arrangements, and optionally re-evaluates the shortlist with a second
higher-fidelity ML calculator.

The intended sequence is::

   selected vacancy-free or O-vacancy source structure
       -> facets
       -> terminations
       -> representative co-dopant depth variants
       -> fast MLFF screen
       -> configurable refinement shortlist
          (global best or best per orientation)
       -> optional higher-fidelity MLFF refinement
       -> final surface shortlist

Leaching and Ir/IrOx deposition are intentionally outside this stage.

Commands
--------

The workflow is split so different calculators can live in different Conda
environments::

   dopingflow surface-scan -c input.toml
   dopingflow surface-refine -c input.toml

If both calculator dependencies are available in one environment::

   dopingflow surface -c input.toml

Structure selection
-------------------

Surface screening uses the same structure-discovery convention as oxidation-state
and electronic-conductivity analysis. Configure::

   source_root = "vacancy-selected"
   include_vacancy_free = true
   include_oxygen_vacancies = false
   target_include = []

``source_root`` contains the selected relaxed parent structures and, when
oxygen-vacancy targets are requested, ``vacancies_database.json``.

``include_vacancy_free`` controls whether relaxed selected parents are scanned.
``include_oxygen_vacancies`` independently controls whether relaxed O-vacancy
structures are scanned. ``target_include`` is optional and accepts the same exact
target IDs and shell-style wildcards used by oxidation/conductivity, for example::

   target_include = ["Sb5_Ti2p5/candidate_014", "Sb10_Nb5/*"]

Leaving ``target_include`` empty uses every discovered structure allowed by the
two vacancy toggles.

For an oxygen-vacancy target, the corresponding periodic oxygen-deficient
structure is the reference structure for that target's slab generation and
same-calculator surface-energy expression. Vacancy-free and vacancy-containing
targets are ranked independently rather than being mixed into one ranking.

Surface orientations
--------------------

For the current rutile SnO2 project the default explicit starting set is
(110), (100), (101), and (001). The list is configurable with miller_list.
Setting orientation_mode = "automatic" uses pymatgen's symmetrically distinct
Miller indices up to max_miller and then applies max_orientations.

Terminations
------------

Each orientation is passed to pymatgen.core.surface.SlabGenerator.
termination_mode = "all" preserves generated terminations up to
max_terminations_per_orientation; "first" keeps only the first.

Slab thickness, vacuum, centering, lattice reorientation, orthogonal-c
conversion, and optional symmetrization are controlled in the surface section.

Representative co-dopant depth scan
-----------------------------------

With dopant_variant_mode = "co-dopant-depth", the workflow identifies a host
cation and selected dopant species and constructs representative variants in
surface, subsurface, and bulk-like cation layers.

For each selected dopant species, one representative dopant atom is swapped
with a host site in the requested zone. Total composition is unchanged. Other
same-species dopants retain their parent ordering.

This is deliberately a controlled screening of depth preference rather than an
exhaustive enumeration of every possible same-species dopant permutation. The
max_dopant_variants_per_termination setting prevents combinatorial growth.

For the ATO/co-doped SnO2 use case an explicit setup may be::

   host_species = "Sn"
   dopant_species = ["Sb", "Ti"]
   anion_species = ["O"]
   depth_zones = ["surface", "subsurface", "bulk"]

If dopant_species is omitted, all non-host, non-anion species are inferred as
dopants.

Screen calculator
-----------------

The surface.screen section defines the fast calculator used on every generated
slab variant. It uses the same ML backend abstraction as the bulk workflow and
supports M3GNet, UMA, MACE, and GRACE.

Example::

   [surface.screen]
   backend = "grace"
   model = "GRACE-1L-OMAT"
   device = "cuda"
   relax = true
   fmax = 0.05
   max_steps = 300

Refinement calculator
---------------------

The surface.refine section is independent of the screening calculator and
re-evaluates only the surfaces selected from the completed screening table.

Example::

   [surface.refine]
   enabled = true
   backend = "mace"
   model = "mh-1"
   task = "matpes_r2scan"
   device = "cuda"
   relax = true
   fmax = 0.03
   max_steps = 500

   # Preserve both orientation and termination diversity:
   selection_mode = "orientation_termination"
   default_terminations_per_orientation = 3
   default_variants_per_termination = 1

   # Keep all successfully refined surfaces:
   final_selection_mode = "all"

   [surface.refine.orientation_limits."1,0,0"]
   terminations = 3
   variants_per_termination = 2

   [surface.refine.orientation_limits."1,1,0"]
   terminations = 4
   variants_per_termination = 2

Any model accepted by the existing backend abstraction may be chosen,
including a supported MACE alias or custom checkpoint path. Refinement therefore
does not mean DFT.

Refinement candidate selection
------------------------------

The screening table is organized hierarchically as::

   source structure
       -> Miller orientation
           -> termination
               -> surface/dopant variant

``[surface.refine].selection_mode`` controls which screened structures are
passed to the higher-fidelity calculator.

``selection_mode = "global"``
   Select the lowest ``selection_top_k`` surface energies for each source
   structure regardless of orientation or termination.

``selection_mode = "per_orientation"``
   Select the lowest ``selection_top_k`` surfaces independently for each
   Miller orientation. This prevents one facet from consuming the complete
   refinement budget, but several selected structures can still belong to the
   same termination.

``selection_mode = "orientation_termination"``
   Preserve both facet and termination diversity. For each Miller orientation,
   DopingFlow first ranks **distinct terminations** by the minimum screened
   surface energy among the variants belonging to that termination. It then
   keeps the requested number of terminations and, within each retained
   termination, the requested number of lowest-energy variants.

Per-orientation controls
~~~~~~~~~~~~~~~~~~~~~~~~

The balanced mode has global defaults::

   default_terminations_per_orientation = 3
   default_variants_per_termination = 1

and optional orientation-specific overrides::

   [surface.refine.orientation_limits."1,0,0"]
   terminations = 3
   variants_per_termination = 2

   [surface.refine.orientation_limits."1,1,0"]
   terminations = 4
   variants_per_termination = 2

This allows different refinement budgets for (100), (110), (101), (001), or
any other requested Miller orientation.

Termination ranking
~~~~~~~~~~~~~~~~~~~

A termination can contain several dopant-depth variants. For the balanced
selector, a termination is ranked using::

   gamma_termination = min_j gamma_(termination,j)

where ``j`` runs over rankable variants of that termination. After the best
distinct terminations are chosen, variants inside each selected termination are
ranked by their own screened surface energies.

Manual override
~~~~~~~~~~~~~~~

Once ``surface_screen_summary.csv`` exists, the GUI can display every rankable
screened surface with an editable ``Include`` checkbox. The automatic shortlist
is shown as the initial selection, and the user may add or remove exact
structures before refinement.

Manual edits are stored as stable surface IDs in
``manual_include_surface_ids`` and ``manual_exclude_surface_ids``. A surface ID
contains the source target, Miller index, termination ID, and variant ID, so the
same manual selection can be reconstructed later without rerunning the screen.

Reusing an existing screen
~~~~~~~~~~~~~~~~~~~~~~~~~~

When ``surface_screen_summary.csv`` already exists, ``surface-refine`` rebuilds
the refinement shortlist from the complete screen table using the **current**
automatic and manual selection settings. Changing the strategy, orientation
limits, or exact checked surfaces therefore does **not** require repeating the
expensive screening calculations.

Final refined set
~~~~~~~~~~~~~~~~~

All successfully refined surfaces are retained by default::

   final_selection_mode = "all"

Therefore ``surface_refine_summary.csv`` contains the complete refined set and
``surface_final_selected.csv`` mirrors that set by default. A legacy/advanced
``final_selection_mode = "global"`` remains available through TOML for users
who explicitly want a final global top-k reduction, but it is no longer a
mandatory control in the GUI.

Calculator-consistent source references
-------------------------------------

Surface and source-reference energies are never silently mixed across calculators.

For every selected source structure, the screening calculator evaluates that
periodic source and uses its energy only for screening surface energies. The
refinement calculator independently evaluates the same source structure and
uses that energy only for refinement surface energies.

The current implementation uses a same-calculator single-point energy on the
already-relaxed periodic source geometry. This gives an internally consistent
ranking within one source target. Absolute publication-quality surface energies should
still be converged with respect to bulk geometry, slab thickness, vacuum,
constraints, and calculator settings.

Surface energy and ranking
--------------------------

For a slab whose composition is proportional to its periodic source structure, the workflow
uses::

   gamma = (E_slab - n E_bulk) / (2 A)

where A is the area of one slab face and n is the bulk-equivalent composition
factor.

Important interpretation:

- the expression is a two-surface average unless the two faces are equivalent;
- only proportional/stoichiometric slabs receive this simple surface energy;
- non-stoichiometric terminations remain in the output with an explicit
  not_computable status and are excluded from ranking;
- raw slab energies are never used to rank structures with different atom
  counts.

A chemical-potential treatment for non-stoichiometric terminations is a
separate future extension.

Constraints
-----------

fix_atoms can constrain middle or bottom layers using a layer count or Cartesian
thickness. The same fixed atoms are enforced during ASE relaxation.

Outputs
-------

The default relative output directory is ``08_surfaces`` **inside the selected ``source_root``**. For example, ``source_root = "vacancy-selected/structures-analysis"`` and ``outdir = "08_surfaces"`` resolve to ``vacancy-selected/structures-analysis/08_surfaces``. An absolute ``outdir`` is used exactly as given.

surface_screen_summary.csv
   Every generated orientation, termination, and dopant-depth variant.

surface_screen_selected.csv
   The exact screened surfaces selected for refinement after the automatic
   strategy and any manual include/exclude overrides.

surface_refine_summary.csv
   Higher-fidelity results for the screening shortlist.

surface_final_selected.csv
   All successfully refined surfaces by default. A later explicit downstream
   selection can reduce this set if desired.

Each variant directory also stores the generated POSCAR, stage-specific
result.json, optional relaxed POSCAR, optimizer log/trajectory, and meta.json.

A typical path is::

   <source_root>/
     08_surfaces/
       targets/
         Sb5_Ti5__candidate_001/
           bulk_reference/
             screen/
             refine/
           hkl_1_1_0/
             term_001/
               variant_001_original/
                 POSCAR
                 screen/
                 refine/

Backward compatibility
----------------------

Older flat surface relaxation keys such as surface_backend, surface_model,
surface_task, surface_device, surface_fmax, and surface_max_steps are mapped to
surface.screen when the nested screen section is not supplied. New studies
should use the staged nested sections.

Graphical interface
-------------------

The Streamlit ``Surface Screening`` page owns the complete surface-stage
configuration. The old Surface editor in the main Input Builder has been
replaced by a link to this page so that the same settings are not maintained in
two places.

The page mirrors the staged CLI design:

- configure and preview vacancy-free and/or O-vacancy source structures;
- edit slab, termination, co-dopant-depth, and constraint settings;
- configure the independent screen and refinement calculators;
- run the screen and refinement either in the current environment or through
  separate named Conda environments;
- inspect surface-energy rankings one selected source structure at a time;
- choose global, per-orientation, or orientation/termination-balanced refinement;
- set different termination/variant budgets for each Miller orientation;
- manually add or remove exact screened surfaces before refinement;
- browse the selected slab geometry interactively;
- inspect the raw screen/refinement tables.

The results explorer focuses on surface-energy ranking. Dopant-depth variants
remain explicit structures (for example surface, subsurface, bulk-like, and the
original cut-slab arrangement), but the workflow no longer reports a separate
segregation-energy metric. Their relative performance is assessed through the
calculated surface energies and the selected refinement strategy.
