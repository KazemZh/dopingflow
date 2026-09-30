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
       -> natural dopant-depth labels inherited from each termination
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

Natural dopant-depth labels
---------------------------

The surface stage no longer moves or swaps dopant atoms. For every generated
termination, the dopants remain exactly on the sites inherited from the
periodic source structure and the selected slab cut.

DopingFlow only labels those original dopant positions. Cations are grouped
into layers along the slab normal. With ``dopant_depth_layers = 1``:

- the outermost cation layer on each exposed slab side is ``surface``;
- the next cation layer on each side is ``subsurface``;
- all remaining cation layers are ``bulk``.

Both slab sides are considered. ``cation_layer_tolerance_A`` controls the
Cartesian tolerance used to group cations into layers.

A co-doped termination may therefore be labeled, for example::

   In: surface | Sb: subsurface

For multiple atoms of the same dopant, the label preserves the counts, e.g.::

   In: surface + bulk×2 | Sb: subsurface

These labels are descriptive metadata only; no atom identities are changed.

Example::

   host_species = "Sn"
   dopant_species = ["Sb", "In"]
   anion_species = ["O"]
   dopant_depth_layers = 1
   cation_layer_tolerance_A = 0.8
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

The natural-termination workflow has one screened structure per termination::

   source structure
       -> Miller orientation
           -> termination

Two automatic selection strategies are available.

``selection_mode = "global"``
   Select the lowest-energy ``selection_top_k`` terminations for each source
   structure regardless of orientation.

``selection_mode = "per_orientation"``
   Select terminations independently for every Miller orientation. The default
   number is controlled by ``default_terminations_per_orientation`` and can be
   overridden separately for every orientation::

      [surface.refine]
      selection_mode = "per_orientation"
      default_terminations_per_orientation = 3

      [surface.refine.orientation_limits."1,0,0"]
      terminations = 3

      [surface.refine.orientation_limits."1,1,0"]
      terminations = 5

This prevents a single low-energy facet from consuming the full refinement
budget while still allowing a different number of terminations for each facet.

Manual override
~~~~~~~~~~~~~~~

Once ``surface_screen_summary.csv`` exists, the GUI can display every rankable
termination with an ``Include`` checkbox. Exact terminations may be added or
removed after the automatic selection. Stable surface IDs contain the source
target, Miller index, and termination ID.

Reusing an existing screen
~~~~~~~~~~~~~~~~~~~~~~~~~~

A screen produced by the natural-termination workflow can be reselected without
rerunning the expensive screening calculations. Screen summaries created by the
older artificial dopant-variant workflow must be regenerated once before
refinement because they represent a different physical model.

Final refined set
~~~~~~~~~~~~~~~~~

All successfully refined terminations are retained by default::

   final_selection_mode = "all"

Thus ``surface_refine_summary.csv`` contains the complete refined set and
``surface_final_selected.csv`` mirrors it by default.
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
   Every generated natural orientation/termination and its dopant-depth label.

surface_screen_selected.csv
   The exact screened surfaces selected for refinement after the automatic
   strategy and any manual include/exclude overrides.

surface_refine_summary.csv
   Higher-fidelity results for the screening shortlist.

surface_final_selected.csv
   All successfully refined surfaces by default. A later explicit downstream
   selection can reduce this set if desired.

Each termination directory also stores the generated POSCAR, stage-specific
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
           term_001_In-surface__Sb-subsurface/
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
- set a different number of natural terminations for each Miller orientation;
- manually add or remove exact screened surfaces before refinement;
- browse the selected slab geometry interactively;
- inspect the raw screen/refinement tables.

The results explorer focuses on surface-energy ranking of the natural terminations.
The displayed dopant-depth label describes where the original dopants ended up
after the slab cut; no atom swapping is performed.
