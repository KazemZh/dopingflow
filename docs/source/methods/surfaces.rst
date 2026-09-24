Staged Surface Screening and Refinement
=======================================

Overview
--------

The surface workflow converts selected low-energy bulk candidates into slab
models, scans orientations, terminations, and representative co-dopant depth
arrangements, and optionally re-evaluates the shortlist with a second
higher-fidelity ML calculator.

The intended sequence is::

   selected bulk candidate
       -> facets
       -> terminations
       -> representative co-dopant depth variants
       -> fast MLFF screen
       -> top-k shortlist
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
   top_k_per_candidate = 10

Refinement calculator
---------------------

The surface.refine section is independent of the screening calculator and
re-evaluates only surface_screen_selected.csv.

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
   top_k_per_candidate = 5

Any model accepted by the existing backend abstraction may be chosen,
including a supported MACE alias or custom checkpoint path. Refinement therefore
does not mean DFT.

Calculator-consistent bulk references
-------------------------------------

Surface and source-reference energies are never silently mixed across calculators.

For every selected parent, the screening calculator evaluates the bulk
structure and that energy is used only for screening surface energies. The
refinement calculator independently evaluates the same parent structure and
that energy is used only for refinement surface energies.

The current implementation uses a same-calculator single-point energy on the
already-relaxed parent bulk geometry. This gives an internally consistent
ranking within a parent. Absolute publication-quality surface energies should
still be converged with respect to bulk geometry, slab thickness, vacuum,
constraints, and calculator settings.

Surface energy and ranking
--------------------------

For a slab whose composition is proportional to its parent bulk, the workflow
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

The default output directory is 08_surfaces.

surface_screen_summary.csv
   Every generated orientation, termination, and dopant-depth variant.

surface_screen_selected.csv
   Top-k rankable variants per bulk candidate after the screen calculator.

surface_refine_summary.csv
   Higher-fidelity results for the screening shortlist.

surface_final_selected.csv
   Final top-k variants after refinement.

Each variant directory also stores the generated POSCAR, stage-specific
result.json, optional relaxed POSCAR, optimizer log/trajectory, and meta.json.

A typical path is::

   08_surfaces/
     Sb5_Ti5/
       candidate_001/
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
- inspect same-termination segregation energies relative to the all-bulk-like
  variant;
- browse the selected slab geometry interactively;
- inspect the raw screen/refinement tables.

The results explorer keeps surface-energy ranking and dopant segregation as
separate quantities. A low surface energy identifies a thermodynamically
favorable exposed slab within the implemented model, whereas a negative
segregation energy indicates that the selected dopant placement is favored
relative to the corresponding bulk-like placement.
