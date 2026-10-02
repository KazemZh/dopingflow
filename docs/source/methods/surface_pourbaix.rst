Surface electrochemistry and Pourbaix diagrams
==============================================

The surface-pourbaix command is Step 16, between surface segregation and
leaching. Its purpose is not to assume that adding protons stabilizes an oxide
surface. Instead, it compares intact-surface chemical states and asks which
state is thermodynamically preferred as a function of electrode potential and
pH.

Input and state search
----------------------

Surface segregation is **not required**. By default, ``source_mode = "surface"``
reads the best available table produced directly by the Surface stage, in this
order: final selected surfaces, all refined surfaces, screened shortlist, then
the full screened/generated surface table.

If a segregation study has been performed and those structures should be used
instead, set ``source_mode = "segregation"``. ``source_mode = "auto"`` also
prefers the direct Surface-stage tables and uses segregation only as a fallback.
An explicit source CSV can still be supplied with ``source_summary``.

The GUI lists the surfaces found in the selected table and allows exact surface
IDs to be picked directly. The generated state families are:

* clean surface;
* protonated surface oxygen sites;
* O* adsorbates;
* OH* adsorbates;
* H2O* adsorbates; and
* mixed O*/OH* states.

Coverage, side selection, and arrangement
----------------------------------------

Coverage is specified as a percentage of the eligible sites on the selected
surface side, rather than as an absolute atom count. Separate grids are available
for protonated lattice O, O*, OH*, and H2O*, plus explicit O*:OH* coverage pairs.
For each surface, DopingFlow converts the requested percentage to the closest
integer occupation allowed by that finite surface cell. When a requested value
lies exactly halfway between two integer occupations, both are retained. For
example, 25% coverage on 12 sites maps to 3 occupied sites, whereas 25% on 10
sites maps to both 2/10 (20%) and 3/10 (30%). Requested and actual coverages are
both written to the state summary.

By default ``max_surface_oxygen_sites = 0`` and
``max_surface_cation_sites = 0``, so all eligible exposed sites are used when
defining coverage. A positive value is an explicit search-site cap and therefore
changes the considered coverage denominator.

``placement_side = "dopant-nearest"`` provides a surface-independent automatic
choice for asymmetric doped slabs. DopingFlow determines the slab normal from
the first two lattice vectors, computes the nearest depth of each requested
dopant species from the top and bottom sides, and averages those nearest depths
*per species*. The side with the lower species-balanced depth is selected. This
prevents a species with more atoms from dominating the decision simply by
multiplicity. ``side_target_species`` can explicitly name co-dopants such as
``["Sb", "In"]``; when it is empty, non-host cations are inferred from each
actual slab. If the two sides are tied within
``dopant_side_tie_tolerance_A``, both sides are retained. Manual ``top``,
``bottom``, and ``both`` modes remain available.

Before any ML relaxation, adsorbate/proton arrangements are symmetry-reduced.
The symmetry operations are determined from the **actual doped/vacancy slab**,
not from the pristine parent crystal. Therefore a dopant or vacancy that breaks
a parent symmetry also prevents that symmetry from being used to discard
chemically distinct configurations. Only operations that map the complete
eligible surface-site set onto itself are accepted.

After symmetry reduction, if more inequivalent configurations remain than
``max_arrangements_per_stoichiometry``, DopingFlow selects a diverse subset
using surface-site geometry, adsorbate separation, underlying cation identity,
and proximity to the target dopants. This replaces the former "first N
combinations" behavior. ``max_raw_configurations_per_stoichiometry`` is a
safety cap for exceptionally large combinatorial spaces.

ML relaxations are checkpointed with a structure and calculator fingerprint,
and the output records raw arrangement count, examined count, number of
symmetry-unique arrangements, symmetry operations used, resolved surface side,
eligible-site counts, and actual coverage.

CHE thermodynamics
------------------

A surface state is represented relative to the clean slab by delta_n_H and
delta_n_O. Dopingflow uses the reaction convention

.. math::

   * + \Delta N_O H_2O + x(H^+ + e^-) \rightarrow *_{state},

where

.. math::

   x = \Delta N_H - 2\Delta N_O.

With the Computational Hydrogen Electrode,

.. math::

   \mu_{H^+ + e^-} = \frac{1}{2}G_{H_2} - eU_{SHE}
                     - k_B T \ln(10)\,pH,

so one expression covers protonated, O*, OH*, H2O*, and mixed O/OH states.
The default grid is pH -1 to 3 and 0--2 V vs SHE. Both limits and steps are
configurable. RHE is also supported; at fixed RHE potential the ideal CHE
proton/electron contribution is pH-independent, as expected from the
reference-scale conversion.

Reference and accuracy levels
-----------------------------

H2 and H2O reference energies can be supplied manually or calculated with the
same ML backend used for state screening. Optional free-energy corrections can
be added explicitly. No ZPE, vibrational entropy, configurational entropy,
explicit-solvent, or solvation correction is silently added.

If the surface_pourbaix.dft table is enabled, states that are stable or within
candidate_window_eV of stability anywhere on the ML grid are refined with the
shared provenance-checked GPAW infrastructure. The clean surface is always
included. This limits DFT work while retaining near-boundary states.

Outputs and leaching hand-off
-----------------------------

The stage writes:

* surface_state_summary.csv -- all sampled states and ML/DFT energies;
* pourbaix_grid.csv -- the preferred state at every pH/potential point;
* stable_surface_states.csv -- compact stability domains; and
* leaching_surface_states.csv -- one row for each electrochemically stable
  state, with pourbaix_structure_path for Step 17.

When leaching uses source_mode = "auto", this hand-off table is preferred over
the older clean-surface tables. Thus dissolution is evaluated from the surface
states that are actually stable in the chosen electrochemical window.
