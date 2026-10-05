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

Coverage is specified as a percentage of **all eligible sites** on the chosen
surface side(s), rather than as an absolute atom count. Separate grids are
available for protonated lattice O, O*, OH*, and H2O*, plus explicit O*:OH*
coverage pairs. A 0% entry is accepted for these single-family coverage lists
but is not generated as a separate state because it is exactly the already
present clean slab. For each surface, DopingFlow converts the requested percentage
to the closest integer occupation allowed by that finite surface cell. When a
requested value lies exactly halfway between two integer occupations, both are
retained. For example, 25% coverage on 12 sites maps to 3 occupied sites,
whereas 25% on 10 sites maps to both 2/10 (20%) and 3/10 (30%). Requested and
actual coverages are both written to the state summary.

The old ``max_surface_oxygen_sites`` and ``max_surface_cation_sites``
settings are accepted only for backward compatibility and are now **ignored**.
They previously changed the physical coverage denominator; for example a saved
cap of 8 could make "100%" protonation add only 8 H even when 10 O atoms were
eligible. The arrangement-count controls, symmetry reduction, and diversity
selection are the correct performance controls and do not redefine coverage.

Surface-site detection defaults to ``surface_site_selection =
"outermost-layer"``. The ``surface_window_A`` value is then only a maximum
normal-depth safety window. Within that window, DopingFlow identifies the
largest normal-direction spacing gap that exceeds
``surface_layer_gap_A`` and treats it as the boundary between the exposed
surface layer/group and deeper crystallographic layers. This prevents a
corrugated oxide surface from accidentally protonating or adsorbing on the next
subsurface layer merely because it lies within 2 Å of the outermost atom.
The legacy ``surface_site_selection = "window"`` mode remains available for
comparison and intentionally reproduces the former height-window behavior.

Adsorbate placement and lattice-O protonation have independent side controls.
``placement_side`` applies to O*, OH*, H2O*, and mixed O*/OH* states.
``protonation_side`` applies only to H added to existing lattice O and accepts
``both`` (default), ``same-as-adsorbates``, ``dopant-nearest``,
``top``, or ``bottom``. With ``protonation_side = "both"``, the
coverage denominator includes eligible exposed O atoms from both slab faces, so
100% protonation adds one H to every eligible O on both faces.

``placement_side = "dopant-nearest"`` (and independently
``protonation_side = "dopant-nearest"``) provides a surface-independent
automatic choice for asymmetric doped slabs. DopingFlow determines the slab
normal from the first two lattice vectors, computes the nearest depth of each
requested dopant species from the top and bottom sides, and averages those
nearest depths *per species*. The side with the lower species-balanced depth is
selected. This prevents a species with more atoms from dominating the decision
simply by multiplicity. ``side_target_species`` can explicitly name
co-dopants such as ``["Sb", "In"]``; when it is empty, non-host cations are
inferred from each actual slab. If the two sides are tied within
``dopant_side_tie_tolerance_A``, both sides are retained. Manual top, bottom,
and both modes remain available.

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

Parallel surface execution
--------------------------

Independent selected surfaces can be evaluated concurrently with
``parallel_surfaces = true`` and ``surface_workers = N``. Parallelism is
deliberately applied at the **surface level**, not by launching multiple states
of the same surface into a shared calculator. Each worker process owns one
surface at a time, initializes its own ML calculator, generates/screens/relaxes
that surface's states, performs post-relaxation validation, writes only to that
surface directory, and returns its summary to the parent process. The parent
then aggregates the combined CSV and Pourbaix outputs in the original target
order.

H2 and H2O ML reference energies are evaluated once before workers start.
Existing compatible state checkpoints remain reusable inside each worker, so
enabling parallelism does not disable restart/recovery behavior.

For CPU execution, the effective worker count is the smaller of
``surface_workers`` and the number of selected surfaces. Users should avoid
CPU oversubscription: approximately
``surface_workers x per-worker threads`` should fit within the available
physical cores. For a single CUDA device, DopingFlow intentionally uses one
effective surface worker because loading several independent foundation-model
copies onto the same GPU can exhaust VRAM or reduce throughput. Local
surface-process parallelism is also disabled while missing GPAW calculations
are being executed, avoiding nested multiprocessing/MPI oversubscription.
Cached/non-executing DFT lookups do not impose that restriction.

The requested/effective worker counts and the reason for any safety fallback are
written to ``surface_pourbaix_summary.json`` and printed in the run log.

Post-relaxation chemistry validation
------------------------------------

The generated state name is a starting hypothesis, not the final phase label.
After each ML relaxation, DopingFlow inspects the relaxed connectivity before
the energy is allowed into the Pourbaix competition.

The validator tracks newly added H/O atoms relative to the parent slab and
classifies:

* protonated original lattice oxygen;
* surface-bound O*, OH*, and H2O*;
* short added-O--added-O motifs (O2/peroxo-like structurally);
* added-O--lattice-O reconstructions;
* detached O, OH, H2O, O2-like, and larger oxygen fragments;
* unbound H / H2-like fragments; and
* adsorption/protonation-site changes at otherwise unchanged coverage.

Distance thresholds are configurable. A short O--O distance is deliberately
reported as an ``O-O-like`` structural motif; the workflow does **not** infer
the electronic identity O2 vs superoxo vs peroxo from bond length alone.

Each state receives a ``state_status``:

* ``retained`` -- intended chemistry and site pattern survived;
* ``reconstructed_same_coverage`` -- same chemistry/coverage, different sites;
* ``reclassified`` -- chemistry changed but the products remain surface-bound;
* ``desorbed`` -- one or more fragments detached from the slab;
* ``fragmented`` -- unbound/abnormal fragments formed;
* ``invalid`` / ``calculation_failed`` -- unusable relaxation.

By default, desorbed and fragmented states are excluded from the Pourbaix
competition. Surface-bound reclassified states remain eligible and use their
**final relaxed chemistry** in the plot label. Thus an intended O*/OH* state
that forms detached O2 is not mislabeled as a stable high-O coverage phase,
whereas a bound reconstruction such as 2OH* -> O* + H2O* can be retained under
a reconstructed final-state label.

The state summary records requested and final labels, final coverages, counts
of surface-bound/desorbed species, the validation reason, and the exact relaxed
structure path for inspection.

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

Interactive structure comparison
--------------------------------

The Results section includes a ``Structure comparison`` tab. For the selected
surface, it lists the states that are stable somewhere on the U-pH map by
default, with an option to include all calculated states. The selected state is
shown in two interactive 3D viewers:

* ``Before relaxation`` -- the generated protonated/adsorbate configuration
  saved as ``POSCAR_initial``;
* ``After ML relaxation`` -- the relaxed geometry saved as ``POSCAR_relaxed``.

The exact paths are printed above each viewer. The tab also exposes the parent
surface path, the state calculation directory, requested/final chemistry, and
post-relaxation validation status. Older runs remain viewable because the GUI
can infer ``POSCAR_initial`` from the existing state directory even when the
older summary CSV did not yet contain ``ml_initial_structure_path``.

Surface-Pourbaix visualization
-----------------------------

The Streamlit results view renders the U-pH grid as a **filled discrete phase
map**, rather than as individual square scatter markers. The plotted category is
the stable chemistry/coverage state. Phase boundaries are drawn between adjacent
grid cells with different stable categories.

Legend labels intentionally omit internal arrangement identifiers such as
``arr006``. New runs use the actual finite-cell coverage recorded during state
generation, for example ``Protonated lattice O — 33.3%``, ``O* — 20%``, or
``Mixed O*/OH* — 20% O* + 30% OH*``. The exact ``state_id``, arrangement,
resolved slab side, symmetry statistics, and CHE free energy remain available
in hover text and the provenance table.

Legacy result files generated before coverage metadata was added are still
viewable, but the GUI warns that a rerun is required before percentage-based
phase labels can be shown reliably.

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
