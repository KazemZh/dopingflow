Surface electrochemistry and Pourbaix diagrams
==============================================

The surface-pourbaix command is Step 16, between surface segregation and
leaching. Its purpose is not to assume that adding protons stabilizes an oxide
surface. Instead, it compares intact-surface chemical states and asks which
state is thermodynamically preferred as a function of electrode potential and
pH.

Input and state search
----------------------

With source_mode = "auto" the stage first looks for
surface_segregation_summary.csv and uses each segregated POSCAR_best_mc. If no
segregation output exists it falls back to the selected/refined surface
tables. The generated state families are:

* clean surface;
* protonated surface oxygen sites;
* O* adsorbates;
* OH* adsorbates;
* H2O* adsorbates; and
* mixed O*/OH* states.

Coverage and arrangement are separate degrees of freedom. The h_counts and
adsorbate_counts settings choose coverages, while
max_arrangements_per_stoichiometry limits the deterministic enumeration of
site arrangements. ML relaxations are checkpointed with a structure and
calculator fingerprint.

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
