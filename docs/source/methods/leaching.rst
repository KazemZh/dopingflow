Dopant leaching from selected surfaces
======================================

The leaching stage runs after :doc:`surfaces`. It removes exposed dopant atoms
from the already generated/relaxed slabs, relaxes the dopant-vacancy slab, and
reports a thermodynamic leaching descriptor while preserving the full surface
provenance (bulk target, vacancy state, Miller index, termination, variant,
dopant, and atom index).

This is a thermodynamic screening stage. It is not a kinetic dissolution-rate
simulation or a complete surface Pourbaix model.

Metal-referenced extraction energy
----------------------------------

For one surface dopant atom `M`, DopingFlow evaluates

.. math::

   E_\mathrm{extract}
   = E_{\mathrm{slab-M}} + \mu_M^\mathrm{metal}
   - E_{\mathrm{slab+M}}.

The dopant-removed slab is relaxed unless `relax_removed_surface = false`.
A larger positive `E_extract` means stronger retention of the dopant against
this neutral, elemental-metal-referenced extraction reaction. Lower values mean
a more weakly retained surface dopant.

The metal chemical potential is resolved in this order:

* `manual_metal_chemical_potentials_eV` in `[leaching]`;
* a compatible metal entry in
  `reference_structures/reference_energies.json`;
* an on-demand calculation of
  `reference_structures/metals/<element>.POSCAR` with the same leaching
  calculator when `compute_missing_metal_references = true`.

A cached reference is reused only when backend/model/task provenance matches.

Optional electrochemical dissolution potential
-----------------------------------------------

A surface-alloy dissolution treatment can reference the surface binding
correction to the pure-metal dissolution potential. This idea follows the
first-principles surface-alloy approach of Greeley and Nørskov
(Electrochimica Acta 52, 5829-5836, 2007,
DOI: 10.1016/j.electacta.2007.02.082), and related doped-oxide screening
work expresses the dissolution potential as a bulk standard dissolution
potential corrected by a dopant stabilization energy (J. Mater. Chem. A 14,
3297-3307, 2026, DOI: 10.1039/D5TA08166A).

When a validated reduction reaction

.. math::

   M^{z+} + z e^- \rightleftharpoons M(s)

is supplied by the user, the implemented simple-ion model is

.. math::

   \Delta G_\mathrm{leach}(U_\mathrm{SHE})
   = E_\mathrm{extract}
   + z(E^\circ_{M^{z+}/M} - U_\mathrm{SHE})
   + k_B T \ln a_M.

The zero-crossing potential is therefore

.. math::

   U_\mathrm{diss,SHE}
   = E^\circ_{M^{z+}/M}
   + \frac{E_\mathrm{extract}}{z}
   + \frac{k_B T}{z}\ln a_M.

Above this threshold, `deltaG_leach_eV < 0` for the modeled oxidation
reaction, so dissolution is thermodynamically open in this simplified model.
Lower dissolution potentials therefore indicate greater vulnerability.

For RHE reporting/input,

.. math::

   U_\mathrm{RHE}
   = U_\mathrm{SHE} + k_B T\ln(10)\,\mathrm{pH}

when energy is expressed in eV per electron.

Why redox data are not hard-coded
---------------------------------

DopingFlow deliberately does **not** guess standard potentials, dissolved
species, or oxidation states. Several oxide dopants can dissolve as hydrolyzed,
oxo-containing, or multiple oxidation-state species rather than one bare
`M^z+` ion. A chemically inappropriate redox couple would produce a precise
but misleading number.

Without redox data the extraction-energy calculation still runs normally and
the electrochemical fields are marked `missing-redox-reference`.

The present simple-ion model does not include explicit water, charged slabs,
potential-dependent surface hydroxylation, aqueous complex speciation, kinetic
barriers, or multi-atom dissolution pathways. Those are appropriate later
high-fidelity extensions for the most promising surfaces.

Configuration
-------------

Minimal extraction-only configuration:

.. code-block:: toml

   [leaching]
   enabled = true
   source_mode = "auto"
   zones = ["surface"]
   dopant_species = []        # infer from surface metadata
   outdir = "09_leaching"

   # By default these inherit the enabled [surface.refine] calculator
   # (or [surface.screen] if refinement is disabled).
   backend = "mace"
   model = "mh-1"
   task = "matpes_r2scan"
   device = "cuda"
   gpu_id = 0
   optimizer = "bfgs"
   fmax = 0.03
   max_steps = 500

   reuse_surface_energy = true
   relax_parent_if_recomputed = true
   relax_removed_surface = true
   inherit_surface_fixed_atoms = true

   reference_energies_file = "reference_structures/reference_energies.json"
   metals_dir = "reference_structures/metals"
   compute_missing_metal_references = true
   relax_metal_reference = false

For the electrochemical extension, add only validated values for the chosen
aqueous species:

.. code-block:: toml

   [leaching]
   # ...settings above...

   oxidation_states = { Sb = 3, Ti = 3 }
   # standard_reduction_potentials_V_SHE = { Sb = <validated>, Ti = <validated> }
   aqueous_species = { Sb = "chosen Sb species", Ti = "chosen Ti species" }
   ion_activities = { Sb = 1e-6, Ti = 1e-6 }

   default_ion_activity = 1e-6
   temperature_K = 298.15
   pH = 0.0
   potential_scale = "RHE"
   potentials_V = [1.23, 1.50, 1.70]

A JSON file can also be selected with `redox_reference_file`. Inline TOML
values override matching JSON fields.

Surface selection
-----------------

`source_mode = "auto"` uses the most selective available table in this order:
`surface_final_selected.csv`, `surface_screen_selected.csv`,
`surface_refine_summary.csv`, then `surface_screen_summary.csv`.

Other accepted modes are `final-selected`, `screen-selected`,
`refine-summary`, and `screen-summary`.

`surface_include` accepts exact IDs or shell-style wildcards.
`zones` defaults to `["surface"]`; add `subsurface` or `bulk` for
controlled depth comparisons. The surface-stage layer classification is
inherited by default.

Commands
--------

Preview the exact atom-removal jobs without running the calculator:

.. code-block:: bash

   dopingflow leaching -c input.toml --dry-run

Run the calculation:

.. code-block:: bash

   dopingflow leaching -c input.toml

The stage is also available as step 15 of `run-all`.

Outputs
-------

The default directory is `09_leaching/`:

`leaching_preview.csv`
   Exact dopant atoms selected before calculation, with coordinates, depth zone,
   local O coordination, facet, termination, and parent target.

`leaching_summary.csv`
   One row per removed dopant site with parent and defective energies, metal
   reference provenance, extraction energy, optional electrochemical quantities,
   and convergence information.

`leaching_surface_summary.csv`
   Per-surface/per-dopant minima and the most vulnerable tested site.

`leaching_potential_scan.csv`
   `Delta G_leach` at every configured potential and a boolean
   thermodynamic-favorability flag.

`leaching_results.json`
   Complete machine-readable results plus the model assumptions.

Each site also has a directory containing `POSCAR_removed_unrelaxed`, the
relaxation output, and `leaching_result.json`.

Interpretation
--------------

Use `extraction_energy_eV` first as the model-consistent structural retention
metric. Use `dissolution_potential_V_SHE/RHE` and `deltaG_leach_eV` only
when the aqueous species, electron count, standard potential, activity, and
potential scale are chemically consistent across the structures being compared.
