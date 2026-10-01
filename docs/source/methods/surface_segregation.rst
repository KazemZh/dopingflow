Surface Segregation Monte Carlo
=================================

Purpose
-------

The surface-segregation stage samples where substitutional dopants prefer to
occupy the cation sublattice of a **user-selected natural surface termination**
at finite temperature.

It is intentionally separate from surface generation:

::

   natural surface termination
       -> choose exact surfaces
       -> classify cation sites as surface / subsurface / bulk
       -> fixed-composition host <-> dopant Monte Carlo
       -> occupancy statistics
       -> effective segregation free energies

The surface stage itself never swaps dopants. Swaps are introduced only here,
where they are part of a controlled statistical-mechanical sampling procedure.

Running the stage
-----------------

The command is::

   dopingflow surface-segregation -c input.toml

Use ``--dry-run`` to resolve and write the selected-surface MC plan without
loading the ML force field::

   dopingflow surface-segregation -c input.toml --dry-run

The Streamlit GUI provides the same controls on the ``Surface Segregation MC``
page.

Surface selection
-----------------

The stage reads one of the existing natural-surface tables:

- ``surface_final_selected.csv``
- ``surface_refine_summary.csv``
- ``surface_screen_selected.csv``
- ``surface_screen_summary.csv``

``source_mode = "auto"`` chooses the first available table in that order.
An explicit CSV can be supplied with ``source_summary``.

Exact surfaces are selected with stable ``surface_id`` values. The GUI exposes
these as checkboxes, together with Miller index, termination ID, and the natural
dopant-depth label inherited from the slab cut.

Monte Carlo ensemble
--------------------

The default sampling is a fixed-composition canonical occupation MC.

At every trial move, one dopant and one host cation exchange identities::

   X@i + Sn@j  ->  Sn@i + X@j

where ``X`` is one of the selected dopant species.

The lattice and Cartesian coordinates are fixed during the Markov chain. Only
cation identities change. Therefore:

- the total numbers of Sn, Sb, In, etc. remain constant;
- oxygen atoms are not moved or removed;
- the slab geometry and vacuum remain unchanged during sampling;
- the single-point MLFF energy determines Metropolis acceptance.

For an energy change ``Delta E = E_new - E_old`` the move is accepted with::

   P_acc = min(1, exp(-Delta E / (k_B T)))

This fixed-geometry occupation model is designed for long configurational
sampling. Structural relaxation of every MC proposal is deliberately avoided
because it would make trajectories orders of magnitude more expensive and
would mix occupation sampling with local geometry optimization.

Cation depth zones
------------------

Cation sites are classified once from the selected starting slab and remain
fixed throughout the MC trajectory.

With ``dopant_depth_layers = 1``:

- outermost cation layer on each slab side -> ``surface``;
- next cation layer on each side -> ``subsurface``;
- all remaining cation layers -> ``bulk``.

The cation layers are grouped using ``cation_layer_tolerance_A``.

Both exposed slab sides are included. The same site-zone definition is used for
all production samples.

Sampling controls
-----------------

Example::

   [surface_segregation]
   enabled = true
   source_mode = "final-selected"
   surface_include = [
     "In2p5_Sb2p5/candidate_001|hkl=1,1,0|term=001",
   ]

   host_species = "Sn"
   anion_species = ["O"]
   dopant_species = ["In", "Sb"]

   temperature_K = 800.0
   steps = 100000
   burn_in = 20000
   sample_interval = 20
   seed = 42

   backend = "mace"
   model = "mh-1"
   task = "matpes_r2scan"
   device = "cuda"
   gpu_id = 0

``dopant_species`` may be left empty to infer every non-host, non-anion cation
present in the selected surface.

``parallel_surfaces`` can run independent surface terminations concurrently on
CPU. A single Metropolis chain remains sequential by construction. CUDA uses
one surface chain at a time.

Backend-aware GUI controls
--------------------------

The Surface Segregation GUI changes the model/task controls immediately when the
backend changes, so values from one backend are not carried into another one:

- ``m3gnet``: uses ``model = "default"`` and no task/head;
- ``grace``: shows a GRACE-model dropdown and disables task/head;
- ``uma``: shows a UMA-model dropdown plus a UMA task dropdown;
- ``mace``: shows the available MACE model catalogue (plus a custom-checkpoint
  option). A head field is shown only for multi-head/custom checkpoints; for
  ``mh-1`` the default head is ``matpes_r2scan``.

When possible, the MACE dropdown is populated from the models supported by the
installed MACE package; otherwise DopingFlow falls back to its built-in model
list.

Production statistics
---------------------

Only samples after ``burn_in`` contribute to occupancy statistics. Samples are
recorded every ``sample_interval`` steps.

For each dopant and each cation site the workflow records the occupation
probability::

   p_i(X) = number of sampled configurations with X on site i / n_samples

This produces the site-resolved occupancy heat map.

For the three depth zones, raw counts are normalized by the number of available
cation sites in that zone. For dopant ``X``::

   rho_zone(X) =
       total X observations in zone /
       (n_samples * number of cation sites in zone)

This normalization is essential. A bulk region can contain many more sites than
the surface and therefore should not be judged from raw visit counts alone.

Effective segregation free energy
---------------------------------

The workflow reports an occupancy-derived effective free-energy preference
relative to the bulk-like region::

   Delta G_seg,eff(zone -> bulk)
       = -k_B T ln[rho_zone / rho_bulk]

Interpretation:

- negative ``Delta G_seg,eff``: the dopant is enriched in that zone relative to
  the bulk-like region;
- zero: equal per-site occupancy to the bulk reference;
- positive: the zone is depleted relative to bulk.

This quantity is a finite-temperature **effective potential of mean force /
occupancy-derived segregation free energy**. It is not the old static
zero-temperature difference between two hand-constructed structures.

At finite dopant concentration, especially when dopant-dopant interactions are
strong, it should be interpreted as an effective population-derived
thermodynamic preference rather than an isolated single-dopant transfer free
energy.

If a dopant is never sampled in the bulk region, a finite bulk-referenced
logarithm cannot be formed and the corresponding value is left undefined rather
than introducing an arbitrary pseudocount.

Site-resolved PMF
-----------------

For visualization, each cation site also receives a relative site PMF::

   Delta G_i = -k_B T ln[p_i / p_max]

where ``p_max`` is the most occupied site for that dopant. The most preferred
sampled site therefore has ``Delta G_i = 0``.

Outputs
-------

The default output root is::

   <surface source root>/09_surface_segregation/

Important files are::

   mc_plan.json
   surface_segregation_summary.csv
   surface_segregation_summary.json
   config_resolved.json

   surfaces/<safe-surface-id>/
       POSCAR_start
       POSCAR_best_mc
       POSCAR_final_mc
       cation_site_zones.csv
       site_occupancy.csv
       zone_occupancy.csv
       mc_trace.csv
       zone_trace.csv
       swap_statistics.csv
       summary.json

The GUI visualizes:

- normalized surface/subsurface/bulk occupancy heat maps;
- site-resolved occupancy heat maps;
- effective ``Delta G_seg`` relative to bulk;
- site-resolved PMF;
- MC energy convergence;
- depth-zone occupancy versus MC step;
- per-dopant move acceptance statistics.

Convergence
-----------

A useful segregation conclusion requires more than a completed run. Check:

- energy trace stationarity after burn-in;
- stable surface/subsurface/bulk populations;
- adequate acceptance ratio;
- sufficient production samples;
- repeatability with different random seeds;
- sensitivity to temperature and MLFF choice.

For publication-quality finite-temperature conclusions, independent chains and
temperature/trajectory-length convergence should be tested. The current
single-chain statistics are intended as the first production implementation,
not a substitute for convergence analysis.
