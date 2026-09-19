Restricted One-Dimensional Alloy Hull
=====================================

The ``alloy-hull`` stage builds a **restricted one-dimensional convex hull**
for a single substitutional alloy path. For example, for Sb substitution in
SnO2 it uses the fixed-composition line:

.. math::

   \mathrm{Sn}_{1-x}\mathrm{Sb}_x\mathrm{O}_2.

This stage is different from ``phase-diagram``:

- ``alloy-hull`` compares only SnO2-type host, the selected Sb substitutional
  candidates, and an optional stoichiometrically compatible endpoint such as
  SbO2.
- ``phase-diagram`` is the full multicomponent Sn-Sb-O stability calculation;
  it may decompose a candidate into phases outside the Sn_(1-x)Sb_xO2 line.

Energy used for the hull
------------------------

The stage uses direct relaxed total energies, normalized per cation:

.. math::

   e_\sigma = \frac{E_\sigma}{N_{\mathrm{Sn}} + N_{\mathrm{Sb}}}.

At each composition, only the lowest-energy candidate is retained. The lower
convex envelope is then constructed from these minima, the relaxed host at
``x = 0``, and the endpoint reference at ``x = 1`` when one is available.

For a candidate at ``x`` lying between two neighbouring hull vertices
``(x_1, e_1)`` and ``(x_2, e_2)``, the stage computes:

.. math::

   e_{\mathrm{hull}}(x)
   = e_1 + \frac{x-x_1}{x_2-x_1}(e_2-e_1),

and reports:

.. math::

   E_{\mathrm{above\ 1D\ hull}}(x)
   = e_\sigma(x)-e_{\mathrm{hull}}(x).

This is reference-independent for a fixed alloy line: adding any term linear
in ``x`` to every energy, such as the oxide-reference or oxygen-chemical-
potential contribution, changes the energy zero and slopes but does not change
the hull vertices or energy above the restricted 1D hull.

Configuration
-------------

The section is optional. The defaults are appropriate for a single-dopant
workflow::

   [alloy_hull]
   dopant = "auto"
   endpoint_reference = "auto"

``dopant = "auto"`` detects the unique dopant in the result database. When
multiple dopants are present, specify one explicitly, for example::

   [alloy_hull]
   dopant = "Sb"
   endpoint_reference = "SbO2"

``endpoint_reference = "auto"`` searches the cached oxide references for a
binary dopant oxide with the same anion-to-cation ratio as the host line. For
SnO2 doped with Sb, it selects SbO2. Set
``endpoint_reference = "none"`` to build a hull only over the sampled range;
this is useful when no physical x = 1 endpoint has been calculated or cached.

Run
---

Run after collection::

   dopingflow collect -c input.toml
   dopingflow alloy-hull -c input.toml

The ``run-all`` command also includes this stage between ``collect`` and
``phase-diagram``.

Outputs
-------

The stage writes:

- ``alloy_hull_results.csv``: one row per relaxed candidate, including
  ``energy_per_cation_eV``, ``hull_energy_1d_eV_per_cation``,
  ``energy_above_1d_hull_eV_per_cation``, ``on_1d_hull``, and the predicted
  1D decomposition between neighbouring hull vertices.
- ``alloy_hull_vertices.csv``: the actual lower-hull vertices, including host
  and endpoint reference points.
- ``alloy_hull_summary.json``: stage metadata and selected endpoint reference.

The same 1D hull fields are also appended to ``results_database.csv`` so they
can be used directly in Jupyter plotting scripts.
