4. Relaxed-Candidate Filtering
==============================

Implementation
--------------

This stage is implemented in:

::

   src/dopingflow/filtering.py

The public entry point is:

::

   run_filtering(...)


Purpose
-------

This stage selects a **reduced set of relaxed candidates** for downstream
property calculations by filtering the results of the relaxation stage.

It operates on the per-folder relaxation ranking produced in Step 03 and writes:

- a filtered ranking table
- a plain text list of selected candidate names

This stage is a lightweight post-processing step; it does not run any atomistic
calculations.


Inputs
------

This stage uses settings from the following sections of ``input.toml``:

- ``[structure]``: provides the output directory containing structure folders.
- ``[filter]``: defines the filtering strategy and thresholds.

It expects that Step 03 has already produced, in each structure folder:

::

   ranking_relax.csv


Method Summary
--------------

For each structure folder inside ``[structure].outdir``:

1. Read ``ranking_relax.csv``.
2. Keep only candidates with ``status == "ok"``.
3. Determine the minimum relaxed energy:

   - :math:`E_{\min} = \min(E_{\mathrm{relaxed}})`

4. Apply one of two filtering modes:

   - *window mode*: keep candidates within an energy window above :math:`E_{\min}`
   - *top-n mode*: keep the lowest-energy ``N`` candidates

5. Write filtered outputs:

   - ``ranking_relax_filtered.csv``
   - ``selected_candidates.txt``


Filtering Modes
---------------

Window mode
~~~~~~~~~~~

If the filter mode is set to ``window``, candidates are kept if:

.. math::

   E_{\mathrm{relaxed}} - E_{\min} \le \Delta E

where:

- :math:`\Delta E = \mathrm{window\_meV}/1000` (converted from meV to eV)

This selects all structures that lie within a user-defined energy window above
the best relaxed candidate.

Top-n mode
~~~~~~~~~~

If the filter mode is set to ``topn``, candidates are kept by selecting the first
``max_candidates`` entries after sorting by relaxed energy.

This guarantees a fixed number of candidates per structure folder (unless fewer
successful relaxations exist).


Selection Basis
---------------

Filtering is based exclusively on:

- ``energy_relaxed_eV`` from ``ranking_relax.csv``

The filter ignores candidates that:

- are missing required fields
- have non-numeric energies
- have ``status`` values other than ``ok``


Outputs
-------

For each structure folder, this stage writes:

Filtered ranking table
~~~~~~~~~~~~~~~~~~~~~~

::

   ranking_relax_filtered.csv

with columns including:

- ``rank_filtered``: rank within the filtered set (starting at 1)
- ``candidate``: candidate folder name
- ``energy_relaxed_eV``: relaxed energy (eV)
- ``delta_e_eV``: energy relative to the folder minimum
- provenance columns copied from the relaxation stage (e.g. scan rank/signature)
- ``filter_mode``: a string describing the applied filter rule

Selected candidate list
~~~~~~~~~~~~~~~~~~~~~~~

::

   selected_candidates.txt

This is a newline-separated list of candidate folder names, in the same order as
the filtered ranking table.


Command-line Overrides and Forcing
----------------------------------

The implementation supports runtime overrides that can force the filter behavior
independently of the default TOML settings:

- overriding ``window_meV`` forces window mode
- overriding ``topn`` forces top-n mode

A force flag can be used to regenerate outputs even if they already exist.


Reproducibility and Skipping
----------------------------

If:

::

   [filter].skip_if_done = true

and both output files already exist in a folder, that folder is skipped unless
a force option is used.

Since this stage is pure file processing, its results are deterministic given the
input ``ranking_relax.csv`` and filter parameters.


Notes and Limitations
---------------------

- This stage performs **no new calculations**; it filters results from Step 03.
- Filtering is done independently per structure folder; it does not compare energies
  across different compositions/folders.
- Energy windows are applied relative to the minimum energy within each folder, not
  relative to a global minimum.
