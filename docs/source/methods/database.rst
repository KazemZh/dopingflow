7. Database Collection
======================

Implementation
--------------

This stage is implemented in:

::

   src/dopingflow/collect.py

The public entry point is:

::

   run_collect(...)


Purpose
-------

This stage consolidates results from previous workflow stages into a **single flat CSV**
that can be used as a lightweight database for downstream analysis, plotting, and reporting.

The database is written to the workflow root as:

::

   results_database.csv

Only **filtered / selected candidates** are included (strict policy), so the database
represents the subset of structures that passed your selection criteria.


Inputs
------

This stage uses settings from two sections of ``input.toml``:

- ``[structure]``: provides the workflow output directory containing composition folders.
- ``[database]``: controls skipping/overwriting of the database CSV.

It expects the standard workflow outputs inside each composition folder (if present):

Composition-level (folder-level)
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

- ``metadata.json`` (from Step 01 generation)
- ``selected_candidates.txt`` (from Step 04 filtering) **preferred**
- ``ranking_relax_filtered.csv`` (from Step 04 filtering) **fallback**
- ``ranking_scan.csv`` (from Step 02 scanning)
- ``bandgap_alignn_summary.csv`` (from Step 05 bandgap prediction)
- ``formation_energies.csv`` (from Step 06 formation energies)

Candidate-level
~~~~~~~~~~~~~~~

- ``candidate_*/02_relax/meta.json`` (from Step 03 relaxation)


Selection Policy
----------------

This stage follows a **strict selection rule**: it will never include unfiltered candidates.

Selection priority within each composition folder:

1. If ``selected_candidates.txt`` exists:
   use exactly those candidate names (one per line).

2. Else, if ``ranking_relax_filtered.csv`` exists:
   include the candidates listed there.

3. Else:
   the folder is skipped (no candidates are collected).

This ensures that the database represents only the candidates you explicitly kept
after filtering.


Method Summary
--------------

For each composition folder inside ``[structure].outdir``:

1. Determine the candidate list using the selection policy above.
2. Load composition metadata from ``metadata.json`` (if available).
3. Load per-stage summary tables (if available):

   - scan ranking table
   - filtered ranking table
   - bandgap summary table
   - formation energy table

4. For each selected candidate:

   a. Read relaxed-energy metadata from ``candidate_*/02_relax/meta.json`` (if available).
   b. Combine composition-level and candidate-level information into one row.

5. Write all rows to:

::

   results_database.csv


Database Schema
---------------

The output database contains the following columns:

Composition-level fields
~~~~~~~~~~~~~~~~~~~~~~~~

- ``composition_tag``:
  Folder name of the composition (effective tag used by the generator).

- ``requested_index``:
  Index of the composition in the generator loop (if available).

- ``requested_pct_json``:
  JSON string of requested dopant percentages (if available).

- ``effective_pct_json``:
  JSON string of effective dopant percentages after rounding (if available).

- ``rounded_counts_json``:
  JSON string of rounded integer substitution counts (if available).

- ``host_species``:
  Host species used for substitution (from generation metadata).

- ``n_host``:
  Number of host sites in the supercell (from generation metadata).

- ``supercell_json``:
  JSON string of the supercell dimensions (if available).

Candidate-level identity
~~~~~~~~~~~~~~~~~~~~~~~~

- ``candidate``:
  Candidate folder name (e.g., ``candidate_001``).

- ``candidate_path``:
  Absolute path to the candidate folder (useful for linking back to files).

Relax (filtered) fields
~~~~~~~~~~~~~~~~~~~~~~~

- ``rank_relax_filtered``:
  Rank within the filtered set (if available).

- ``E_relaxed_eV_filtered``:
  Relaxed energy read from the filtered ranking table (if available).

Scan fields
~~~~~~~~~~~

- ``rank_scan``:
  Single-point rank from scanning stage (if available).

- ``E_scan_eV``:
  Single-point energy from scanning stage (if available).

Relax (meta) fields
~~~~~~~~~~~~~~~~~~~

- ``E_relaxed_eV``:
  Relaxed energy extracted from ``candidate_*/02_relax/meta.json`` (if available).

Bandgap fields
~~~~~~~~~~~~~~

- ``bandgap_eV``:
  Predicted bandgap value from the bandgap summary (if available).

Formation-energy fields
~~~~~~~~~~~~~~~~~~~~~~~

- ``E_form_eV_total``:
  Total formation energy in eV (if available).

- ``E_form_norm``:
  Normalized formation energy (as written by Step 06, depends on your normalization mode).

- ``n_dopant_atoms``:
  Total dopant atoms used in the candidate (if available).

- ``dopant_counts``:
  Compact dopant-count string from Step 06 (if available).


Outputs
-------

This stage writes one file in the workflow root:

::

   results_database.csv

Each row corresponds to **one selected candidate** in **one composition folder**.


Reproducibility and Skipping
----------------------------

If:

::

   [database].skip_if_done = true

and ``results_database.csv`` already exists, the stage is skipped.

Set ``skip_if_done = false`` to overwrite and regenerate the database.


Notes and Limitations
---------------------

- The database is intentionally flat and file-based; it is designed for quick use
  in pandas, spreadsheets, or plotting scripts.
- Missing upstream files are handled gracefully: columns are left empty/``None``
  when a stage output is unavailable.
- Only the filtered/selected candidate subset is included by design.
  If you later want a “full database” including all candidates, the selection
  policy can be relaxed as an optional mode.
