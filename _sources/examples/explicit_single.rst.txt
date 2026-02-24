Explicit Example — Single Target Composition
===========================================

Purpose
-------

This example generates and evaluates **one explicit co-doped composition**.

Use this mode when:
- You already know the exact dopant percentages.
- You want exactly one generated structure per composition.
- You want the full workflow executed (00–07).

Workflow
--------

Run the complete pipeline:

::

   dopingflow run-all -c input.toml


Required Files
--------------

The working directory must contain:

::

   input.toml
   reference_structures/
       base.POSCAR
       Sn.POSCAR
       Sb.POSCAR
       Zr.POSCAR

- ``base.POSCAR``: pristine unit cell.
- Elemental ``*.POSCAR``: bulk reference structures for chemical potentials.


Example input.toml
------------------

::

   [structure]
   base_poscar = "reference_structures/base.POSCAR"
   supercell = [5, 2, 1]
   outdir = "random_structures"

   [doping]
   mode = "explicit"
   host_species = "Sn"
   compositions = [
     { Sb = 5.0, Zr = 5.0 }
   ]

   [generate]
   poscar_order = ["Zr","Sb","Sn","O"]
   seed_base = 12345
   clean_outdir = true

   [scan]
   topk = 15
   symprec = 1e-3
   max_enum = 50000000
   nproc = 12
   chunksize = 50
   anion_species = ["O"]
   max_unique = 200000
   skip_if_done = true

   [relax]
   fmax = 0.05
   n_workers = 6
   tf_threads = 1
   omp_threads = 1
   skip_if_done = true
   skip_candidate_if_done = true

   [filter]
   mode = "window"
   window_meV = 50.0
   max_candidates = 12
   skip_if_done = true

   [bandgap]
   skip_if_done = true
   cutoff = 8.0
   max_neighbors = 12

   [formation]
   skip_if_done = true
   normalize = "per_dopant"

   [references]
   source = "local"
   bulk_dir = "reference_structures/"
   pristine_poscar = "reference_structures/base.POSCAR"
   fmax = 0.02
   skip_if_done = true


Outputs
-------

- ``reference_structures/reference_energies.json``
- ``random_structures/<composition>/``
  - ``ranking_scan.csv``
  - ``ranking_relax.csv``
  - ``ranking_relax_filtered.csv``
  - ``bandgap_alignn_summary.csv``
  - ``formation_energies.csv``
- ``results_database.csv``
