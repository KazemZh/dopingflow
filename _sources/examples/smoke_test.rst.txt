Smoke Test — Minimal Fast Run
=============================

Purpose
-------

Quickly test installation and pipeline functionality
with reduced computational cost.

This example:
- Uses smaller supercell
- Uses small topk
- Uses top-N filtering
- Can stop early


Run only steps 00–04:

::

   dopingflow run-all -c input.toml --start 0 --stop 4


Example input.toml
------------------

::

   [structure]
   base_poscar = "reference_structures/base.POSCAR"
   supercell = [3, 1, 1]
   outdir = "random_structures"

   [doping]
   mode = "explicit"
   host_species = "Sn"
   compositions = [
     { Sb = 5.0, Ti = 5.0 },
     { Sb = 5.0, Zr = 5.0 }
   ]

   [generate]
   poscar_order = ["Ti","Zr","Sb","Sn","O"]
   seed_base = 1
   clean_outdir = true

   [scan]
   topk = 5
   nproc = 4
   anion_species = ["O"]
   skip_if_done = false

   [relax]
   fmax = 0.08
   n_workers = 2
   skip_if_done = false
   skip_candidate_if_done = false

   [filter]
   mode = "topn"
   max_candidates = 3
   skip_if_done = false

   [references]
   source = "local"
   bulk_dir = "reference_structures/"
   pristine_poscar = "reference_structures/base.POSCAR"
   fmax = 0.03
   skip_if_done = false


Expected Result
---------------

You should obtain:

::

   random_structures/<composition>/
       ranking_scan.csv
       ranking_relax.csv
       ranking_relax_filtered.csv

This confirms:
- generation works
- scan works
- relaxation works
- filtering works
