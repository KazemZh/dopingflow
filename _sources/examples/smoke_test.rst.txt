Smoke Test — Minimal Fast Run
==============================

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
   outdir = "random_structures"

   [references]
   reference_mode = "metal"
   host = "SnO2"
   host_dir = "reference_structures/"
   supercell = [ 5, 2, 1]
   metals_ref = [ "Ti","Zr","Nb","Sb","Sn"]
   oxides_dir = "reference_structures/"
   fmax = 0.02
   max_steps = 300
   tf_threads = 1
   omp_threads = 1
   device = "cpu"
   gpu_id = 0
   backend = "m3gnet"
   model = "default"
   task = ""
   optimizer = "bfgs"   
   skip_if_done = false

   [generate]
   poscar_order = ["Ti","Zr","Nb","Sb","Sn","O"]
   seed_base = 2026
   clean_outdir = true

   [scan]
   backend = "m3gnet"
   model = "default"    
   task = "" 
   poscar_in = "POSCAR"
   topk = 10
   symprec = 1e-3
   max_enum = 10
   n_workers = 4
   chunksize = 10
   anion_species = ["O"]
   max_unique = 50000
   skip_if_done = false
   mode = "auto"
   sample_budget = 5000
   sample_batch_size = 64
   sample_patience = 1000
   sample_seed = 42
   sample_max_saved = 10000
   device = "cpu"

   [doping]
   mode = "explicit"
   host_species = "Sn"
   compositions = [
   { Sb = 5.0, Ti = 5.0 },
   { Sb = 5.0, Zr = 5.0 },
   { Sb = 10.0, Nb = 5.0 }
   ]

   [relax]
   backend = "m3gnet"
   model = "default"
   task = ""      
   optimizer = "bfgs"
   fmax = 0.05
   max_steps = 300
   n_workers = 6
   tf_threads = 1
   omp_threads = 1
   skip_if_done = false
   skip_candidate_if_done = false
   device = "cpu"
   gpu_id = 0

   [filter]
   mode = "window"
   window_meV = 50.0
   max_candidates = 12
   skip_if_done = false

   [bandgap]
   skip_if_done = false
   cutoff = 8.0
   max_neighbors = 12
   n_workers = 4
   device = "cpu"

   [formation]
   skip_if_done = true
   normalize = "per_dopant"

   [database]


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
