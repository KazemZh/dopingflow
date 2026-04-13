Explicit Example — Single Target Composition using Oxide References
====================================================================

Purpose
-------

This example generates and evaluates **one explicit co-doped composition using oxides references**.

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
       SnO2.POSCAR
       Sb2O3.POSCAR
       ZrO2.POSCAR
       O2.POSCAR

- ``SnO2.POSCAR``: pristine unit cell for Host strucutre.
- Elemental ``*.POSCAR``: bulk oxide reference structures for chemical potentials.


Example input.toml
------------------

::

   [structure]
   outdir = "random_structures"

   [references]
   reference_mode = "oxide"
   host = "SnO2"
   host_dir = "reference_structures/"
   supercell = [ 5, 2, 1]
   oxides_ref = [ "Sb2O3", "ZrO2"]
   oxides_dir = "reference_structures/"
   gas_ref = "O2"
   gas_dir = "reference_structures/"
   oxygen_mode = "O-rich"
   muO_shift_ev = 0.0
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
