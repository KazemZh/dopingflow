Enumerate Example — Systematic Dopant Screening
================================================

Purpose
-------

Screen multiple dopants and concentrations automatically.

Use this mode when:
- You want combinatorial exploration.
- You define allowed dopants and doping levels.
- You limit the maximum number of dopants per structure.

Note
----

The current implementation supports **up to 3 dopants per structure**
via:

::

   max_dopants_total = 3


Run
---

::

   dopingflow run-all -c input.toml


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
   seed_base = 12345
   clean_outdir = true

   [doping]
   mode = "enumerate"
   host_species = "Sn"
   dopants = ["Sb", "Ti", "Zr", "Nb"]
   must_include = ["Sb"]
   max_dopants_total = 3
   allowed_totals = [5.0, 10.0, 15.0]
   levels = [5.0, 10.0, 15.0]

   # Other sections identical to previous example


Result
------

The workflow automatically generates composition folders like:

::

   random_structures/
       Sb5/
       Sb5_Ti5/
       Sb5_Ti5_Zr5/
       Sb10_Nb5/
       ...

Each is processed independently through all stages.
