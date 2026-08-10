Sequential Example — Gradual Sb Doping
======================================

Purpose
-------

This example performs **gradual sequential Sb doping** of ``SnO2``.

Use this mode when:

- You want to increase the dopant concentration step by step.
- You want each composition to start from the lowest-energy relaxed structure of the previous composition.
- You want to improve structural optimization along a doping path.
- You want to recompute formation and mixing energies later using different oxide references without regenerating or relaxing structures.


Workflow
--------

Run the sequential workflow:

::

   dopingflow sequential-run -c input.toml

If ``[energy_correction].enabled = true``, this command fits or reuses one
signature-matched correction model before starting the composition loop. It
does not refit at every step. Run ``refs-build`` first and use matching
``[references]`` and ``[relax]`` backend/model/task settings.

In ``mode = "full"``, each composition step runs:

::

   generate -> scan -> relax -> filter -> optional bandgap -> formation -> collect

After each step, the lowest-energy relaxed structure is copied to:

::

   sequential_structures/step_xxx_<composition>/best_relaxed/POSCAR

and used as the starting structure for the next composition.


Required Files
--------------

The working directory must contain:

::

   input.toml
   reference_structures/
       SnO2.POSCAR
       Sb2O5.POSCAR
       O2.POSCAR

- ``SnO2.POSCAR``: pristine host oxide structure.
- ``Sb2O5.POSCAR``: dopant oxide reference structure.
- ``O2.POSCAR``: oxygen gas reference structure.


Example input.toml
------------------

::

   [structure]
   outdir = "Sb_sequential_SnO2_mace"

   [references]
   reference_mode = "oxide"
   skip_if_done = false
   fmax = 0.02
   max_steps = 300
   tf_threads = 1
   omp_threads = 1
   device = "cpu"
   gpu_id = 0
   backend = "mace"
   model = "small"
   task = ""
   optimizer = "bfgs"
   host = "SnO2"
   host_dir = "reference_structures/"
   supercell = [2, 2, 5]
   metal_ref = ["Sn", "Sb"]
   metals_dir = "reference_structures/"
   oxides_ref = ["Sb2O5"]
   oxides_dir = "reference_structures/"
   gas_ref = "O2"
   gas_dir = "reference_structures/"
   oxygen_mode = "O-rich"
   muO_shift_ev = 0.0

   [generate]
   poscar_order = ["Sb", "Sn", "O"]
   seed_base = 12345

   [sequential]
   outdir = "sequential_structures"
   mode = "full"

   [doping]
   mode = "enumerate"
   host_species = "Sn"
   must_include = ["Sb"]
   dopants = ["Sb"]
   max_dopants_total = 1
   allowed_totals = [2.5, 5.0, 7.5, 10.0]
   levels = [2.5, 5.0, 7.5, 10.0]

   [scan]
   backend = "mace"
   model = "small"
   task = ""
   poscar_in = "POSCAR"
   topk = 20
   symprec = 0.001
   n_workers = 8
   chunksize = 10
   max_enum = 10000
   max_unique = 5000
   anion_species = ["O"]
   skip_if_done = false
   mode = "auto"
   sample_budget = 10000
   sample_batch_size = 20
   sample_patience = 60
   sample_seed = 42
   sample_max_saved = 1000
   device = "cpu"
   gpu_id = 0

   [relax]
   backend = "mace"
   model = "small"
   task = ""
   relax_mode = "full"
   cell_filter = "frechet"
   optimizer = "bfgs"
   fmax = 0.05
   max_steps = 300
   n_workers = 4
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
   enabled = false
   skip_if_done = false
   cutoff = 8.0
   max_neighbors = 12
   n_workers = 4
   device = "cpu"
   gpu_id = 0
   batch_size = 32

   [formation]
   skip_if_done = false
   normalize = "total"

   [database]
   skip_if_done = false


Recomputing Energies with a Different Reference
-----------------------------------------------

After the full sequential workflow has finished, the same relaxed structures can be reused to recompute formation and mixing energies with a different oxide reference.

For example, change:

::

   oxides_ref = ["Sb2O5"]

to another reference, then rebuild references:

::

   dopingflow refs-build -c input.toml

Then set:

::

   [sequential]
   mode = "recompute_energies"

and rerun:

::

   dopingflow sequential-run -c input.toml

This skips:

::

   generate -> scan -> relax -> filter -> bandgap

and reruns only:

::

   formation -> collect

for the existing sequential structures.


Outputs
-------

Each sequential step writes its own folder:

::

   sequential_structures/
       step_001_Sb2p5/
       step_002_Sb5/
       step_003_Sb7p5/
       step_004_Sb10/

Each step contains:

- ``input_step.json``
- ``random_structures/<composition>/``
- ``best_relaxed/POSCAR``
- ``results_database.csv``
- ``sequential_step_summary.json``

The final merged database is written to the project root:

::

   results_database.csv
