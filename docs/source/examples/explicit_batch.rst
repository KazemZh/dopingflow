Explicit Example — Multiple Target Compositions
==============================================

Purpose
-------

Generate several explicitly defined compositions in one run.

Each composition produces its own folder under ``random_structures/``.


Run
---

::

   dopingflow run-all -c input.toml


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
     { Sb = 5.0, Ti = 5.0 },
     { Sb = 5.0, Zr = 5.0 },
     { Sb = 10.0, Nb = 5.0 }
   ]

   [generate]
   poscar_order = ["Ti","Zr","Nb","Sb","Sn","O"]
   seed_base = 2026
   clean_outdir = true

   # Other sections identical to previous example


Result
------

Each composition gets its own directory:

::

   random_structures/
       Sb5_Ti5/
       Sb5_Zr5/
       Sb10_Nb5/

Each folder contains full scan, relax, filter, bandgap and formation results.
