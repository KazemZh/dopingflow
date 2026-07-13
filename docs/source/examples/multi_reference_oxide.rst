Multiple Oxide References and Relative Energies
===============================================

This example evaluates Sb-doped SnO2 against three separate Sb oxide references.
The workflow keeps **one candidate per row** and writes a separate column for
all energies calculated from each oxide reference.

Configuration
-------------

Add every required binary oxide to the existing ``[references].oxides_ref`` list::

   [references]
   reference_mode = "oxide"
   host = "SnO2"
   host_dir = "reference_structures/"
   supercell = [2, 2, 5]
   oxides_dir = "reference_structures/"
   oxides_ref = ["SbO2", "Sb2O3", "Sb2O5"]
   gas_ref = "O2"
   gas_dir = "reference_structures/"

   [formation]
   skip_if_done = false
   normalize = "per_dopant"
   relative_enabled = true
   endpoint_x = "auto"

Both relative-energy parameters remain directly inside the existing
``[formation]`` section. No nested configuration section is required.
``"auto"`` identifies the pure oxide endmember and is recorded as
``endpoint_x = 1.0`` in the output metadata.

Place one POSCAR file for every oxide in ``reference_structures/``:

::

   reference_structures/SbO2.POSCAR
   reference_structures/Sb2O3.POSCAR
   reference_structures/Sb2O5.POSCAR

Then rebuild the cached reference energies before calculating formation
energies::

   dopingflow refs-build -c input.toml
   dopingflow formation -c input.toml
   dopingflow collect -c input.toml

Wide output format
------------------

For one Sb-doped candidate, the output remains one row. The oxide choice is
encoded in the column suffix::

   candidate,x_dopant,E_form_eV_total__SbO2,E_form_eV_total__Sb2O3,E_form_eV_total__Sb2O5,...
   candidate_000,0.05,...,...,...,...

The following values are written for every oxide reference:

- ``E_form_eV_total__<reference>``
- ``E_form_eV_per_atom__<reference>``
- ``E_form_eV_per_cation__<reference>``
- ``E_form_eV_per_dopant__<reference>``
- ``E_mix_eV_total__<reference>``
- ``E_mix_eV_per_atom__<reference>``
- ``E_mix_eV_per_cation__<reference>``
- ``E_mix_eV_per_dopant__<reference>``
- ``E_form_rel_eV_per_cation__<reference>``
- ``E_mix_rel_eV_per_cation__<reference>``

Relative-energy definition
--------------------------

The oxide chemical potentials and atom-balanced mixing reaction already place
each candidate on the host-oxide/dopant-oxide tie-line reference. Applying a
second database-derived endpoint subtraction would double-correct the energy.
Therefore the relative per-cation columns preserve the corresponding
oxide-referenced formation and mixing values. The pure oxide endpoint energy,
the composition-weighted co-doping correction, and the reference identity are
written as separate provenance columns.

Collection and sequential merging preserve these formation-stage values. The
legacy database endpoint calculation is used only when reading an older result
database that has absolute reference-specific columns but no relative columns.

Co-doping
---------

For co-doping, dopingflow evaluates all valid combinations of the listed
binary oxide references. For example, Sb2O3 together with TiO2 is written as
one suffix::

   E_mix_eV_per_cation__Sb2O3__TiO2

The dopant order in these combined suffixes is alphabetical and deterministic.
