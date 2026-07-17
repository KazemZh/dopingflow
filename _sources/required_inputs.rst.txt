.. _input_file_req:

Required Input Files
====================

This page summarizes **all external files required to run the workflow**.

The workflow itself is code-driven, but it depends on a small number
of user-provided structure files and a configuration file.

Overview
--------

At minimum, the following files are required:

1. :ref:`input.toml <input_file_spec>` (workflow configuration)
2. Pristine unit-cell structure of the crystal to be doped (POSCAR format)

Additional reference structure files may be required depending on:

- the selected reference configuration
- whether formation energies are computed
- whether local structure files are used for references

All file paths are interpreted relative to the directory
containing ``input.toml`` unless absolute paths are used.

Directory Layout Example
------------------------

A clean minimal directory structure may look like:

::

   project_root/
       input.toml
       reference_structures/
           base.POSCAR        # pristine unit cell
           host.POSCAR        # host reference structure
           dopant1.POSCAR     # dopant reference
           dopant2.POSCAR     # dopant reference
           dopant3.POSCAR     # dopant reference
           ...

This flat layout remains valid as long as the file names and paths match
what is specified in ``input.toml``.

Structured layouts are also supported, for example:

::

   project_root/
       input.toml
       reference_structures/
           oxides/
               SnO2.POSCAR
               Sb2O5.POSCAR
               TiO2.POSCAR
           metals/
               Sn.POSCAR
               Sb.POSCAR
               Ti.POSCAR
           gas/
               O2.POSCAR

The exact directory organization is user-defined.

Notes:

- All structure files may be placed inside ``reference_structures/`` or subdirectories below it.
- ``base.POSCAR`` is typically the pristine crystal structure used for supercell generation.
- Reference POSCAR files are only required if the corresponding reference mode or formation-energy workflow is used.
- The exact filenames are user-defined, but must match what is specified in ``input.toml``.

Pristine Structure
------------------

The pristine crystal structure is required for structure generation.

Typical example:

::

   reference_structures/base.POSCAR

This structure is used to:

- build the supercell
- generate substitutional doped structures
- provide the structural starting point for later workflow stages

Reference Structures
--------------------

Reference structure files are used for thermodynamic reference construction
and downstream formation-energy evaluation.

Depending on the selected workflow setup, these may include:

- host reference structures
- dopant reference structures
- oxide reference structures
- gas reference structures such as O₂

Examples of valid local reference files include:

::

   reference_structures/host.POSCAR
   reference_structures/dopant1.POSCAR
   reference_structures/dopant2.POSCAR

or, in a more explicit chemistry-based naming style:

::

   reference_structures/metals/Sn.POSCAR
   reference_structures/metals/Sb.POSCAR
   reference_structures/oxides/Sb2O5.POSCAR
   reference_structures/gas/O2.POSCAR

These files are interpreted according to the settings in ``input.toml``.

ALIGNN Model Directory (Environment Variable)
---------------------------------------------

For bandgap prediction (Step 05), a trained ALIGNN model must be available.

The path must be set via environment variable:

::

   export ALIGNN_MODEL_DIR=/path/to/alignn/model

This directory must contain:

- ``config.json``
- ``checkpoint_*.pt``

Without this variable, Step 05 will fail.

Important Notes
---------------

- Dopant unit-cell POSCAR files are **not required for structure generation**.
  Doping is substitutional and uses the pristine structure only.

- Reference POSCAR files are only needed if:
  - formation energies are computed using local references, or
  - the selected reference mode requires them

- The workflow does **not** require separate dopant unit-cell structures
  for substitution.

- The workflow may use either a flat local file layout or a more structured
  directory layout. Both are valid as long as the paths in ``input.toml`` are correct.

Summary
-------

Minimum to start:

- ``input.toml``
- Pristine POSCAR

To enable full workflow including formation energies and bandgaps:

- Reference structure files as required by your reference setup
- ALIGNN model directory