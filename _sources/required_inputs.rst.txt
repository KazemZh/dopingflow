.. _input_file_req:

Required Input Files
====================

This page summarizes **all external files required to run the workflow**.

The workflow itself is code-driven, but it depends on a small number
of user-provided structure files and a configuration file.

Overview
--------

At minimum, the following files are required:

1. :ref:`input.toml <input_file_spec>` (workflow configuration).
2. Pristine unit-cell structure of the crystal to be doped (POSCAR format)
3. Optional: host and dopant reference bulk structures (POSCAR format)

All file paths are interpreted relative to the directory
containing ``input.toml`` unless absolute paths are used.


Directory Layout Example
-------------------------

A clean minimal directory structure may look like:

::

   project_root/
       input.toml
       reference_structures/
           base.POSCAR        # pristine unit cell
           host.POSCAR        # host elemental bulk (for formation energies)
           dopant1.POSCAR     # dopant bulk reference
           dopant2.POSCAR     # dopant bulk reference
           dopant3.POSCAR     # dopant bulk reference
           ...

Notes:

- All structure files may be placed inside ``reference_structures/``.
- ``base.POSCAR`` is the pristine crystal structure used for supercell generation.
- The elemental POSCAR files (host.POSCAR, dopant1.POSCAR, dopant2.POSCAR ...) are only required
  if formation energies are computed using local bulk references.
- The exact filenames are user-defined, but must match what is specified
  in ``input.toml``.


ALIGNN Model Directory (Environment Variable)
-------------------------------------------------

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
- Dopant bulk POSCAR files are only needed if:
  - You compute formation energies using local references.
- The workflow does **not** require separate dopant unit-cell structures
  for substitution.


Summary
-------

Minimum to start:

- ``input.toml``
- Pristine POSCAR

To enable full workflow including formation energies and bandgaps:

- Reference bulk structures (or database access)
- ALIGNN model directory
