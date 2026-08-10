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
- whether the optional experimental energy correction is enabled

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
           corrections/                 # only when energy correction is enabled
               calibration_manifest.csv # required by manifest selection
               calibration_structures/
                   TiO2.POSCAR
                   ZrO2.POSCAR

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

Optional Energy-Correction Inputs
---------------------------------

When ``[energy_correction].enabled = true``, the additional inputs depend on
``calibration_selection``:

- With ``calibration_selection = "manifest"``, ``calibration_manifest`` is
  required. Every included row must point to an existing ``structure_path`` so
  composition, phase, space group, and oxygen environment can be checked. Paths
  are resolved relative to the manifest first and then relative to the project.
- With ``calibration_selection = "phase_resolved"``, dopingflow selects all
  strict non-generic ordinary oxide records whose non-oxygen elements are in
  the workflow's complete host-and-dopant scope. Exact phase-matched manifest structures are
  reused when present. If ``auto_fetch_phase_structures = true``, missing
  structures are retrieved by curated ``likely_mpid`` from the configured
  OPTIMADE endpoint, so the manifest itself may be absent. If fetching is false,
  the manifest must materialize every selected record.
- ``experimental_data`` is required only for
  ``experimental_source = "custom"`` or ``"kingsbury+custom"``. It is not
  required for the curated Kingsbury-only source.

The calibration structures are independent of the competing phases listed in
``[references].oxides_ref``. A structure remains mandatory even when its
``energy_total_eV`` is precomputed. Such a row must also provide exact
backend/model/task, installed backend version, calculation-settings text and
signature hash, plus ``converged = true``. If the energy-above-hull filter is
enabled, it additionally requires same-backend hull provenance and matching
hull backend/model/task, backend-version, and calculation-settings-hash
columns. See :doc:`methods/energy_corrections` for the complete schema.

Automatically acquired OPTIMADE geometry is stored as an immutable response
and POSCAR pair under the backend-specific correction directory. A complete
hash-matching pair is reused; a partial or changed cache fails rather than being
overwritten. The structure is relaxed with the active reference backend, and
its chemical-system calibration hull is built only from same-backend energies. No
Materials Project energy or hull value is imported. This process does not
fetch or rerelax doped candidate structures.

The curated Kingsbury source requires the optional installation extra::

   pip install "dopingflow[corrections]"

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

To additionally enable experimental energy correction:

- Calibration manifest and every referenced structure in ``manifest`` mode; or
  complete local/immutable OPTIMADE structures in ``phase_resolved`` mode
- Custom experimental CSV only when a custom source mode is selected
