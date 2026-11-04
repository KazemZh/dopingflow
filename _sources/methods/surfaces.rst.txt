Surface Generation and Relaxation
=================================

Overview
--------

The surface stage extends the bulk workflow by constructing slab models from
selected relaxed bulk candidates and optionally relaxing those slabs.

This stage operates **after** bulk structure generation, filtering, bandgap
prediction, and formation energy evaluation.

The workflow consists of:

1. Selecting candidate structures from a database
2. Generating slab geometries for chosen surface orientations
3. Enumerating surface terminations
4. Optionally fixing part of the slab
5. Optionally relaxing the slab using a machine-learning backend
6. Writing structures and metadata

Input Data
----------

The surface stage reads candidate data from a CSV database, typically:

::

   results_database.csv

Each row corresponds to a bulk candidate and contains:

- ``composition_tag``
- ``candidate`` (e.g. ``candidate_001``)
- ``candidate_path``
- ``E_form_norm``
- ``bandgap_eV``

The corresponding relaxed bulk structure is loaded from:

::

   candidate_path / 02_relax / POSCAR

Candidate Selection
-------------------

Selection is performed in two steps:

1. **Composition filtering**

   The dataset is first restricted using:

   - ``composition_tag``
   - or ``composition_tags``

2. **Selection mode**

   Within the selected composition subset, candidates are filtered using:

   - ID selection (``candidate_id``)
   - multiple IDs (``candidate_ids``)
   - ranking-based selection
   - numeric filters (formation energy and bandgap)

This two-step strategy ensures that comparisons are always made within the same
chemical composition.

Slab Generation
---------------

For each selected bulk structure, slabs are generated using
``pymatgen.core.surface.SlabGenerator``.

The user can choose between two modes:

Explicit orientation
^^^^^^^^^^^^^^^^^^^^

The Miller indices are directly provided:

::

   miller_list = [[1, 0, 0], [1, 1, 0], [1, 1, 1]]

Automatic orientation
^^^^^^^^^^^^^^^^^^^^^

All Miller indices up to ``max_miller`` are generated, and the first
``max_orientations`` are kept.

Slab construction
^^^^^^^^^^^^^^^^^

The slab is built according to:

- minimum slab thickness
- minimum vacuum thickness
- optional lattice reduction
- optional primitive reduction
- optional lattice reorientation

Important:

If ``in_unit_planes = false``, both slab and vacuum thickness are interpreted
in Ångström.

Orthogonalization
^^^^^^^^^^^^^^^^^

If ``orthogonal_c = true``, the slab is transformed so that:

- the ``c`` vector is perpendicular to the surface plane
- ``a`` and ``b`` span the surface

This is recommended for:

- clean POSCAR output
- stable relaxation behavior
- easier interpretation of slab thickness

Surface Terminations
--------------------

Each orientation can produce multiple terminations.

The workflow uses:

::

   slabs = SlabGenerator(...).get_slabs()

Each slab corresponds to a different termination of the same surface.

The user controls:

- whether to keep all terminations
- or only the first one
- the maximum number of terminations per orientation

No symmetry-based deduplication is applied by default, ensuring that all
physically distinct terminations are preserved.

Atom Ordering
-------------

Before writing structures, atoms are reordered according to:

::

   [generate].poscar_order

Within each species, atoms are sorted by:

1. fractional z
2. fractional y
3. fractional x

This ensures:

- clean and consistent POSCAR files
- compatibility with VASP and post-processing tools

Fixed Atoms (Selective Dynamics)
--------------------------------

The surface stage supports fixing part of the slab.

Two strategies are available:

Layer-based fixing
^^^^^^^^^^^^^^^^^^

Atoms are grouped into layers based on their Cartesian z-coordinate.

Grouping is controlled by a tolerance:

::

   fix_layer_tolerance_A

The user selects:

- bottom layers
- or middle layers

Thickness-based fixing
^^^^^^^^^^^^^^^^^^^^^^

Atoms are selected based on their z-position:

- bottom region: atoms within a thickness from the minimum z
- middle region: atoms around the slab center

Output behavior
^^^^^^^^^^^^^^^

Fixed atoms are written in the same ``POSCAR`` file using VASP selective dynamics:

- fixed atoms: ``F F F``
- free atoms: ``T T T``

Important:

These flags are not only written to file but are also enforced during relaxation
using ASE constraints.

Surface Relaxation
------------------

If enabled, slabs are relaxed using the same abstraction layer as bulk relaxation.

Workflow
^^^^^^^^

1. Convert the slab to ASE atoms
2. Attach the ML calculator
3. Apply ``FixAtoms`` constraint (if needed)
4. Run ASE optimizer
5. Convert back to pymatgen structure

Backends
^^^^^^^^

The relaxation uses the same backend system as the bulk stage:

- M3GNet
- ALIGNN (if available)
- other supported ML potentials

Runtime configuration includes:

- device (CPU or CUDA)
- GPU ID
- threading settings

Optimizer
^^^^^^^^^

Supported optimizers:

- BFGS
- LBFGS
- FIRE
- MDMin
- QuasiNewton

The optimizer runs until:

- maximum force < ``surface_fmax``
- or maximum steps reached

Constraints
^^^^^^^^^^^

Fixed atoms are enforced using:

::

   ase.constraints.FixAtoms

This ensures:

- atoms marked as fixed do not move during relaxation
- constraints are physically enforced, not only written in POSCAR

Surface Energy
--------------

The surface stage can evaluate the surface energy of each slab when a slab
energy is available (typically after surface relaxation).

Definition
^^^^^^^^^^

The surface energy is computed using the symmetric slab expression:

::

   :math:`\gamma = \frac{E_{\mathrm{slab}} - n E_{\mathrm{bulk}}}{2A}`

where:

- ``E_slab`` is the slab total energy
- ``E_bulk`` is the relaxed bulk energy of the parent candidate
- ``n`` is the number of bulk-equivalent units contained in the slab
- ``A`` is the surface area
- the factor of 2 accounts for the two exposed surfaces

Bulk reference
^^^^^^^^^^^^^^

The bulk energy is taken from the relaxed candidate structure selected
from the workflow database.

This ensures consistency between bulk and surface calculations for the
same doped composition.

Slab energy
^^^^^^^^^^^

Surface energy is computed only when a slab energy is available.

In the current implementation, this corresponds to:

- relaxed slab energy when ``relax_surface = true``

If no slab energy is available, surface energy is not computed.

Bulk equivalence factor
^^^^^^^^^^^^^^^^^^^^^^^

The factor ``n`` is determined automatically from the composition of the slab
and the parent bulk.

For each species, the ratio:

::

   N_slab / N_bulk

is computed.

If all species share the same ratio within numerical tolerance, the slab is
considered proportional to the bulk composition and this value defines ``n``.

If the ratios differ between species, the slab is not considered proportional
and surface energy is not assigned.

Stoichiometry requirement
^^^^^^^^^^^^^^^^^^^^^^^^^

Surface energy is only computed for slabs whose composition is proportional
to the bulk.

This includes:

- stoichiometric slabs
- symmetric cleavage surfaces

Surface energy is not computed for:

- non-stoichiometric terminations
- slabs missing bulk species
- slabs containing additional species

Status reporting
^^^^^^^^^^^^^^^^

Each slab is assigned a status flag:

- ``ok`` — surface energy successfully computed
- ``missing_slab_energy`` — slab energy not available
- ``missing_bulk_energy`` — bulk energy not available
- ``not_computable_not_proportional`` — composition mismatch
- ``not_computable_missing_species``
- ``not_computable_extra_species``
- ``failed`` — numerical or runtime failure

Units
^^^^^

Surface energy is reported in:

- eV/Å²
- J/m²

The conversion used is:

::

   1 eV/Å² = 16.02176634 J/m²

Outputs
-------

Each slab produces a directory:

::

   composition_tag / candidate / hkl / termination

Example:

::

   Sb50/candidate_001/hkl_1_0_0/term_001/

Files
^^^^^

``POSCAR``
   Generated slab (with optional selective dynamics)

``CONTCAR``
   Relaxed slab structure (if relaxation is enabled)

``surface_relax.log``
   Optimizer log

``surface_relax.traj``
   ASE trajectory

``surface_relax.json``
   Relaxation metadata

``meta.json``
   General slab metadata

Metadata
--------

Each slab has a ``meta.json`` file containing:

- composition and candidate information
- Miller index and termination ID
- number of atoms
- surface area
- slab and vacuum thickness estimates
- fixing parameters
- relaxation settings and results
- surface energy (when available)
- surface energy status flag
- bulk-equivalent scaling factor

This metadata is also aggregated into:

::

   surface_summary.csv

Design Considerations
---------------------

The surface stage is designed to:

- remain fully decoupled from bulk workflow stages
- allow flexible selection of candidates
- preserve all physically meaningful terminations
- support reproducible slab generation
- reuse the same backend abstraction for relaxation

This ensures consistency between:

- bulk relaxation
- surface relaxation
- downstream surface simulations
