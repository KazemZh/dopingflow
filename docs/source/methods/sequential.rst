Sequential Doping Workflow
==========================

The sequential doping workflow is designed for gradual composition-by-composition
doping studies. Instead of generating each target concentration independently
from the pristine host, the workflow uses the lowest-energy relaxed structure
from the previous composition as the base structure for the next composition.

Command
-------

::

   dopingflow sequential-run -c input.toml

If ``[energy_correction].enabled = true``, the command first fits or reuses the
signature-matched correction model once, before any composition step. It does
not refit for each composition. Run ``refs-build`` first so the active
reference cache exists, and keep ``[references]`` and ``[relax]`` aligned in
backend, model, and task.

Execution Logic
---------------

For each composition generated from the ``[doping]`` section, the workflow creates
a separate step folder under ``[sequential].outdir``:

::

   sequential_structures/
      step_001_Sb2p5/
      step_002_Sb5/
      step_003_Sb7p5/
      ...

In ``mode = "full"``, each step runs:

::

   generate -> scan -> relax -> filter -> optional bandgap -> formation -> collect

The optional correction fit described above is a one-time preflight stage and
is therefore not repeated in this per-step sequence.

Formation and collection caches are forced off inside each step because those
stages publish project-level files before the workflow copies them into the
step directory. This guarantees that a later composition cannot reuse the
previous composition's database. The final merge includes only the exact steps
from the current invocation, so obsolete historical step directories are not
silently reintroduced. If the legacy global-endpoint relative-energy fallback
is enabled, it is evaluated once after this merge.

After relaxation, the lowest-energy relaxed candidate is copied to:

::

   step_xxx_<composition>/best_relaxed/POSCAR

This structure is then used as the base POSCAR for the next sequential step.

Modes
-----

full
~~~~

Runs the complete sequential workflow for each composition.

::

   [sequential]
   mode = "full"

recompute_energies
~~~~~~~~~~~~~~~~~~

Reuses existing relaxed sequential structures and reruns only formation-energy
evaluation and database collection.

Both stages are rebuilt even if their normal ``skip_if_done`` settings are
true, so changed reference energies or formation settings are applied to every
step.

This is useful when changing the thermodynamic reference, for example:

::

   oxides_ref = ["Sb2O5"]

without regenerating, scanning, or relaxing structures again.

::

   [sequential]
   mode = "recompute_energies"

Outputs
-------

Each step writes its own local database:

::

   sequential_structures/step_001_Sb2p5/results_database.csv
   sequential_structures/step_002_Sb5/results_database.csv

At the end, all step databases are merged into the project-root database:

::

   results_database.csv

Bandgap Control
---------------

The bandgap stage can be skipped in sequential calculations using:

::

   [bandgap]
   enabled = false

When disabled, the final database is still written, but the ``bandgap_eV`` column
is left empty.
