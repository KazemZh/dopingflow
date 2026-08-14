.. image:: _static/logo.png
   :align: center
   :width: 600px


dopingflow
==========


ML-Driven High-Throughput Doping Workflow for Oxide Materials
--------------------------------------------------------------

The workflow integrates:

- Structure generation and symmetry-aware enumeration
- ML-based screening and relaxation using configurable backends (M3GNet, UMA, MACE, GRACE)
- Sequential gradual doping with stepwise relaxation and energy recomputation
- Formation energy calculations using configurable thermodynamic reference schemes
- Optional backend-specific formation-energy corrections fitted to experimental data
- Restricted one-dimensional alloy convex hulls and full multicomponent phase diagrams
- Bandgap prediction using ALIGNN
- Automated database collection
- Fully reproducible, stage-isolated execution

All stages are controlled through a single ``input.toml`` file.

The workflow is modular: each stage can be executed independently or combined
into a full pipeline using the ``run-all`` command.


User Guide
==========

.. toctree::
   :maxdepth: 2

   workflow_overview
   installation_and_usage
   required_inputs
   input_file


Workflow Stages
===============

The workflow is organized into modular stages.
Each stage can be executed independently and uses its own configuration block.

.. toctree::
   :maxdepth: 1

   methods/references
   methods/oxygen_thermodynamics
   methods/oxygen_calibration
   methods/generation
   methods/sequential
   methods/scanning
   methods/relaxation
   methods/filtering
   methods/bandgap
   methods/energy_corrections
   methods/formation_energy
   methods/database
   methods/alloy_hull
   methods/phase_diagram
   methods/vacancies
   methods/surfaces


Examples
========

.. toctree::
   :maxdepth: 1

   examples/explicit_single
   examples/explicit_single_oxides
   examples/multi_reference_oxide
   examples/explicit_batch
   examples/enumerate_screening
   examples/smoke_test
   examples/sequential_workflow
   examples/vacancies


API Reference
=============

.. toctree::
   :maxdepth: 1

   api/modules

===== END =====