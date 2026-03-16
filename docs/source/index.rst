.. image:: _static/logo.png
   :align: center
   :width: 600px


dopingflow
==========


ML-Driven High-Throughput Doping Workflow for Oxide Materials
--------------------------------------------------------------

The workflow integrates:

- Structure generation and symmetry-aware enumeration
- ML-based relaxation and screening (M3GNet, ALIGNN)
- Formation energy calculations using configurable thermodynamic references
- Automated database collection
- Fully reproducible, stage-isolated execution

All stages are controlled through a single ``input.toml`` file.


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
Each stage can be executed independently.

.. toctree::
   :maxdepth: 1

   methods/references
   methods/generation
   methods/scanning
   methods/relaxation
   methods/filtering
   methods/bandgap
   methods/formation_energy
   methods/database


Examples
========

.. toctree::
   :maxdepth: 1

   examples/explicit_single
   examples/explicit_single_oxides
   examples/explicit_batch
   examples/enumerate_screening
   examples/smoke_test


API Reference
=============

.. toctree::
   :maxdepth: 1

   api/modules

===== END =====