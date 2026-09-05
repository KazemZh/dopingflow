Vacancy workflow example
========================

The complete example is in ``examples/vacancies``. It assumes the normal bulk
workflow has already produced selected relaxed candidates.

.. code-block:: bash

   dopingflow vacancies -c examples/vacancies/input.toml

The example uses one flat ``[vacancies]`` table and a single MACE calculator for
screening and relaxation. Adjust ``device`` and ``model`` to match the installed
environment.

It explicitly selects ``search_method = "enumeration"`` and ``supercell =
[1, 1, 1]``. The checked-in TOML also contains a complete commented Monte Carlo
configuration. Monte Carlo supports constant-temperature sampling with
``mc_annealing = false`` and an optional high-temperature hold plus linear
cooling schedule when enabled. Both methods feed the same top-k relaxation and
reranking stages.

The checked-in file retains the backward-compatible ``reference_file`` oxygen
reference, with commented guidance for switching to the new ``global`` or
``chemistry-specific`` experimental oxygen calibration. Automatic calibration
uses only real ordinary binary oxides already calculated by ``refs-build`` and
requires same-backend bulk-metal references plus matching experimental 298 K
formation enthalpies.

See :doc:`../methods/oxygen_calibration` for the calibration equations,
eligibility rules, optional ideal vacancy configurational entropy, and the
finite-temperature NIST O2 convention.
