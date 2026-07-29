Vacancy workflow example
========================

The complete example is in ``examples/vacancies``. It assumes the normal bulk
workflow has already produced selected relaxed candidates.

.. code-block:: bash

   dopingflow vacancies -c examples/vacancies/input.toml

The example uses one flat ``[vacancies]`` table and a single MACE calculator for
screening and relaxation. Adjust ``device`` and ``model`` to match the installed
environment.
