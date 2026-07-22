Getting started with qPOTS
==========================

This guide runs an unconstrained, two-objective optimization from problem
definition to result inspection. It uses the high-level runner, which handles
initial sampling, model fitting, candidate selection, and observation updates.

Install qPOTS
-------------

.. code-block:: console

   python -m pip install qpots

Define the problem
------------------

The Branin--Currin benchmark has two design variables and two competing
objectives:

.. code-block:: python

   from qpots import Function

   problem = Function("branincurrin", dim=2, nobj=2)
   print(problem.get_bounds())

Bounds have shape ``(2, dimension)``. The first row contains lower bounds and
the second row contains upper bounds. Points passed to ``evaluate`` and points
returned to users are in this physical domain.

Configure the optimization
--------------------------

.. code-block:: python

   from qpots import QPOTSConfig

   config = QPOTSConfig(
       n_initial=20,
       iterations=10,
       batch_size=2,
       generations=10,
       seed=1023,
   )

The most commonly changed settings are:

.. list-table::
   :header-rows: 1
   :widths: 24 20 56

   * - Setting
     - Example value
     - Meaning
   * - ``n_initial``
     - ``20``
     - Number of initial designs sampled before Bayesian optimization.
   * - ``iterations``
     - ``10``
     - Number of model-fit and candidate-selection cycles.
   * - ``batch_size``
     - ``2``
     - Number of new designs proposed per cycle.
   * - ``generations``
     - ``10``
     - Evolutionary generations used to optimize each posterior sample path.
   * - ``seed``
     - ``1023``
     - PyTorch seed controlling initial sampling and stochastic acquisition.

Larger values of ``generations`` generally spend more computation optimizing
the sampled Pareto front. Start small while checking a workflow, then increase
it for a production run.

Run the optimization
--------------------

.. code-block:: python

   from qpots import QPOTSRunner

   runner = QPOTSRunner(problem, config)
   result = runner.run()

The runner fits once at the beginning of every Bayesian optimization
iteration. It does not perform an unnecessary fit after the final observation
unless ``refit_final_model=True`` is explicitly requested.

Inspect the result
------------------

.. code-block:: python

   print(result.train_x.shape)             # (40, 2)
   print(result.train_y.shape)             # (40, 2)
   print(result.train_x_normalized.shape)  # (40, 2)

``train_x`` contains physical-domain designs. ``train_x_normalized`` contains
the corresponding internal unit-hypercube coordinates. ``train_y`` contains
the observations in the representation optimized by qPOTS. Built-in
benchmarks configured with ``negate=True`` are maximized internally, so negate
their objective columns when a conventional minimization plot is desired.

Each entry in ``result.iterations`` records one batch:

.. code-block:: python

   final_iteration = result.iterations[-1]
   print(final_iteration.candidate_x)
   print(final_iteration.evaluations.objectives)

Monitor an interactive run
--------------------------

Callbacks receive an :class:`qpots.runner.IterationResult` after each batch is
observed:

.. code-block:: python

   def report(iteration):
       print(iteration.iteration, iteration.candidate_x)

   runner = QPOTSRunner(problem, config, callbacks=[report])
   result = runner.run()

For a custom loop, call ``runner.step()`` once per desired iteration instead
of ``runner.run()``.

Next steps
----------

* Follow :doc:`constrained_tutorial` for objectives with feasibility
  constraints and a reproduction of the JOSS paper's Figure 1 workflow.
* Read :doc:`decoupled_osy_example` when objectives or constraints can be
  evaluated separately.
* See :doc:`qpots_runner` for all runner and configuration fields.
