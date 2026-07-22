Tutorial: constrained Branin--Currin
====================================

This tutorial reproduces the workflow illustrated in Figure 1 of the JOSS
paper. The constrained Branin--Currin benchmark has two design variables, two
competing objectives, and a circular feasible region in the input space.

.. image:: ../../assets/qpots_constrained_illustration.png
   :alt: qPOTS observations, Pareto-front approximation, feasible input region, and selected batch for constrained Branin--Currin
   :align: center
   :width: 95%

The left panel shows objective space. The right panel shows the physical input
space and its constraint boundary. Initial observations, Bayesian optimization
observations, a dense-grid Pareto approximation, and the final proposed batch
are displayed separately.

Install plotting support
------------------------

The reproduction script uses Matplotlib in addition to qPOTS:

.. code-block:: console

   python -m pip install "qpots[examples]"

Set up a constrained problem
----------------------------

.. code-block:: python

   from qpots import Function, QPOTSConfig

   problem = Function("constrainedbc", dim=2, nobj=2)
   config = QPOTSConfig(
       n_initial=20,
       iterations=40,
       batch_size=2,
       n_constraints=1,
       generations=20,
       seed=1023,
   )

``n_constraints=1`` must agree with the constraint column returned by the
problem. qPOTS treats a constraint value greater than or equal to zero as
feasible. Built-in constrained Branin--Currin objectives are negated so the
optimizer can maximize them; the plotting code reverses that sign for the
customary minimization view.

Run and reproduce the plot
--------------------------

The complete, executable script is shown below.

.. literalinclude:: ../../examples/constrained_branin_currin_tutorial.py
   :language: python
   :linenos:
   :caption: constrained_branin_currin_tutorial.py

Run it from the repository root:

.. code-block:: console

   python examples/constrained_branin_currin_tutorial.py

It writes ``qpots_constrained_tutorial.png``. Because posterior sampling and
evolutionary optimization are stochastic, exact candidates can differ across
hardware and dependency versions even with the same seed. The script
reproduces the problem, data layers, feasibility boundary, and Pareto-front
construction used by the figure.

For a quick end-to-end check, reduce the computational settings:

.. code-block:: console

   python examples/constrained_branin_currin_tutorial.py \
       --n-initial 8 --iterations 2 --batch-size 1 --generations 2 \
       --grid-size 40 --output qpots_constrained_quick.png

Interpret the output
--------------------

The returned ``train_y`` tensor contains two objective columns followed by the
constraint column. A feasible nondominated mask can therefore be computed as:

.. code-block:: python

   from botorch.utils.multi_objective.pareto import is_non_dominated

   feasible = result.train_y[:, 2] >= 0
   nondominated = is_non_dominated(result.train_y[feasible, :2])
   pareto_points = result.train_y[feasible, :2][nondominated]

This filtering order matters: infeasible points must be removed before the
Pareto set is calculated.

Use another constrained problem
-------------------------------

The same runner pattern applies to registered problems such as ``weldedbeam``,
``discbrake``, and ``osy``. Change the function name, dimensionality, and
``n_constraints`` together. For user-defined problems, return objectives and
constraints through :class:`qpots.function.EvaluationResult`; see
:doc:`qpots_function`.
