Introduction and statement of need
==================================

What qPOTS solves
-----------------

Many scientific and engineering design problems have several competing
objectives. An aircraft design may need low drag and low structural weight; a
chemical process may need high yield and low energy use. Improving one
objective can make another worse, so the result is usually a set of trade-off
designs called a **Pareto front**, rather than one optimum.

Evolutionary multiobjective optimizers can approximate this front when
function evaluations are cheap. They become impractical when every evaluation
requires a high-fidelity simulation, laboratory experiment, or hardware test.
Bayesian optimization reduces that expense by learning surrogate models from
the observations already collected and using their uncertainty to choose the
next evaluations.

Why another multiobjective optimizer is needed
----------------------------------------------

Analytical multiobjective acquisition functions can be difficult or costly to
optimize, especially when selecting a batch of points. qPOTS uses a different
route:

1. Fit Gaussian-process models to the observed objectives and constraints.
2. Draw approximate sample paths from the model posterior.
3. Optimize those inexpensive sample paths with an evolutionary optimizer.
4. Select a diverse batch from the sampled Pareto set.
5. Evaluate the real problem at that batch and update the data.

This posterior-sampling approach avoids direct optimization of a complicated
analytical acquisition function and produces batches naturally. qPOTS builds
on PyTorch, GPyTorch, BoTorch, and pymoo instead of replacing their modeling
and optimization infrastructure.

Who should use qPOTS
--------------------

qPOTS is intended for researchers and engineers who need reproducible,
sample-efficient multiobjective optimization workflows, including:

* application researchers optimizing expensive black-box systems;
* method developers comparing multiobjective Bayesian optimization methods;
* users with feasibility constraints; and
* users whose objective and constraint measurements can be performed
  separately.

The high-level :class:`qpots.runner.QPOTSRunner` interface is the recommended
starting point. The lower-level :class:`qpots.model_object.ModelObject` and
:class:`qpots.acquisition.Acquisition` interfaces remain available for custom
research workflows.

Choose a workflow
-----------------

.. list-table::
   :header-rows: 1
   :widths: 25 35 40

   * - Evaluation setting
     - Recommended mode
     - Next page
   * - Objectives only
     - Independent Gaussian processes
     - :doc:`getting_started`
   * - Objectives and constraints evaluated together
     - Coupled constrained qPOTS
     - :doc:`constrained_tutorial`
   * - Objectives or constraints can be measured separately
     - qPOTS-Decoupled with a multitask Gaussian process
     - :doc:`qpots_doe`

Scope and references
--------------------

The original qPOTS method is described in the `AISTATS 2025 paper
<https://proceedings.mlr.press/v258/renganathan25a.html>`_. The package also
contains baseline acquisition methods and an optional TS-EMO integration for
comparative research. See :doc:`citation` for the complete citation.
