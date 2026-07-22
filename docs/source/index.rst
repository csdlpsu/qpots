qPOTS documentation
===================

.. image:: ../../assets/qpots-logo.png
   :alt: qPOTS logo
   :align: center
   :width: 180px

**qPOTS** is a Python package for sample-efficient, batch multiobjective
Bayesian optimization. It helps researchers and engineers choose which
expensive simulations, experiments, or hardware tests to run next when several
objectives compete and evaluations may be constrained.

The package combines Gaussian-process surrogate models, Thompson sampling,
and evolutionary optimization of posterior sample paths. It supports ordinary
coupled evaluations as well as **qPOTS-Decoupled**, where selected objectives
or constraints can be queried separately.

Start with :doc:`introduction` to understand the problem qPOTS addresses, then
follow :doc:`installation` and the :doc:`getting_started` guide. The
:doc:`constrained_tutorial` reproduces the workflow illustrated in Figure 1 of
the JOSS paper, and :doc:`qpots_doe` explains when and how to use decoupled
evaluations.

.. toctree::
   :maxdepth: 2
   :caption: About qPOTS

   introduction
   installation
   citation
   api_compatibility

.. toctree::
   :maxdepth: 2
   :caption: User guides

   getting_started
   constrained_tutorial
   qpots_doe

.. toctree::
   :maxdepth: 1
   :caption: Worked examples

   unconstrained_example
   constrained_example
   decoupled_osy_example

.. toctree::
   :maxdepth: 1
   :caption: API reference

   qpots_acquisition
   qpots_config
   qpots_function
   qpots_model_object
   qpots_runner
   qpots_ts_emo_wrappers
   qpots_tsemo_runner
   qpots_utils_acq_utils
   qpots_utils_pymoo_problem
   qpots_utils_tc_utils
   qpots_utils_utils
