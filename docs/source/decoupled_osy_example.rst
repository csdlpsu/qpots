Decoupled OSY Example
=====================

This example demonstrates how to run **qPOTS** with decoupled evaluations on
the constrained **OSY** benchmark.

OSY has two objectives and six inequality constraints. In a coupled workflow,
every candidate is evaluated against all eight outputs. In a decoupled
workflow, qPOTS can select which objectives or constraints to query at each
candidate, reducing unnecessary oracle calls when outputs come from separate
simulators, experiments, or analyses.

Overview
--------

- Optimizes the **6-dimensional OSY problem** with **2 objectives** and
  **6 constraints**.
- Uses a **MultiTaskGP** so objectives and constraints are modeled jointly.
- Enables decoupled selection with ``partial_info=1``.
- Stores missing task evaluations as ``NaN`` and refits the multitask model
  with partially observed data.
- Tracks true hypervolume from fully observed bookkeeping data.
- Fills missing values with multitask posterior means for downstream analysis.

Key Settings
------------

The important decoupled-evaluation settings are:

.. code-block:: python

   "mt": 1,
   "partial_info": 1,
   "threshold": 1e-4,

``mt=1`` enables the joint multitask model. ``partial_info=1`` tells qPOTS to
return both candidate points and the selected task indices. ``threshold``
controls the total-correlation gate used to decide when to query only a subset
of outputs; use ``None`` to decouple unconditionally.

Script Details
--------------

.. literalinclude:: ../../examples/decoupled_osy_example.py
   :language: python
   :linenos:
   :caption: decoupled_osy_example.py

Example Output
--------------

.. code-block:: console

   Initial hypervolume: 0.0000
   Iteration   0 | oracles queried: 8/16 | time: 1.43s | HV: 0.0000
   Iteration   1 | oracles queried: 10/16 | time: 1.37s | HV: 12.4815
   ...
   Optimization complete.
   Final hypervolume: 52.7342
   train_y shape (with NaNs filled): torch.Size([160, 8])

Usage
-----

Run the script locally with:

.. code-block:: sh

   python examples/decoupled_osy_example.py

This example is more computationally expensive than the basic examples because
it refits a multitask Gaussian process after each partially observed batch.
