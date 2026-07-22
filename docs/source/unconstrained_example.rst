Unconstrained Branin--Currin example
====================================

This executable example runs qPOTS on the two-dimensional, two-objective
Branin--Currin benchmark using independent Gaussian processes.

Configuration
-------------

The script starts with 20 random designs, performs 100 optimization iterations,
and proposes one new design per iteration. Ten NSGA-II generations are used to
optimize each posterior sample path. A fixed seed makes the initial design and
candidate-selection sequence reproducible within a compatible environment.

Complete script
---------------

.. literalinclude:: ../../examples/unconstrained_branin.py
   :language: python
   :linenos:
   :caption: unconstrained_branin.py

The callback prints each physical-domain candidate. On completion, the script
saves:

``train_x.pt``
   All initial and infill design points.

``train_y.pt``
   The corresponding two objective values.

Run the example
---------------

.. code-block:: console

   python examples/unconstrained_branin.py

Reduce ``iterations`` and ``generations`` for a quick local check. For a
line-by-line introduction to the same workflow, see :doc:`getting_started`.
