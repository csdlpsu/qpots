Constrained Welded Beam example
===============================

This executable example runs coupled, constrained qPOTS on the Welded Beam
benchmark. The problem has four design variables, two objectives, and four
inequality constraints.

Configuration
-------------

The script uses 40 initial designs and proposes batches of two designs for 50
iterations. ``multitask=True`` models all six outputs jointly, while
``partial_evaluations`` retains its default value of ``False``. Consequently,
both objectives and all four constraints are observed for every design.

Complete script
---------------

.. literalinclude:: ../../examples/constrained_example.py
   :language: python
   :linenos:
   :caption: constrained_example.py

The callback prints each physical-domain candidate batch and the shape of its
observed output matrix. On completion, the script saves:

``weldedbeam_train_x.pt``
   All initial and infill design points.

``weldedbeam_train_y.pt``
   Two objective columns followed by four constraint columns.

Run the example
---------------

.. code-block:: console

   python -m examples.constrained_example

For the reviewer-sized end-to-end reproduction on CPU, use:

.. code-block:: console

   python -m examples.constrained_example --quick --output-dir /tmp/qpots-weldedbeam

The output tensor has two objective columns followed by four constraint
columns. A row is feasible only when **all four** constraint values are
nonnegative. For a
guided two-dimensional constrained workflow and visualization, see
:doc:`constrained_tutorial`.
