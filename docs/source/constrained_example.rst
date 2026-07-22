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

   python examples/constrained_example.py

qPOTS considers a constraint feasible when its value is nonnegative. For a
guided two-dimensional constrained workflow and visualization, see
:doc:`constrained_tutorial`.
