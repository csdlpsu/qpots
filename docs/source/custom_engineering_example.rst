Custom engineering problem: cantilever beam
============================================

This worked example shows the complete path from engineering equations to a
constrained qPOTS run. The analytic equations are inexpensive so the example
can run anywhere, but the same callable boundary can invoke a finite-element
model, laboratory controller, or external simulation.

Problem definition
------------------

The rectangular beam has width ``b`` and height ``h``. The objectives are to
minimize mass and tip deflection under a bending-stress limit. These objectives
conflict: increasing the cross-section reduces deflection but increases mass.

qPOTS maximizes model outputs, so the example negates both minimization
objectives. It returns normalized stress slack as the final column:

.. math::

   g(b,h) = (\sigma_{allow} - \sigma(b,h)) / \sigma_{allow}.

The design is feasible when ``g >= 0``. This sign convention is the same for
all qPOTS constraints.

Run the workflow
----------------

Install plotting support and run the full seeded example:

.. code-block:: console

   python -m pip install "qpots[examples]"
   python -m examples.custom_function_example --output cantilever_qpots.png

A small CPU smoke run is also available:

.. code-block:: console

   python -m examples.custom_function_example --quick \
       --output /tmp/cantilever-qPOTS-quick.png

These commands assume a source checkout. With a PyPI-only installation, save
the complete script below as ``main.py`` and replace
``python -m examples.custom_function_example`` with ``python main.py``. See
:ref:`Running the repository examples <running-the-repository-examples>`.

The generated figure distinguishes infeasible and feasible designs, highlights
the feasible nondominated set, and reports feasible hypervolume as observations
accumulate. Hypervolume filtering removes infeasible rows before discarding the
constraint column.

Complete script
---------------

.. literalinclude:: ../../examples/custom_function_example.py
   :language: python
   :linenos:
   :caption: custom_function_example.py

Replace the analytic model
--------------------------

To connect an external evaluator, retain the ``EvaluationResult`` contract and
replace the body of ``evaluate_cantilever``. Return a tensor with one row per
input design, keep objectives first, and return constraint slacks separately.
If the external system uses ``c(x) <= 0``, negate that value before returning it
so qPOTS receives nonnegative feasible slack.
