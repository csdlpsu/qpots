Decoupled OSY example
=====================

This executable example applies :doc:`qpots_doe` to the constrained OSY
benchmark. OSY has six design variables, two objectives, and six inequality
constraints, giving eight output oracles in total.

Why OSY is useful here
----------------------

A coupled batch of two designs requires 16 scalar oracle evaluations. In
qPOTS-Decoupled mode, the acquisition returns a task subset for each design. Values
from tasks outside that subset are stored as ``NaN`` and excluded from the next
multitask Gaussian-process fit.

Key configuration
-----------------

.. code-block:: python

   config = QPOTSConfig(
       n_initial=60,
       iterations=50,
       batch_size=2,
       n_constraints=6,
       generations=20,
       multitask=True,
       partial_evaluations=True,
       correlation_threshold=1e-4,
       seed=1023,
   )

The multitask model shares information across all eight outputs.
``partial_evaluations`` enables output selection, while
``correlation_threshold`` activates the total-correlation gate and
mutual-information subset rule.

Complete script
---------------

.. literalinclude:: ../../examples/decoupled_osy_example.py
   :language: python
   :linenos:
   :caption: decoupled_osy_example.py

The callback reports how many scalar outputs were retained from each possible
16-output batch. The saved ``osy_partial_train_y.pt`` tensor contains objective
columns first and constraint columns second, with ``NaN`` at unobserved tasks.

Run the example
---------------

.. code-block:: console

   python examples/decoupled_osy_example.py

This example is substantially more expensive than the introductory examples
because it refits a joint multitask Gaussian process after every partially
observed batch. Reduce ``n_initial``, ``iterations``, and ``generations`` while
validating a local environment.

The built-in OSY function is fully evaluated for benchmark bookkeeping, but
only selected values enter model training. See :ref:`Benchmarking versus real
oracle calls <benchmarking-versus-real-oracle-calls>` for the distinction
between this emulation and an external workflow that avoids unselected oracle
calls entirely.
