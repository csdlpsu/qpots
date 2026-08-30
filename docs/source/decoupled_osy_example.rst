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

   python -m examples.decoupled_osy_example

For a small end-to-end check on CPU, use:

.. code-block:: console

   python -m examples.decoupled_osy_example --quick --output-dir /tmp/qpots-osy

These commands assume a source checkout. With a PyPI-only installation, save
the complete script above as ``main.py`` and replace
``python -m examples.decoupled_osy_example`` with ``python main.py``. See
:ref:`Running the repository examples <running-the-repository-examples>`.

This example is substantially more expensive than the introductory examples
because it refits a joint multitask Gaussian process after every partially
observed batch. Reduce ``n_initial``, ``iterations``, and ``generations`` while
validating a local environment.

The built-in OSY function is fully evaluated for benchmark bookkeeping, but
only selected values enter model training. See :ref:`Benchmarking versus real
oracle calls <benchmarking-versus-real-oracle-calls>` for the distinction
between this emulation and an external workflow that avoids unselected oracle
calls entirely.

The saved output order is always two objectives followed by six constraints.
When a complete row is available, it is feasible only if every constraint is
greater than or equal to zero. ``NaN`` denotes an output that was not queried;
it is not a feasibility value.
