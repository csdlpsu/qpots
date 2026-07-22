Runtime Configuration
=====================

.. automodule:: qpots.config
   :members:
   :undoc-members:
   :show-inheritance:

Precision and Device
--------------------

qPOTS centralizes floating-point precision and device selection in
``qpots.config``. The default device uses CUDA when PyTorch reports an
available GPU and falls back to CPU otherwise. The default dtype is
``torch.float64``, matching the double-precision convention commonly used in
BoTorch Gaussian-process workflows.

Change the package-wide runtime for newly created qPOTS objects with
:func:`qpots.config.set_default_runtime`; editing the installed source is not
required. For example, select CPU execution and single precision with:

.. code-block:: python

   import torch
   from qpots import set_default_runtime

   set_default_runtime(device="cpu", dtype=torch.float32)

Use :class:`qpots.config.RuntimeConfig` when an object or workflow needs its
own settings:

.. code-block:: python

   import torch
   from qpots import Function, RuntimeConfig

   runtime = RuntimeConfig(device="cpu", dtype=torch.float64)
   problem = Function("branincurrin", dim=2, nobj=2, runtime=runtime)

Explicit ``device`` or ``dtype`` arguments on an object take precedence over
its injected runtime configuration, which in turn takes precedence over the
package-wide default. The legacy ``DEFAULT_DEVICE`` and ``DEFAULT_DTYPE``
constants remain available for compatibility but should not be edited to
configure an installed package.
