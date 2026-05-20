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

To change package-wide precision, edit ``DEFAULT_DTYPE`` in
``qpots/config.py``. For example, use ``torch.float32`` for lower memory usage
or keep ``torch.float64`` for the default double-precision behavior.
