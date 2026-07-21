High-level optimization runner
==============================

Use :class:`qpots.runner.QPOTSRunner` for a complete optimization workflow, or
call :meth:`qpots.runner.QPOTSRunner.step` when objective evaluations need to be
coordinated with an external system. Design points accepted and returned by the
runner use the physical problem bounds; normalized points are also included in
the result objects for advanced workflows.

.. automodule:: qpots.runner
   :members:
   :undoc-members:
   :show-inheritance:
