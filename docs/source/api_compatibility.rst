Public API and compatibility
============================

qPOTS exposes its supported interfaces directly from the package root. Prefer
these imports in new code:

.. code-block:: python

   from qpots import (
       Acquisition,
       EvaluationResult,
       Function,
       ModelObject,
       QPOTSConfig,
       QPOTSRunner,
       RuntimeConfig,
   )

The package version is available as ``qpots.__version__``. qPOTS also ships a
``py.typed`` marker so type checkers can use the annotations included with the
installed package.

Supported surface
-----------------

The following interfaces are covered by compatibility regression tests:

* top-level names listed in ``qpots.__all__``;
* the historical module imports ``qpots.function.Function``,
  ``qpots.model_object.ModelObject``, and ``qpots.acquisition.Acquisition``;
* positional constructor arguments supported by the qPOTS 2.0 release;
* ``Function.evaluate()``, ``Function.get_bounds()``, and
  ``Function.get_cons()``;
* ``DEFAULT_DEVICE``, ``DEFAULT_DTYPE``, and the runtime helper functions in
  :mod:`qpots.config`;
* the keyword dictionary accepted by the low-level
  :meth:`qpots.acquisition.Acquisition.qpots` workflow.

Modules below ``qpots.utils`` and methods whose names start with an underscore
are implementation details. They may evolve as the algorithms and their
upstream BoTorch interfaces change.

Compatibility policy
--------------------

qPOTS uses semantic version numbers. Within the 2.x release series, supported
public call patterns will remain compatible. When an interface must be
replaced, the old form will normally remain available for at least one minor
release and emit ``DeprecationWarning`` before removal. An unavoidable removal
or behavioral incompatibility will be documented in the changelog and released
with a new major version.

Additive changes, such as new optional arguments, result fields, benchmark
entries, or top-level exports, may appear in minor releases. Patch releases are
reserved for compatible fixes and documentation or packaging corrections.

Migration from low-level qPOTS 2.0 workflows
--------------------------------------------

Existing low-level workflows do not need to be rewritten. The newer interfaces
provide optional higher-level alternatives:

.. list-table::
   :header-rows: 1
   :widths: 33 33 34

   * - Existing interface
     - New alternative
     - Compatibility behavior
   * - ``DEFAULT_DEVICE`` and ``DEFAULT_DTYPE``
     - :class:`qpots.RuntimeConfig` and
       :func:`qpots.set_default_runtime`
     - Constants remain available; runtime objects allow configuration without
       editing installed source.
   * - ``Function.evaluate()`` and ``get_cons()``
     - ``Function.evaluate_all()`` returning
       :class:`qpots.EvaluationResult`
     - Existing objective-only and separate-constraint calls are unchanged.
   * - Manual model/acquisition loop
     - :class:`qpots.QPOTSRunner`
     - Direct :class:`qpots.ModelObject` and :class:`qpots.Acquisition` use
       remains supported.
   * - ``qPOTS-DOE`` documentation name
     - ``qPOTS-Decoupled``
     - Configuration fields and the existing ``qpots_doe`` documentation URL
       remain unchanged.

All ``Function`` bounds use shape ``(2, dimension)``: lower bounds in the first
row and upper bounds in the second. This convention now has explicit validation;
invalid or ambiguous shapes raise ``ValueError`` rather than being interpreted
silently.
