Installation
============

Requirements
------------

qPOTS requires Python 3.11 or newer and is continuously tested on Python 3.11,
3.12, 3.13, and 3.14. A virtual environment is recommended so that optimization
dependencies remain isolated from other projects.

Install from PyPI
-----------------

Install the core package with:

.. code-block:: console

   python -m pip install qpots

Verify the installation:

.. code-block:: console

   python -c "from importlib.metadata import version; print(version('qpots'))"

The printed version should match |release| for this documentation build. If an
older version is installed, upgrade it before running the examples:

.. code-block:: console

   python -m pip install --upgrade "qpots>=2.2.0"

Optional dependencies
---------------------

Plots and data analysis in the examples require Matplotlib and pandas:

.. code-block:: console

   python -m pip install "qpots[examples]"

MPI-based high-performance-computing examples additionally require ``mpi4py``:

.. code-block:: console

   python -m pip install "qpots[hpc]"

The MATLAB Engine is needed only for the optional TS-EMO interoperability
layer. Core qPOTS, the BoTorch-based acquisition functions, and the tutorials
do not require MATLAB. qPOTS does not redistribute the TS-EMO MATLAB sources;
users must obtain an authorized checkout independently and pass its root as
``tsemo_path``. Install the engine version matching the local MATLAB release by
following the `MathWorks installation guide
<https://www.mathworks.com/help/matlab/matlab_external/install-the-matlab-engine-for-python.html>`_.
For example, MATLAB R2023b uses:

.. code-block:: console

   python -m pip install matlabengine==23.2.1

For example:

.. code-block:: python

   runner = TSEMORunner(..., tsemo_path="/path/to/external/TS-EMO")

The path must contain ``TSEMO_run.m`` and its original support directories.
See ``THIRD_PARTY_NOTICES.md`` in the source repository for the distribution
and provenance policy.

Install from source
-------------------

To work with the current repository version:

.. code-block:: console

   git clone https://github.com/csdlpsu/qpots.git
   cd qpots
   python -m pip install .

For development, install the package in editable mode with its test and
documentation tools:

.. code-block:: console

   python -m pip install -e ".[test,docs,examples]"
   python -m pytest -q tests/

.. _running-the-repository-examples:

Running the repository examples
-------------------------------

Commands beginning with ``python -m examples`` run scripts from the source
repository; the ``examples`` directory is not installed as a Python package by
``pip install qpots``. Run those commands from the root of a matching source
checkout. With a PyPI-only installation, save the complete script displayed on
an example page as ``main.py`` and run ``python main.py`` instead.

Hardware and precision
----------------------

qPOTS uses CUDA when PyTorch detects an available GPU and otherwise uses the
CPU. The default floating-point type is ``torch.float64``. Runtime settings can
be changed without editing the installed package:

.. code-block:: python

   import torch
   from qpots import RuntimeConfig, set_default_runtime

   set_default_runtime(RuntimeConfig(device="cpu", dtype=torch.float64))

An object's explicit ``device`` or ``dtype`` argument takes precedence over
the package-wide default. See :doc:`qpots_config` for the complete API.

The continuous-integration test matrix is deliberately pinned to CPU so its
behavior does not depend on whether a runner exposes CUDA. Device-propagation
tests cover the runtime configuration API, but the project does not claim a
CUDA CI job.

Next step
---------

Continue to :doc:`getting_started` for a complete optimization run.
