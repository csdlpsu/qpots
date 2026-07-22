Installation
============

Requirements
------------

qPOTS requires Python 3.11 or newer and is continuously tested on Python 3.11,
3.12, and 3.13. A virtual environment is recommended so that optimization
dependencies remain isolated from other projects.

Install from PyPI
-----------------

Install the core package with:

.. code-block:: console

   python -m pip install qpots

Verify the installation:

.. code-block:: console

   python -c "from importlib.metadata import version; print(version('qpots'))"

Optional dependencies
---------------------

Plots and data analysis in the examples require Matplotlib and pandas:

.. code-block:: console

   python -m pip install "qpots[examples]"

MPI-based high-performance-computing examples additionally require ``mpi4py``:

.. code-block:: console

   python -m pip install "qpots[hpc]"

The MATLAB Engine is needed only for the optional TS-EMO baseline. Core qPOTS,
the BoTorch-based acquisition functions, and the tutorials do not require
MATLAB. Install the engine version matching the local MATLAB release by
following the `MathWorks installation guide
<https://www.mathworks.com/help/matlab/matlab_external/install-the-matlab-engine-for-python.html>`_.
For example, MATLAB R2023b uses:

.. code-block:: console

   python -m pip install matlabengine==23.2.1

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

   python -m pip install -e ".[test,docs]"
   python -m pytest -q tests/

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

Next step
---------

Continue to :doc:`unconstrained_example` for a complete optimization run.
