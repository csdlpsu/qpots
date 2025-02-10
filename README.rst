qPOTS: Batch Pareto Optimal Thompson sampling
=============================================

This repository contains the code for qPOTS, a multi-objective Bayesian optimization algorithm. 
Read the paper on arXiv: `here <https://arxiv.org/pdf/2310.15788>`_.

This repository is maintained by the Computational Complex Engineered Systems Design Laboratory (CSDL_) at Penn State.

.. _CSDL: https://sites.psu.edu/csdl/

================
Installing qPOTS
================

To install qpots with pip run the following command in a terminal::

    pip install qPOTS

This will install all of the necessary dependencies except for matlab engine which is only needed for TS-EMO.
To install matlab engine follow the instructions at this link: `https://www.mathworks.com/help/matlab/matlab_external/install-the-matlab-engine-for-python.html`_.
Note that matlab engine is only required if one plans on using TS-EMO and must be installed for Python>=3.10. The BoTorch implementation of the other acquisition functions (including qPOTS) 
only requires Python>=3.10 and the dependencies automatically installed by pip

To build from source clone the repository and run pip in the top-level directory::
    
    git clone https://github.com/csdlpsu/qpots

Then run pip::

    pip install .

===============
Quick Start
===============

