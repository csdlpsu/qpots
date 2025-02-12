Installation
============

qPOTS can be installed from pip or by source.

To install qPOTS with pip, run the following command in a terminal::

    pip install qPOTS

This will install all of the necessary dependencies except for the MATLAB Engine, which is only needed for TS-EMO.  
To install the MATLAB Engine, follow the instructions at this link:  
`Install MATLAB Engine for Python <https://www.mathworks.com/help/matlab/matlab_external/install-the-matlab-engine-for-python.html>`_.

**Note:** The MATLAB Engine is only required if you plan on using TS-EMO and must be installed for Python>=3.10 and the corresponding MATLAB version on your machine (MATLAB installation required).  
The BoTorch implementation of the other acquisition functions (including qPOTS) only requires Python>=3.10 and the dependencies automatically installed by pip.

To build from source, clone the repository and run pip in the top-level directory::

    git clone https://github.com/csdlpsu/qpots
    cd qpots
    pip install .

Quick Note on MATLAB Engine Install
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

Installing MATLAB engine is significantly easier if using a MATLAB version starting from 2022 to the latest version because it can be installed with pip. When Installing
MATLAB engine with pip, the version has to match your installed version of MATLAB. For example, if I have MATLAB release version 2023b, I would run the following command in the terminal::

    pip install matlabengine==23.2.1