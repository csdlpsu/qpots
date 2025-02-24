qPOTS: Batch Pareto Optimal Thompson Sampling
=============================================

This repository contains the code for qPOTS, a multi-objective Bayesian optimization algorithm.  
Read the paper on arXiv: `here <https://arxiv.org/abs/2310.15788>`_.

This repository is maintained by the Computational Complex Engineered Systems Design Laboratory (`CSDL`_) at Penn State.

.. _CSDL: https://sites.psu.edu/csdl/

Read the `documentation <https://qpots-batch-pareto-optimal-thompson-sampling.readthedocs.io/en/latest/>`_.

Cite the paper:

.. code-block:: bibtex

    @article{renganathan2023qpots,
      title={qPOTS: Efficient batch multiobjective Bayesian optimization via Pareto optimal Thompson sampling},
      author={Renganathan, S Ashwin and Carlson, Kade E},
      journal={arXiv preprint arXiv:2310.15788},
      year={2023}
    }

Installing qPOTS
================

To install qPOTS with pip, run the following command in a terminal::

    pip install qpots

To build from source, clone the repository and run pip in the top-level directory::

    git clone https://github.com/csdlpsu/qpots
    cd qpots
    pip install .

This will install all of the necessary dependencies except for the MATLAB Engine, which is only needed for TS-EMO.  
To install the MATLAB Engine, follow the instructions at this link:  
`Install MATLAB Engine for Python <https://www.mathworks.com/help/matlab/matlab_external/install-the-matlab-engine-for-python.html>`_.

**Note:** The MATLAB Engine is only required if you plan on using TS-EMO and must be installed for Python>=3.10 and the corresponding MATLAB version on your machine (MATLAB installation required).  
The BoTorch implementation of the other acquisition functions (including qPOTS) only requires Python>=3.10 and the dependencies automatically installed by pip.

Quick Start
===========

A quick demonstration of qPOTS is below. This code can be run to test your qPOTS installation.

For more thorough demonstrations on how qPOTS should be used, please see the `examples/` directory.

.. code-block:: python

    import torch 
    import warnings
    import time
    from botorch.utils.transforms import unnormalize

    warnings.filterwarnings('ignore')
    device = torch.device("cpu")

    from qpots.acquisition import Acquisition 
    from qpots.model_object import ModelObject 
    from qpots.function import Function 
    from qpots.utils.utils import expected_hypervolume

    args = dict(
        {
            "ntrain": 20,
            "iters": 50,
            "reps": 20,
            "q": 1,
            "wd": ".",
            "ref_point": torch.tensor([-300.0, -18.0]),
            "dim": 2,
            "nobj": 2,
            "ncons": 0,
            "nystrom": 0,
            "nychoice": "pareto",
            "ngen": 10,
        }
    )

    tf = Function('branincurrin', dim=args["dim"], nobj=args["nobj"])
    f = tf.evaluate
    bounds = tf.get_bounds()

    torch.manual_seed(1023)

    train_x = torch.rand([args["ntrain"], args["dim"]], dtype=torch.float64)
    train_y = f(unnormalize(train_x, bounds))

    gps = ModelObject(train_x=train_x, train_y=train_y, bounds=bounds, nobj=args["nobj"], ncons=0, device=device)
    gps.fit_gp()

    acq = Acquisition(tf, gps, device=device, q=args["q"])

    for i in range(args["iters"]):
        t1 = time.time()
        newx = acq.qpots(bounds, i, **args)
        t2 = time.time()
        
        newy = f(unnormalize(newx.reshape(-1, args["dim"]), bounds))
        hv, _ = expected_hypervolume(gps, ref_point=args['ref_point'])
            
        print(f"Iteration: {i}, New candidate: {newx}, Time: {t2 - t1}, HV: {hv}")
            
        train_x = torch.row_stack([train_x, newx.view(-1, args["dim"])])
        train_y = torch.row_stack([train_y, newy])
        gps = ModelObject(train_x, train_y, bounds, args["nobj"], args["ncons"], device=device)
        gps.fit_gp()

This code prints the results to the terminal. If this works, then congratulations, you have successfully installed qPOTS!
