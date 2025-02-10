"""
This example demonstrates how to use qPOTS on a BoTorch multiobjective test function called BraninCurrin.
This is not an HPC implementation.
"""
import warnings
import os
import time
import sys

warnings.filterwarnings('ignore')

from qpots.acquisition import Acquisition
from qpots.model_object import ModelObject
from qpots.utils.utils import arg_parser, expected_hypervolume

import torch
from qpots.function import Function
from botorch.utils.transforms import unnormalize, normalize

HPC = False # If this is true use the arg parser (Note: if this is true the user must provide command line arguments. See qpots/utils/command_line_utils.py)
device = torch.device("cpu")
if HPC:
    # Argument parser (command line arguments)
    args = arg_parser() # Passed in as dictionary to qpots function
    args = vars(args)
else:
    # Easier to change these values in a non-HPC environment than input a bunch of command line arguments. See docs for what they mean
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

# Set up problem
tf = Function('branincurrin', dim=2, nobj=2)
f = tf.evaluate
bounds = tf.get_bounds()

os.makedirs(args["wd"], exist_ok=True)
torch.manual_seed(1023)

# set up the training points
train_x = torch.rand([args["ntrain"], args["dim"]], dtype=torch.double)
train_y = f(unnormalize(train_x, bounds))

# fit the GP models
gps = ModelObject(train_x, train_y, bounds, args["nobj"], args["ncons"], device=device)
gps.fit_gp()

# initialize 
acq = Acquisition(tf, gps, device=device, q=args["q"])


for i in range(args["iters"]):
    t1 = time.time() # tracking time
    newx = acq.qpots(bounds, i, **args)
    t2 = time.time()
    
    newy = f(unnormalize(newx.reshape(-1, args["dim"]), bounds))
    hv, _ = expected_hypervolume(gps, ref_point=args['ref_point'])
        
    print(f"Iteration: {i}, New candidate: {newx}, Time: {t2 - t1}, HV: {hv}")
        
    train_x = torch.row_stack([train_x, newx.view(-1, args["dim"])])
    train_y = torch.row_stack([train_y, newy])
    gps = ModelObject(train_x, train_y, bounds, args["nobj"], args["ncons"], device=device)
    gps.fit_gp()




