"""
This example demonstrates how to use qPOTS on a BoTorch multiobjective test function called BraninCurrin.
This is not an HPC implementation.
"""
import warnings
import os
import time
import numpy as np

warnings.filterwarnings('ignore')

from qpots.acquisition import Acquisition
from qpots.model_object import ModelObject
from qpots.utils.utils import expected_hypervolume

import torch
from qpots.function import Function
from botorch.utils.transforms import unnormalize

device = torch.device("cpu")
#Added mt as multi-task to args, 0 is false 1 is true
args = dict(
    {
        "ntrain": 300,
        "iters": 300,
        "reps": 20,
        "q": 1,
        "wd": "..",
        "ref_point": torch.tensor([-1.2, -1.2]),
        "dim": 2,
        "nobj": 2,
        "ncons": 0,
        "nystrom": 0,
        "nychoice": "pareto",
        "ngen": 10,
        "mt": 1,
    }
)

# Set up problem
tf = Function('zdt3', dim=args["dim"], nobj=args["nobj"])
f = tf.evaluate
bounds = tf.get_bounds()
print("Bounds:\n",bounds)

os.makedirs(args["wd"], exist_ok=True)
torch.manual_seed(1023)

# set up the training points
train_x = torch.rand([args["ntrain"], args["dim"]], dtype=torch.double)
train_y = f(unnormalize(train_x, bounds))

# fit the GP models
gps = ModelObject(train_x, train_y, bounds, args["nobj"], args["ncons"], device=device)
if args["mt"]==1:
    gps.fit_multitask_gp()
else:
    gps.fit_gp()

# initialize 
acq = Acquisition(tf, gps, device=device, q=args["q"])

times, hvs = [], []
for i in range(args["iters"]):
    t1 = time.time() # tracking time
    newx = acq.qpots(bounds, i, **args)
    
    newy = f(unnormalize(newx.reshape(-1, args["dim"]), bounds))
    hv, _ = expected_hypervolume(gps, ref_point=args['ref_point'])
    hvs.append(hv)

        
    train_x = torch.row_stack([train_x, newx.view(-1, args["dim"])])
    train_y = torch.row_stack([train_y, newy])
    gps = ModelObject(train_x, train_y, bounds, args["nobj"], args["ncons"], device=device)

    if args["mt"]==1:
        gps.fit_multitask_gp()
        np.save(f"{args['wd']}/train_x_zdt3_30.npy", train_x)
        np.save(f"{args['wd']}/train_y_zdt3_30.npy", train_y)
        np.save(f"{args['wd']}/hv_zdt3_30.npy", hvs)
        np.save(f"{args['wd']}/times_zdt3_30.npy", times)
    else:
        gps.fit_gp()
        np.save(f"{args['wd']}/train_x_Model_list_zdt3_30.npy", train_x)
        np.save(f"{args['wd']}/train_y_Model_list_zdt3_30.npy", train_y)
        np.save(f"{args['wd']}/hv_Model_list_zdt3_30.npy", hvs)
        np.save(f"{args['wd']}/times_Model_list_zdt3_30.npy", times)


    t2 = time.time()
    times.append(t2 - t1)
    print(f"Iteration: {i}, New candidate: {newx}, Time: {t2 - t1}, HV: {hv}")