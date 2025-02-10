"""
This file demonstrates how to use multiple acquisition functions
"""
import warnings
import time
import os
import numpy as np

warnings.filterwarnings('ignore')

from qpots.acquisition import Acquisition
from qpots.model_object import ModelObject
from qpots.utils.utils import expected_hypervolume
from qpots.function import Function

import torch
from botorch.utils.transforms import unnormalize

device = torch.device("cpu")
args = dict(
        {
            "ntrain": 20,
            "iters": 20,
            "reps": 20,
            "q": 1,
            "wd": ".",
            "ref_point": torch.tensor([-300, -18]),
            "dim": 2,
            "nobj": 2,
            "ncons": 0,
            "nystrom": 0,
            "nychoice": "pareto",
            "ngen": 10,
        }
    )

tf = Function('branincurrin', dim=args["dim"])
f = tf.evaluate
bounds = tf.get_bounds()

os.makedirs(args["wd"], exist_ok=True)
torch.manual_seed(1023)

train_x = torch.rand([args["ntrain"], args["dim"]], dtype=torch.double)
train_y = f(unnormalize(train_x, bounds))

gps = ModelObject(train_x=train_x, train_y=train_y, bounds=bounds, nobj=args["nobj"], ncons=args["ncons"], device=device)
gps.fit_gp()

acq = Acquisition(tf, gps, cons=None, device=device, q=args["q"])

times, hvs = [], []
args["wd"] = "qpots"
for i in range(args["iters"]):
    t1 = time.time()
    newx = acq.qpots(bounds=bounds, iteration=i, **args)
    t2 = time.time()
    times.append(t2 - t1)

    newy = f(unnormalize(newx.reshape(-1, args["dim"]), bounds))
    hv, _ = expected_hypervolume(gps, ref_point=args['ref_point'])
    hvs.append(hv)
        
    print(f"Iteration: {i}, New candidate: {newx}, Time: {t2 - t1}, HV: {hv}")

    train_x = torch.row_stack([train_x, newx.view(-1, args["dim"])])
    train_y = torch.row_stack([train_y, newy])
    gps = ModelObject(train_x, train_y, bounds, args["nobj"], args["ncons"], device=device)
    gps.fit_gp()

    np.save(f"{args["wd"]}/train_x.npy", train_x)
    np.save(f"{args["wd"]}/train_y.npy", train_y)
    np.save(f"{args["wd"]}/hv.npy", hvs)
    np.save(f"{args["wd"]}/times.npy", times)

"""
To use another acquisition function in the same script, the process must be repeated
starting at train_x. It is recommended to use the same seed for initialization
"""

train_x = torch.rand([args["ntrain"], args["dim"]], dtype=torch.double)
train_y = f(unnormalize(train_x, bounds))

gps_par = ModelObject(train_x=train_x, train_y=train_y, bounds=bounds, nobj=args["nobj"], ncons=args["ncons"], device=device)
gps_par.fit_gp()

acq = Acquisition(tf, gps_par, cons=None, device=device, q=args["q"])

times, hvs = [], []
args["wd"] = "parego"
for j in range(args["iters"]):
    t1 = time.time()
    newx = acq.parego() # The only thing that really changes is this line, asking for a new acquisition
    t2 = time.time()
    times.append(t2 - t1)

    newy = f(unnormalize(newx.reshape(-1, args["dim"]), bounds))
    hv, _ = expected_hypervolume(gps_par, ref_point=args["ref_point"])
    hvs.append(hv)

    print(f"Iteration: {j}, New candidate: {newx}, Time: {t2-t1}, HV: {hv}")

    train_x = torch.row_stack([train_x, newx.view(-1, args["dim"])])
    train_y = torch.row_stack([train_y, newy])
    gps_par = ModelObject(train_x, train_y, bounds, args["nobj"], args["ncons"], device=device)
    gps_par.fit_gp()

    np.save(f"{args["wd"]}/train_x.npy", train_x)
    np.save(f"{args["wd"]}/train_y.npy", train_y)
    np.save(f"{args["wd"]}/hv.npy", hvs)
    np.save(f"{args["wd"]}/times.npy", times)
