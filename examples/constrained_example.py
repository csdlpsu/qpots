"""
This file demonstrates an optimization of a constrained problem
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
from botorch.utils.transforms import unnormalize, normalize

device = torch.device("cpu")
args = dict(
        {
            "ntrain": 40,
            "iters": 200,
            "reps": 20,
            "q": 1,
            "wd": ".",
            "ref_point": -1*torch.tensor([5.8, 4.0]),
            "dim": 4,
            "nobj": 2,
            "ncons": 4,
            "nystrom": 0,
            "nychoice": "pareto",
            "ngen": 10,
        }
    )

tf = Function('discbrake', dim=args["dim"], nobj=args["nobj"])
f = tf.evaluate
bounds = tf.get_bounds()
cons = tf.get_cons()

os.makedirs(args["wd"], exist_ok=True)
torch.manual_seed(1023)

train_x = torch.rand([args["ntrain"], args["dim"]], dtype=torch.double)
train_y = f(unnormalize(train_x, bounds))
train_y = torch.column_stack([train_y, cons(unnormalize(train_x, bounds))]) # Stack constraints on top of objectives

print(train_y.shape, train_x.shape) # This should be n_train x (nobj + ncons) tensor

gps = ModelObject(train_x=train_x, train_y=train_y, bounds=bounds, nobj=args["nobj"], ncons=args["ncons"], device=device)
gps.fit_gp()

acq = Acquisition(tf, gps, cons=cons, device=device, q=args["q"])

hvs, times = [], []
for i in range(args["iters"]):
    t1 = time.time()
    newx = acq.qpots(bounds=bounds, iteration=i, **args)
    t2 = time.time()
    times.append(t2 - t1)

    newy = f(unnormalize(newx.reshape(-1, args["dim"]), bounds))
    newconsy = cons(unnormalize(newx.reshape(-1, args["dim"]), bounds))
    newy = torch.column_stack([newy.reshape(args["q"], args["nobj"]),
                                newconsy.reshape(args["q"], args["ncons"])])
    hv, _ = expected_hypervolume(gps, ref_point=args['ref_point'])
    hvs.append(hv)
        
    print(f"Iteration: {i}, New candidate: {newx}, Time: {t2 - t1}, HV: {hv}")

    train_x = torch.row_stack([train_x, newx.view(-1, args["dim"])])
    train_y = torch.row_stack([train_y, newy])
    gps = ModelObject(train_x, train_y, bounds, args["nobj"], args["ncons"], device=device)
    gps.fit_gp()

    np.save(f"{args['wd']}/train_x.npy", train_x)
    np.save(f"{args['wd']}/train_y.npy", train_y)
    np.save(f"{args['wd']}/hv.npy", hvs)
    np.save(f"{args['wd']}/times.npy", times)

