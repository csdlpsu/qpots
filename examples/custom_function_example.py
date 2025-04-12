"""
This file demonstrates how to use a custom function
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
from torch import Tensor
from botorch.utils.transforms import unnormalize

device = torch.device("cpu")
args = dict(
        {
            "ntrain": 20,
            "iters": 100,
            "reps": 1,
            "q": 1,
            "wd": ".",
            "ref_point": -1*torch.tensor([5000., 300.]),
            "dim": 2,
            "nobj": 2,
            "ncons": 0,
            "nystrom": 0,
            "nychoice": "pareto",
            "ngen": 10,
        }
    )

# Negating the objective for maximization
def custom_function(X: Tensor) -> Tensor:
    f1 = X[:, 0]**2 + X[:, 1]**2 + 2*X[:, 0]*X[:, 1]**2
    f2 = (X[:, 0] - 1)**2 + (X[:, 1] - 1)**2
    return -1*torch.stack([f1, f2], dim=-1)

custom_bounds = torch.tensor([(-5., 0.), (10., 15.)]) # Normalized lower and upper bounds, two objectives. If bounds are unnormalized then it is proper to normalize them
tf = Function(name=None, dim=args["dim"], nobj=args["nobj"], custom_func=custom_function, bounds=custom_bounds.detach().tolist())
f = tf.evaluate

os.makedirs(args["wd"], exist_ok=True)
torch.manual_seed(1023)

train_x = torch.rand([args["ntrain"], args["dim"]], dtype=torch.double) # Normalized
train_y = f(unnormalize(train_x, bounds=custom_bounds))

print(min(train_y[:,0]), min(train_y[:,1]))

gps = ModelObject(train_x=train_x, train_y=train_y, bounds=custom_bounds, nobj=args["nobj"], ncons=args["ncons"], device=device)
gps.fit_gp()

acq = Acquisition(f, gps, cons=None, device=device, q=args["q"])

times, hvs = [], []
for i in range(args["iters"]):
    t1 = time.time()
    newx = acq.qpots(bounds=custom_bounds, iteration=i, **args) # Returned as normalized values
    t2 = time.time()
    times.append(t2 - t1)

    newy = f(unnormalize(newx.reshape(-1, args["dim"]), custom_bounds))
    hv, _ = expected_hypervolume(gps, ref_point=args['ref_point'])
    hvs.append(hv)

    print(f"Iteration: {i}, New candidate: {newx}, Time: {t2 - t1}, HV: {hv}")
        
    train_x = torch.row_stack([train_x, newx.view(-1, args["dim"])])
    train_y = torch.row_stack([train_y, newy])
    gps = ModelObject(train_x, train_y, custom_bounds, args["nobj"], args["ncons"], device=device)
    gps.fit_gp()

    np.save(f"{args['wd']}/train_x.npy", train_x)
    np.save(f"{args['wd']}/train_y.npy", train_y)
    np.save(f"{args['wd']}/hv.npy", hvs)
    np.save(f"{args['wd']}/times.npy", times)
    