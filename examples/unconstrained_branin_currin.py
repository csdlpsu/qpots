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
        "ntrain": 20,
        "iters": 50,
        "reps": 20,
        "q": 4,
        "wd": "..",
        "ref_point": torch.tensor([-300.0, -18.0]),
        "dim": 2,
        "nobj": 2,
        "ncons": 0,
        "nystrom": 0,
        "nychoice": "pareto",
        "ngen": 10,
        "mt": 1,
        "partial_info": 1,
    }
)

# Set up problem
tf = Function('branincurrin', dim=args["dim"], nobj=args["nobj"])
f = tf.evaluate
bounds = tf.get_bounds()

os.makedirs(args["wd"], exist_ok=True)
torch.manual_seed(1023)

# set up the training points
train_x = torch.rand([args["ntrain"], args["dim"]], dtype=torch.double)
train_y = f(unnormalize(train_x, bounds))
full_y=train_y

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
    if args["partial_info"]==1:
        newx,new_task_id = acq.qpots(bounds, i, **args)
        #print("New_Task_ID:\n",new_task_id)
        #print("newx:\n",newx)
    else:
        newx = acq.qpots(bounds, i, **args)
        
    
    t2 = time.time()
    times.append(t2 - t1)

    #EDIT HERE to get the new_y to work:
    if args["partial_info"]==1:
        full_newy = f(unnormalize(newx.reshape(-1, args["dim"]), bounds))
        #print("full function evaluation:\n",full_newy)
        newy = torch.full_like(full_newy, float('nan'))
        #print(newy)
        newy[torch.arange(args["q"]),new_task_id.squeeze(-1)]=full_newy[torch.arange(args["q"]), new_task_id.squeeze(-1)]
        #print(newy)
    hv, _ = expected_hypervolume(gps, ref_point=args['ref_point'])
    hvs.append(hv)
        
    print(f"Iteration: {i}, New candidate: {newx}, Time: {t2 - t1}, HV: {hv}\n")
        
    train_x = torch.row_stack([train_x, newx.view(-1, args["dim"])])
    train_y = torch.row_stack([train_y, newy])
    full_y=torch.row_stack([full_y, full_newy])
    #gps = ModelObject(train_x, train_y, bounds, args["nobj"], args["ncons"], device=device)
    gps.train_x=train_x
    gps.train_y=train_y

    if args["mt"]==1:
        gps.fit_multitask_gp()
        np.save(f"{args['wd']}/Partial_BC_train_x.npy", train_x)
        np.save(f"{args['wd']}/Partial_BC_train_y.npy", train_y)
        np.save(f"{args['wd']}/Partial_BC_hv.npy", hvs)
        np.save(f"{args['wd']}/Partial_BC_times.npy", times)
        np.save(f"{args['wd']}/Partial_BC_full_y.npy", full_y) #Full y is without the NaNs, using for pareto sorting later
    else:
        gps.fit_gp()
        np.save(f"{args['wd']}/train_x_Model_list.npy", train_x)
        np.save(f"{args['wd']}/train_y_Model_list.npy", train_y)
        np.save(f"{args['wd']}/hv_Model_list.npy", hvs)
        np.save(f"{args['wd']}/times_Model_list.npy", times)

