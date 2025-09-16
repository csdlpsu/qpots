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
from qpots.utils.utils import expected_hypervolume, full_hypervolume
from qpots.utils.utils import posterior_mean_fill

import torch
from qpots.function import Function
from botorch.utils.transforms import unnormalize

device = torch.device("cpu")
#Added mt as multi-task to args, 0 is false 1 is true
args = dict(
    {
        "ntrain": 20,
        "iters": 100,
        "reps": 20,
        "q": 4,
        "wd": "..",
        "ref_point": torch.tensor([-10., -10.]),
        "dim": 2,
        "nobj": 2,
        "ncons": 0,
        "nystrom": 0,
        "nychoice": "pareto",
        "ngen": 10,
        "mt": 0,
        "partial_info": 0,
        "variance_threshold": None, #torch.tensor([3.00e-5,6.00e-5]) for Matern, .0011 for RBFtorch.tensor([.0011,.0011])
    }
)

# Set up problem
tf = Function('zdt3', dim=args["dim"], nobj=args["nobj"])
f = tf.evaluate
bounds = tf.get_bounds()
#print("Bounds:\n",bounds)

os.makedirs(args["wd"], exist_ok=True)
torch.manual_seed(1022) #1022

# set up the training points
train_x = torch.rand([args["ntrain"], args["dim"]], dtype=torch.double)
train_y = f(unnormalize(train_x, bounds))
full_y=train_y

# fit the GP models
gps = ModelObject(train_x, train_y, bounds, args["nobj"], args["ncons"], args["ntrain"], device=device)
if args["mt"]==1:
    gps.fit_multitask_gp()
else:
    gps.fit_gp()

# initialize 
acq = Acquisition(tf, gps, device=device, q=args["q"])

times, hvs, hvs_full = [], [], []
for i in range(args["iters"]):
    t1 = time.time() # tracking time
    if args["partial_info"]==1:
        newx,new_task_id = acq.qpots(bounds, i, **args)
    else:
        newx = acq.qpots(bounds, i, **args)
    
    if args["partial_info"]==1:
        full_newy = f(unnormalize(newx.reshape(-1, args["dim"]), bounds))
        
        newy = torch.full_like(full_newy, float('nan'))
    
        for j in range(newx.shape[0]):
            cols = new_task_id[j]
            valid_mask = ~torch.isnan(cols)           
            cols = cols[valid_mask].long()             
            newy[j, cols] = full_newy[j, cols]
    else:
        newy = f(unnormalize(newx.reshape(-1, args["dim"]), bounds))

    hv, pf = expected_hypervolume(gps, ref_point=args['ref_point'])
    hv_full, _ = full_hypervolume(gps, full_y, ref_point=args['ref_point'])
    hvs.append(hv)
    hvs_full.append(hv_full)

        
    train_x = torch.row_stack([train_x, newx.view(-1, args["dim"])])
    train_y = torch.row_stack([train_y, newy])
    if args["partial_info"]==1:
        full_y=torch.row_stack([full_y, full_newy])

    gps = ModelObject(train_x, train_y, bounds, args["nobj"], args["ncons"], args["ntrain"], device=device)

    if args["mt"]==1:
        gps.fit_multitask_gp()
        if args["variance_threshold"] is None:
            if args["partial_info"] == 1:
                tag="rand"
            else:
                tag="joint"
        else:
            tag="var_thresh"
        np.save(f"{args['wd']}/ZDT3_"+tag+"_train_x.npy", train_x)
        np.save(f"{args['wd']}/ZDT3_"+tag+"_train_y.npy", train_y)
        hvs_tensor = torch.stack(hvs)
        np.save(f"{args['wd']}/ZDT3_"+tag+"_hv.npy", hvs_tensor.detach().cpu().numpy())
        hvs_full_tensor = torch.stack(hvs_full)
        np.save(f"{args['wd']}/ZDT3_"+tag+"_hv_full.npy", hvs_full_tensor.detach().cpu().numpy())
        np.save(f"{args['wd']}/ZDT3_"+tag+"_times.npy", times)
        np.save(f"{args['wd']}/ZDT3_"+tag+"_pareto_front.npy", pf.detach().cpu().numpy())
        if args["partial_info"]==1:
            np.save(f"{args['wd']}/ZDT3_"+tag+"_full_y.npy", full_y) #Full y is without the NaNs, using for pareto sorting later
    else:
        gps.fit_gp()
        np.save(f"{args['wd']}/ZDT3_Model_list_train_x.npy", train_x)
        np.save(f"{args['wd']}/ZDT3_Model_list_train_y.npy", train_y)
        np.save(f"{args['wd']}/ZDT3_Model_list_hv.npy", hvs)
        np.save(f"{args['wd']}/ZDT3_Model_list_times.npy", times)
        np.save(f"{args['wd']}/ZDT3_Model_list_pareto_front.npy", pf.detach().cpu().numpy())


    t2 = time.time()
    times.append(t2 - t1)
    print(f"Iteration: {i}, New candidate: {newx}, Time: {t2 - t1}, HV: {hv} and {hv_full}")

    if args["partial_info"]==1:
        train_y_filled=posterior_mean_fill(gps)
        np.save(f"{args['wd']}/ZDT3_"+tag+"_train_y_filled.npy", train_y_filled.detach().cpu().numpy())
