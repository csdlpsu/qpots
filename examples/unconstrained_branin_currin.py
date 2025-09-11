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
        "ref_point": torch.tensor([-18.0, -6.0]),
        "dim": 2,
        "nobj": 2,
        "ncons": 0,
        "nystrom": 0,
        "nychoice": "pareto",
        "ngen": 10,
        "mt": 1,
        "partial_info": 1,
        "variance_threshold": .0012, #.0012
    }
)

# Set up problem
tf = Function('branincurrin', dim=args["dim"], nobj=args["nobj"])
f = tf.evaluate
bounds = tf.get_bounds()

os.makedirs(args["wd"], exist_ok=True)
torch.manual_seed(2046)#OLD: 1023

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

    #New_y, Need to bring this into qPOTS at some point, and fix it such that it actually does partial evaluations
    if args["partial_info"]==1:
        full_newy = f(unnormalize(newx.reshape(-1, args["dim"]), bounds))
        
        newy = torch.full_like(full_newy, float('nan'))
    
        #New version of newy, works for more than 1 task selected per new_x
        #This will ultimately have to change and be brought into qPOTS, as it still obfuscates the problem of evaluating only some objectives
        for j in range(newx.shape[0]):
            cols = new_task_id[j]
            valid_mask = ~torch.isnan(cols)           
            cols = cols[valid_mask].long()             
            newy[j, cols] = full_newy[j, cols]
    else:
        newy = f(unnormalize(newx.reshape(-1, args["dim"]), bounds))

    hv, _ = expected_hypervolume(gps, ref_point=args['ref_point'])
    hvs.append(hv)
        
    print(f"Iteration: {i}, New candidate: {newx}, Time: {t2 - t1}, HV: {hv}\n")
        
    train_x = torch.row_stack([train_x, newx.view(-1, args["dim"])])
    train_y = torch.row_stack([train_y, newy])
    if args["partial_info"]==1:
        full_y=torch.row_stack([full_y, full_newy])
    
    #9/3 Try reinitializing class again:
    gps = ModelObject(train_x, train_y, bounds, args["nobj"], args["ncons"], args["ntrain"], device=device)
    
    #Failsafe update version
    #gps.train_x=train_x
    #gps.train_y=train_y

 ############### Saving Files and updating GP Block ############### 
    if args["mt"]==1:
        gps.fit_multitask_gp()
        if args["variance_threshold"] is None:
            if args["partial_info"] == 1:
                tag="rand"
            else:
                tag="joint"
        else:
            tag="var_thresh"
        np.save(f"{args['wd']}/BC_"+tag+"_train_x.npy", train_x)
        np.save(f"{args['wd']}/BC_"+tag+"_train_y.npy", train_y)
        hvs_tensor = torch.stack(hvs)
        np.save(f"{args['wd']}/BC_"+tag+"_hv.npy", hvs_tensor.detach().cpu().numpy())
        np.save(f"{args['wd']}/BC_"+tag+"_times.npy", times)
        if args["partial_info"]==1:
            np.save(f"{args['wd']}/BC_"+tag+"_full_y.npy", full_y) #Full y is without the NaNs, using for pareto sorting later
    else:
        gps.fit_gp()
        np.save(f"{args['wd']}/BC_Model_list_train_x.npy", train_x)
        np.save(f"{args['wd']}/BC_Model_list_train_y.npy", train_y)
        np.save(f"{args['wd']}/BC_Model_list_hv.npy", hvs)
        np.save(f"{args['wd']}/BC_Model_list_times.npy", times)

#New addition to fill with the means at the locations where train_y is NaN
if args["partial_info"]==1:
    train_y_filled=posterior_mean_fill(gps)
    np.save(f"{args['wd']}/BC_"+tag+"_train_y_filled.npy", train_y_filled.detach().cpu().numpy())
