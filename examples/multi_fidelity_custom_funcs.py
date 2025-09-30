"""
This example demonstrates how to use qPOTS on a BoTorch multiobjective test function called BraninCurrin.
This is not an HPC implementation.
"""
import warnings
import os
import time
import numpy as np
import sys
import ast
import torch

warnings.filterwarnings('ignore')

from qpots.acquisition import Acquisition
from qpots.model_object import ModelObject
from qpots.utils.utils import expected_hypervolume, full_hypervolume
from qpots.utils.utils import posterior_mean_fill
from qpots.function import Function
from botorch.utils.transforms import unnormalize

device = torch.device("cpu")
test_function_tag="mfcurrin"
if test_function_tag =="mfcurrin":
    dim_pass=2
    nobj_pass=2
    ntrain_pass=20
    ref_pass=[20,20]
elif test_function_tag =="mfforrester":
    dim_pass=1
    nobj_pass=3
    ntrain_pass=10
    ref_pass=[20,20,20]

#Added mt as multi-task to args, 0 is false 1 is true
args = dict(
    {
        "ntrain": ntrain_pass,
        "iters": 100,
        "reps": 20,
        "q": 4,
        "wd": "..",
        "ref_point": torch.tensor(ref_pass),
        "dim": dim_pass,
        "nobj": nobj_pass,
        "ncons": 0,
        "nystrom": 0,
        "nychoice": "pareto",
        "ngen": 10,
        "mt": 0,
        "partial_info": 0,
        "variance_threshold": 0, #
    }
)

# Set up problem
tf = Function(name=test_function_tag, dim=args["dim"], nobj=args["nobj"])
f = tf.evaluate
bounds = tf.get_bounds()

os.makedirs(args["wd"], exist_ok=True)
manual_seed=1023
torch.manual_seed(manual_seed)#OLD: 1023, NEW: 2046 (better pareto front)

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
iteration_tracker=[]
task_id=torch.arange(args["nobj"]+args["ncons"]).expand(args["ntrain"],args["nobj"]+args["ncons"]).reshape(-1,1)
for i in range(args["iters"]):
    print(f"\nIteration: {i}")
    t1 = time.time() # tracking time
    if args["partial_info"]==1:
        newx,new_task_id = acq.qpots(bounds, i, **args)

        #9/15, getting task_id and iteration ID for extension plotting
        #Be careful changing anything's shape, new_task_id_ext should be fine, just do not cahnge new_task_id shape
        #print("Chosen Task IDs in example:\n",new_task_id)
        new_mask = ~torch.isnan(new_task_id)   
        new_task_id_ext = new_task_id[new_mask]
        new_task_id_ext = new_task_id_ext.unsqueeze(1)
        
        task_id=torch.cat([task_id,new_task_id.reshape(-1,1)])
        iteration_tracker.extend([i] * new_task_id_ext.shape[0])
    else:
        newx = acq.qpots(bounds, i, **args)
        
    #New_y, Need to bring this into qPOTS at some point, and fix it such that it actually does partial evaluations
    if args["partial_info"]==1:
        full_newy = f(unnormalize(newx.reshape(-1, args["dim"]), bounds))
        
        newy = torch.full_like(full_newy, float('nan'))
    
        #New version of newy, works for more than 1 task selected per new_x
        #This will ultimately have to change and be brought into qPOTS, as it still obfuscates the problem of evaluating only some objectives
        
        #9/15 issue resolved here
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
        np.save(f"{args['wd']}/{test_function_tag}_{tag}_train_x.npy", train_x)
        np.save(f"{args['wd']}/{test_function_tag}_{tag}_train_y.npy", train_y)
        hvs_tensor = torch.stack(hvs)
        np.save(f"{args['wd']}/{test_function_tag}_{tag}_hv.npy", hvs_tensor.detach().cpu().numpy())
        hvs_full_tensor = torch.stack(hvs_full)
        np.save(f"{args['wd']}/{test_function_tag}_{tag}_hv_full.npy", hvs_full_tensor.detach().cpu().numpy())
        np.save(f"{args['wd']}/{test_function_tag}_{tag}_pareto_front.npy", pf.detach().cpu().numpy())
        np.save(f"{args['wd']}/{test_function_tag}_{tag}_times.npy", times)
        if args["partial_info"]==1:
            np.save(f"{args['wd']}/{test_function_tag}_{tag}_full_y.npy", full_y) #Full y is without the NaNs, using for pareto sorting later
    else:
        tag="Model_list"
        gps.fit_gp()
        np.save(f"{args['wd']}/{test_function_tag}_{tag}_train_x.npy", train_x)
        np.save(f"{args['wd']}/{test_function_tag}_{tag}_train_y.npy", train_y)
        np.save(f"{args['wd']}/{test_function_tag}_{tag}_hv.npy", hvs)
        np.save(f"{args['wd']}/{test_function_tag}_{tag}_times.npy", times)
        np.save(f"{args['wd']}/{test_function_tag}_{tag}_pareto_front.npy", pf.detach().cpu().numpy())

    t2 = time.time()
    times.append(t2 - t1)
    print(f"New candidate: {newx}, Time: {t2 - t1}, HV: {hv}\n")

#New addition to fill with the means at the locations where train_y is NaN
if args["partial_info"]==1:
    train_y_filled=posterior_mean_fill(gps)
    np.save(f"{args['wd']}/{test_function_tag}_{tag}_train_y_filled.npy", train_y_filled.detach().cpu().numpy())
    np.save(f"{args['wd']}/{test_function_tag}_{tag}_task_id.npy", task_id.detach().cpu().numpy()) #9/15
    np.save(f"{args['wd']}/{test_function_tag}_{tag}_iteration_tracker.npy", np.array(iteration_tracker,dtype=int)) #9/15
