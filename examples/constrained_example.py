"""
This file demonstrates an optimization of a constrained problem
"""

import warnings
import time
import os
import numpy as np

warnings.filterwarnings('ignore')

from qpots.acquisition import Acquisition
from qpots.config import DEFAULT_DEVICE, DEFAULT_DTYPE
from qpots.config import DEFAULT_DEVICE, DEFAULT_DTYPE
from qpots.model_object import ModelObject
from qpots.utils.utils import expected_hypervolume
from qpots.utils.utils import posterior_mean_fill
from qpots.function import Function

import torch
from botorch.utils.transforms import unnormalize, normalize

test_function_tag="weldedbeam"
if test_function_tag =="discbrake":
    dim_pass=4
    nobj_pass=2
    ntrain_pass=10*dim_pass
    ncons_pass=4
    ref_pass=[5.8, 4.0]
    var_pass=[-0.5,-0.5]
elif test_function_tag =="weldedbeam":
    dim_pass=4
    nobj_pass=2
    ntrain_pass=10*dim_pass
    ncons_pass=4
    ref_pass=[40, 10]
    var_pass=[-0.5,-0.5]

device = DEFAULT_DEVICE
args = dict(
        {
            "ntrain": ntrain_pass,
            "iters": 50,
            "reps": 20,
            "q": 2,
            "wd": "..",
            "ref_point": -1*torch.tensor(ref_pass, device=device, dtype=DEFAULT_DTYPE),
            "dim": dim_pass,
            "nobj": nobj_pass,
            "ncons": ncons_pass,
            "nystrom": 0,
            "nychoice": "pareto",
            "ngen": 10,
            "mt": 1,
            "partial_info": 0,
            "variance_threshold": None, #torch.tensor([9.790e-5,9.790e-5,9.790e-5,9.790e-5,9.790e-5,9.790e-5])
        }
    )

tf = Function(test_function_tag, dim=args["dim"], nobj=args["nobj"])
f = tf.evaluate
bounds = tf.get_bounds()
cons = tf.get_cons()

os.makedirs(args["wd"], exist_ok=True)
torch.manual_seed(1023) #1023

train_x = torch.rand([args["ntrain"], args["dim"]], device=device, dtype=DEFAULT_DTYPE)
train_y = f(unnormalize(train_x, bounds))
train_y = torch.column_stack([train_y, cons(unnormalize(train_x, bounds))]) # Stack constraints on top of objectives
full_y=train_y

print(train_y.shape, train_x.shape) # This should be n_train x (nobj + ncons) tensor

gps = ModelObject(train_x, train_y, bounds, args["nobj"], args["ncons"], args["ntrain"], device=device)
if args["mt"]==1:
    gps.fit_multitask_gp()
else:
    gps.fit_gp()

acq = Acquisition(tf, gps, cons=cons, device=device, q=args["q"])

hvs, times = [], []
for i in range(args["iters"]):

    t1 = time.time() # tracking time
    if args["partial_info"]==1:
        newx,new_task_id = acq.qpots(bounds, i, **args)
    else:
        newx = acq.qpots(bounds, i, **args)
        
    t2 = time.time()
    times.append(t2 - t1)
    

    if args["partial_info"]==1:
        #Getting full cons and y
        full_newy = f(unnormalize(newx.reshape(-1, args["dim"]), bounds))
        full_newconsy = cons(unnormalize(newx.reshape(-1, args["dim"]), bounds))

        #Attaching constraints 
        full_newy = torch.column_stack([full_newy.reshape(newx.shape[0], args["nobj"]),
                                full_newconsy.reshape(newx.shape[0], args["ncons"])])
        
        newy = torch.full_like(full_newy, float('nan'))

        for j in range(newx.shape[0]):
            cols = new_task_id[j]
            valid_mask = ~torch.isnan(cols)           
            cols = cols[valid_mask].long()             
            newy[j, cols] = full_newy[j, cols]

        #newy = torch.column_stack([newy.reshape(newx.shape[0], args["nobj"]), newconsy.reshape(newx.shape[0], args["ncons"])])
        
        
        print("Partial_info newy:\n",newy)
    else:
        newy = f(unnormalize(newx.reshape(-1, args["dim"]), bounds))
        newconsy = cons(unnormalize(newx.reshape(-1, args["dim"]), bounds))
        newy = torch.column_stack([newy.reshape(args["q"], args["nobj"]),
                                newconsy.reshape(args["q"], args["ncons"])])
    
    hv, _ = expected_hypervolume(gps, ref_point=args['ref_point'])
    hvs.append(hv)

    print(f"Iteration: {i}, New candidate: {newx}, Time: {t2 - t1}, HV: {hv}")   

    train_x = torch.row_stack([train_x, newx.view(-1, args["dim"])])
    train_y = torch.row_stack([train_y, newy])
    
    if args["partial_info"]==1:
        full_y=torch.row_stack([full_y, full_newy])

    gps = ModelObject(train_x, train_y, bounds, args["nobj"], args["ncons"], args["ntrain"], device=device)
    if args["mt"]==1:
        if args["variance_threshold"] is None:
            if args["partial_info"] == 1:
                tag="rand"
            else:
                tag="joint"
        else:
            tag="var_thresh"
        gps.fit_multitask_gp()
        np.save(f"{args['wd']}/cons_"+tag+"_train_x.npy", train_x)
        np.save(f"{args['wd']}/cons_"+tag+"_train_y.npy", train_y)
        hvs_tensor = torch.stack(hvs)
        np.save(f"{args['wd']}/cons_"+tag+"_hv.npy", hvs_tensor.detach().cpu().numpy())
        np.save(f"{args['wd']}/cons_"+tag+"_times.npy", times)
        if args["partial_info"]==1:
            np.save(f"{args['wd']}/cons_"+tag+"_full_y.npy", full_y) #Full y is without the NaNs, using for pareto sorting later
    else:
        gps.fit_gp()
        np.save(f"{args['wd']}/cons_Model_list_train_x.npy", train_x)
        np.save(f"{args['wd']}/cons_Model_list_train_y.npy", train_y)
        np.save(f"{args['wd']}/cons_Model_list_hv.npy", hvs)
        np.save(f"{args['wd']}/cons_Model_list_times.npy", times)


if args["partial_info"]==1:
    train_y_filled=posterior_mean_fill(gps)
    np.save(f"{args['wd']}/cons_"+tag+"_train_y_filled.npy", train_y_filled.detach().cpu().numpy())

