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

#importing from the wrapper file
manual_seed = int(sys.argv[1])
print("Got param:", manual_seed)

func_sent = str(sys.argv[2])
q_sent = int(sys.argv[3])
dim_sent = int(sys.argv[4])
nobj_sent = int(sys.argv[5])
ncons_sent = int(sys.argv[6])
ntrain_sent = int(sys.argv[7])
iters_sent = int(sys.argv[8])
mtgp_sent = int(sys.argv[9])
partial_sent = int(sys.argv[10])

#arrays
thresh_sent = str(sys.argv[11])
if thresh_sent == "None":
    thresh_sent=None
else:
    thresh_sent= ast.literal_eval(sys.argv[11])
    thresh_sent=torch.tensor(thresh_sent)
    print(thresh_sent)
ref_point =  ast.literal_eval(sys.argv[12])

warnings.filterwarnings('ignore')

from qpots.acquisition import Acquisition
from qpots.model_object import ModelObject
from qpots.utils.utils import expected_hypervolume
from qpots.utils.utils import posterior_mean_fill, mtgp_posterior_mean_hypervolume
from qpots.function import Function
from botorch.utils.transforms import unnormalize
from botorch.utils.multi_objective.box_decompositions import FastNondominatedPartitioning
from botorch.utils.multi_objective.hypervolume import Hypervolume
from botorch.utils.multi_objective.pareto import is_non_dominated
from qpots.utils.tc_utils import get_model_identified_hv_maximizing_set, qmaximin, computeTC, argmax_mi_subset_bruteforce
from qpots.utils.utils import hypervolume_from_posterior_mean_mtgp
from botorch.test_functions.multi_objective import BraninCurrin


dtype = torch.double
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print("device:", device)
#Added mt as multi-task to args, 0 is false 1 is true
args = dict(
    {
        "ntrain": ntrain_sent,
        "iters": iters_sent,
        "reps": 20,
        "q": q_sent,
        "wd": "..",
        "ref_point": torch.tensor(ref_point),
        "dim": dim_sent,
        "nobj": nobj_sent,
        "ncons": ncons_sent,
        "nystrom": 0,
        "nychoice": "pareto",
        "ngen": 20,
        "mt": mtgp_sent,
        "partial_info": partial_sent,
        "threshold": thresh_sent, 
    }
)

# Set up problem
tf = Function(func_sent, dim=args["dim"], nobj=args["nobj"])
f = tf.evaluate
bounds = tf.get_bounds()

problem = BraninCurrin(negate=True).to(device=device, dtype=dtype)

if args["ncons"] > 0:
    cons = tf.get_cons()

os.makedirs(args["wd"], exist_ok=True)
torch.manual_seed(manual_seed)

# set up the training points
train_X = torch.rand([args["ntrain"], args["dim"]], dtype=torch.double)
train_Y = f(unnormalize(train_X, bounds))


# fit the GP models
gps = ModelObject(train_X, train_Y, bounds, args["nobj"], args["ncons"], args["ntrain"], device=device)
gps.fit_multitask_gp()
mt_model=gps.models[0]

############# NEW
train_X_full = train_X.clone()
train_Y_full = train_Y.clone()


hv = hypervolume_from_posterior_mean_mtgp(
    mt_model,
    X=train_X_full,                 # (n, d)
    task_feature=-1,
    ref_point=[-300.0, -20.0], # length K
    maximize=True,
)

hvs = [hv]
times = []
for iter in range(args["iters"]):
    t1 = time.time() # tracking time
    res, hv = get_model_identified_hv_maximizing_set(mt_model,problem=problem)
    #print("res.X in Loop 2:\n", res.X)
    
    x_new = qmaximin(train_X_full, torch.tensor(res.X), q=args["q"])
    

    y_new = torch.full([args["q"], args["nobj"]], torch.nan, dtype=torch.double) #torch.zeros(q, problem.num_objectives)
    
    tc_i = []
    for i in range(args["q"]):
        tc = computeTC(x_new[i],mt_model=mt_model)

        tc_i.append(torch.abs(tc).item())
        if torch.abs(tc) > 1e-6:
            post = mt_model.posterior(x_new[i].view(-1,args["dim"]))
            cov  = post.distribution.covariance_matrix.detach()  # 2x2 (materialized)
            res = argmax_mi_subset_bruteforce(cov, assume_samples=False, base=2.0)
            y_new[i, res["S"]]  = f(unnormalize(x_new[i], bounds))[res["S"]]
            
            # y_new[i] = torch.tensor([test_fn(x_new[i])[0], torch.math.nan]) # simply choose first objective
        else:
            y_new[i] = f(unnormalize(x_new[i], bounds))
            

    print(f"iter {iter}\n, newx {x_new}, \n, newy {y_new}, \n tc {tc_i}")

    train_X_full = torch.row_stack([train_X_full, x_new])
    train_Y_full = torch.row_stack([train_Y_full, y_new])
    #"""
    gps = ModelObject(train_X_full, train_Y_full, bounds, args["nobj"], args["ncons"], args["ntrain"], device=device)
    gps.fit_multitask_gp()
    mt_model=gps.models[0]

    hv = hypervolume_from_posterior_mean_mtgp(
        mt_model,
        X=train_X_full,                 # (n, d)
        task_feature=-1,
        ref_point=[-300.0, -20.0], # length K
        maximize=True,
    )

    hvs.append(hv)

 ############### Saving Files and updating GP Block ############### 
    if args["mt"]==1:
        if args["threshold"] is None:
            if args["partial_info"] == 1:
                tag="rand"
            else:
                tag="joint"
        else:
            tag="thresh"
        np.save(f"{args['wd']}/{func_sent}_{tag}_train_x.npy", train_X_full)
        np.save(f"{args['wd']}/{func_sent}_{tag}_train_y.npy", train_Y_full)
        #hvs_tensor = torch.stack(hvs)
        np.save(f"{args['wd']}/{func_sent}_{tag}_hv.npy", hvs)
        #np.save(f"{args['wd']}/{func_sent}_{tag}_pareto_front.npy", np.array(pf, dtype=object)) #Will have to unpack when loading
        np.save(f"{args['wd']}/{func_sent}_{tag}_times.npy", times)
        #if args["partial_info"]==1:
        #    np.save(f"{args['wd']}/{func_sent}_{tag}_full_y.npy", full_y) #Full y is without the NaNs, using for pareto sorting later
    else:
        tag="Model_list"
        gps.fit_gp()
        np.save(f"{args['wd']}/{func_sent}_{tag}_train_x.npy", train_X_full)
        np.save(f"{args['wd']}/{func_sent}_{tag}_train_y.npy", train_Y_full)
        np.save(f"{args['wd']}/{func_sent}_{tag}_hv.npy", hvs)
        np.save(f"{args['wd']}/{func_sent}_{tag}_times.npy", times)
        #np.save(f"{args['wd']}/{func_sent}_{tag}_pareto_front.npy", pf.detach().cpu().numpy())

    t2 = time.time()
    times.append(t2 - t1)
    print(f"New candidate: {x_new}, Time: {t2 - t1}, HV: {hv}\n")

#New addition to fill with the means at the locations where train_y is NaN
#if args["partial_info"]==1:
    #train_y_filled=posterior_mean_fill(gps)
    #np.save(f"{args['wd']}/{func_sent}_{tag}_train_y_filled.npy", train_y_filled.detach().cpu().numpy())
    #np.save(f"{args['wd']}/{func_sent}_{tag}_task_id.npy", task_id.detach().cpu().numpy()) #9/15
    #np.save(f"{args['wd']}/{func_sent}_{tag}_iteration_tracker.npy", np.array(iteration_tracker,dtype=int)) #9/15
