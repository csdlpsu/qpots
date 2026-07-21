"""
This example demonstrates how to use qPOTS on a BoTorch multiobjective test function called BraninCurrin.
This is not an HPC implementation.
Install MPI support with ``pip install qpots[hpc]``.
"""
import warnings
import os
import time
import numpy as np
import sys
import ast
import torch
from mpi4py import MPI

#MPI Setup
comm = MPI.COMM_WORLD
rank = comm.Get_rank()
size = comm.Get_size()

################### Importing settings ##########################
import argparse


"""
Example call of Parallel_TC from the qPOTS folder w/ Branin-Currin and 4 Processes:
srun python -m examples.Parallel_TC --ntrain 20 --iters 20 --q 5 --func "branincurrin" --ref_point -300.0 -20.0 --dim 2 --nobj 2 --ncons 0 --use_mtgp --use_partial --thresh 1e-6 --num_test 1 --start_seed 1023

"""
def arg_parser():
    #default is Branin-Currin
    parser = argparse.ArgumentParser(description="Multi-objective Bayesian Optimization")

    # Experiment settings
    parser.add_argument("--ntrain", type=int, default=20, help="Number of initial training points.")
    parser.add_argument("--iters", type=int, default=100, help="Number of optimization iterations.")
    parser.add_argument("--q", type=int, default=4, help="Batch size for sampled points per iteration.")

    # Function and optimization settings
    parser.add_argument("--func", type=str, default="branincurrin", help="Test function for optimization. If using a custom function, do not specify.")  # HPC
    parser.add_argument("--ref_point", type=float, default=[-18.0, -6.0], nargs="+", help="Reference point for hypervolume calculation.")  # HPC
    parser.add_argument("--dim", type=int, default=2, help="Dimensionality of the input space.")
    parser.add_argument("--nobj", type=int, default=2, help="Number of objectives.")
    parser.add_argument("--ncons", type=int, default=0, help="Number of constraints.")

    # Multitask Options
    parser.add_argument("--use_mtgp", action="store_true", help="Use MultiTaskGP or not True/False.")
    parser.add_argument("--use_partial", action="store_true", help="Use partial info or not True/False.")
    parser.add_argument("--thresh", type=float, default=None, nargs="+", help="Use  thresholding or not, leave blank if not using, else a single threshold value.")

    # Multi_Seed_Wrapper Options
    parser.add_argument("--num_tests", type=int, default=1, required=True, help="Number of tests to run with different seeds.")
    parser.add_argument("--start_seed", type=int, default=1023, required=True, help="Seed that manual_seed starts with.")
    
    return parser.parse_args()

args = arg_parser()

func_sent=args.func
q_sent=args.q
dim_sent=args.dim
nobj_sent=args.nobj
ncons_sent=args.ncons
ntrain_sent=args.ntrain
iters_sent=args.iters
ref_point_sent=args.ref_point
thresh_sent=args.thresh
REPS=args.num_tests
manual_seed=args.start_seed

mtgp_sent=args.use_mtgp
if mtgp_sent:
    mtgp_sending=1
else: 
    mtgp_sending=0
partial_sent=args.use_partial

if partial_sent:
    partial_sending=1
else:
    partial_sending=0

warnings.filterwarnings('ignore')

from qpots.acquisition import Acquisition
from qpots.config import DEFAULT_DEVICE, DEFAULT_DTYPE
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


dtype = DEFAULT_DTYPE
device = DEFAULT_DEVICE
print("device:", device)
#Added mt as multi-task to args, 0 is false 1 is true
args = dict(
    {
        "ntrain": ntrain_sent,
        "iters": iters_sent,
        "reps": 20,
        "q": q_sent,
        "wd": "..",
        "ref_point": torch.tensor(ref_point_sent),
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

#problem = BraninCurrin(negate=True).to(device=device, dtype=dtype)

if args["ncons"] > 0:
    cons = tf.get_cons()

os.makedirs(args["wd"], exist_ok=True)


#MPI
for REP in range(REPS):

    if REP % size == rank:
        # set up the training points
        torch.manual_seed(manual_seed+REP)

        train_X = torch.rand([args["ntrain"], args["dim"]], dtype=DEFAULT_DTYPE)
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
            ref_point=args["ref_point"], # length K
            maximize=True,
        )

        hvs = [hv]
        times = []
        for iter in range(args["iters"]):
            t1 = time.time() # tracking time
            res, hv = get_model_identified_hv_maximizing_set(mt_model,problem=tf,ref_point=args["ref_point"])
            
            x_new = qmaximin(train_X_full, torch.tensor(res.X), q=args["q"])
            

            y_new = torch.full([args["q"], args["nobj"]], torch.nan, dtype=DEFAULT_DTYPE) #torch.zeros(q, problem.num_objectives)
            
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

            train_X_full = torch.row_stack([train_X_full, x_new])
            train_Y_full = torch.row_stack([train_Y_full, y_new])
            
            gps = ModelObject(train_X_full, train_Y_full, bounds, args["nobj"], args["ncons"], args["ntrain"], device=device)
            gps.fit_multitask_gp()
            mt_model=gps.models[0]

            hv = hypervolume_from_posterior_mean_mtgp(
                mt_model,
                X=train_X_full,                 # (n, d)
                task_feature=-1,
                ref_point=args["ref_point"], # length K
                maximize=True,
            )
            hvs.append(hv)

        ############### Saving Files  ############### 
            if args["mt"]==1:
                if args["threshold"] is None:
                    if args["partial_info"] == 1:
                        tag="rand"
                    else:
                        tag="joint"
                else:
                    tag="thresh"
                np.save(f"{args['wd']}/{REP}_{func_sent}_{tag}_train_x.npy", train_X_full)
                np.save(f"{args['wd']}/{REP}_{func_sent}_{tag}_train_y.npy", train_Y_full)
                np.save(f"{args['wd']}/{REP}_{func_sent}_{tag}_hv.npy", hvs)
                np.save(f"{args['wd']}/{REP}_{func_sent}_{tag}_times.npy", times)
                
            else:
                tag="Model_list"
                gps.fit_gp()
                np.save(f"{args['wd']}/{REP}_{func_sent}_{tag}_train_x.npy", train_X_full)
                np.save(f"{args['wd']}/{REP}_{func_sent}_{tag}_train_y.npy", train_Y_full)
                np.save(f"{args['wd']}/{REP}_{func_sent}_{tag}_hv.npy", hvs)
                np.save(f"{args['wd']}/{REP}_{func_sent}_{tag}_times.npy", times)
                
            t2 = time.time()
            times.append(t2 - t1)
            print(f"iter {iter}, Time: {t2 - t1}, HV: {hv}, tc {tc_i}, newx {x_new}, newy {y_new}\n") #iter output statement

#Wait for all to finish, then merge
comm.Barrier()  

#Merging all into master file
if rank == 0:
    train_y_all=[]
    train_x_all=[]
    hv_all=[]
    times_all=[]
    
    for REP in range(REPS):
        train_y=np.load(f"../{REP}_{func_sent}_{tag}_train_y.npy")
        train_x=np.load(f"../{REP}_{func_sent}_{tag}_train_x.npy")
        hv=np.load(f"../{REP}_{func_sent}_{tag}_hv.npy")
        times=np.load(f"../{REP}_{func_sent}_{tag}_times.npy")
        
        train_y_all.append(train_y)
        train_x_all.append(train_x)
        hv_all.append(hv)
        times_all.append(times)
    
    np.save(f"../all_{func_sent}_{tag}_train_y.npy", np.array(train_y_all, dtype=object))
    np.save(f"../all_{func_sent}_{tag}_train_x.npy", np.array(train_x_all, dtype=object))
    np.save(f"../all_{func_sent}_{tag}_hv.npy", np.array(hv_all, dtype=object))
    np.save(f"../all_{func_sent}_{tag}_times.npy", np.array(times_all, dtype=object))
