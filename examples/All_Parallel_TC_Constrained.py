from __future__ import annotations
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
from mpi4py import MPI

#MPI Setup
comm = MPI.COMM_WORLD
rank = comm.Get_rank()
size = comm.Get_size()
print(f"[MPI CHECK] rank={rank}, size={size}",flush=True)

################### Importing settings ##########################
import argparse


"""
Example call of Parallel_TC from the qPOTS folder w/ Branin-Currin and 4 Processes:
srun python -m examples.Parallel_TC --ntrain 20 --iters 20 --q 5 --func "branincurrin" --ref_point -300.0 -20.0 --dim 2 --nobj 2 --ncons 0 --use_mtgp --use_partial --thresh 1e-6 --num_test 1 --start_seed 1023

mpirun -np 1 python -m examples.Parallel_TC --ntrain 20 --iters 10 --q 5 --func "branincurrin" --ref_point -300.0 -20.0 --dim 2 --nobj 2 --ncons 0 --acq "hvkg" --num_test 1 --start_seed 1023

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
    
    # Acquisition Function Options
    parser.add_argument("--acq", type=str, default="qpots", required=False, help="Which acquisition function to use.")
    parser.add_argument("--cost", type=float, default=[3.0,1.0] , nargs="+", required=False, help="Costs for each objective")
    
    return parser.parse_args()

# Saving Files
def file_saving_inloop(func_sent,tag,train_X_full,train_Y_full,hvs,true_hvs,times,REP,coupled_y):
    np.save(f"{args['wd']}/{REP}_{func_sent}_{tag}_train_x.npy", train_X_full)
    np.save(f"{args['wd']}/{REP}_{func_sent}_{tag}_train_y.npy", np.array(train_Y_full, dtype=object))
    np.save(f"{args['wd']}/{REP}_{func_sent}_{tag}_coupled_y.npy", np.array(coupled_y, dtype=object))
    np.save(f"{args['wd']}/{REP}_{func_sent}_{tag}_hv.npy", hvs)
    np.save(f"{args['wd']}/{REP}_{func_sent}_{tag}_true_hv.npy", true_hvs)
    np.save(f"{args['wd']}/{REP}_{func_sent}_{tag}_times.npy", times)
    

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
acquisition_function=args.acq
cost_sent=args.cost
if len(cost_sent) != nobj_sent+ncons_sent:
    print("Warning: Need same number of costs as tasks for HVKG",flush=True)
    if acquisition_function=="hvkg":
        raise IndexError(
            f"Need --cost to have {nobj_sent+ncons_sent} inputs"
        )

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
from qpots.model_object import ModelObject
from qpots.utils.utils import expected_hypervolume
from qpots.utils.utils import posterior_mean_fill, mtgp_posterior_mean_hypervolume
from qpots.function import Function
from botorch.utils.transforms import unnormalize,normalize
from botorch.utils.multi_objective.box_decompositions import FastNondominatedPartitioning
from botorch.utils.multi_objective.hypervolume import Hypervolume
from botorch.utils.multi_objective.pareto import is_non_dominated
from qpots.utils.tc_utils import get_model_identified_hv_maximizing_set, qmaximin, computeTC, argmax_mi_subset_bruteforce
from qpots.utils.utils import hypervolume_from_posterior_mean_mtgp, compute_true_hypervolume
from botorch.test_functions.multi_objective import (
    BraninCurrin, DTLZ1, DTLZ2, DTLZ3, DTLZ7, GMM, DH1, Penicillin,
    VehicleSafety, CarSideImpact, ConstrainedBraninCurrin,
    ZDT2,ZDT3, DiscBrake, MW7, OSY, WeldedBeam, C2DTLZ2,BNH,SRN,CONSTR
)
from botorch.exceptions.errors import ModelFittingError

dtype = torch.double
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
tkwargs = {
    "dtype": torch.double,
    "device": torch.device("cuda" if torch.cuda.is_available() else "cpu"),
}
print("device:", device,flush=True)
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

if func_sent=="branincurrin":
    problem = BraninCurrin(negate=True).to(device=device, dtype=dtype)
elif func_sent=="zdt2":
    problem = ZDT2(negate=True,dim=args["dim"]).to(device=device, dtype=dtype)
elif func_sent=="zdt3":
    problem = ZDT3(negate=True,dim=args["dim"]).to(device=device, dtype=dtype)
elif func_sent=="dtlz2":
    problem = DTLZ2(negate=True,dim=args["dim"],num_objectives=args["nobj"]).to(device=device, dtype=dtype)
elif func_sent=="dtlz3":
    problem = DTLZ3(negate=False,dim=args["dim"],num_objectives=args["nobj"]).to(device=device, dtype=dtype)
elif func_sent=="dtlz7":
    problem = DTLZ7(negate=True,dim=args["dim"],num_objectives=args["nobj"]).to(device=device, dtype=dtype)
elif func_sent=="penicillin": 
    problem = Penicillin(negate=True).to(device=device, dtype=dtype)
elif func_sent=="vehicle":
    problem = VehicleSafety(negate=True).to(device=device, dtype=dtype)
elif func_sent=="carside": 
    problem = CarSideImpact(negate=True).to(device=device, dtype=dtype)
elif func_sent=="constrainedbc":
    problem = ConstrainedBraninCurrin(negate=True).to(device=device, dtype=dtype)
elif func_sent=='discbrake':
    problem = DiscBrake(negate=True).to(device=device, dtype=dtype)
elif func_sent=='osy':
    problem = OSY(negate=True).to(device=device, dtype=dtype)
elif func_sent=='mw7':
    problem = MW7(negate=True,dim=args["dim"]).to(device=device, dtype=dtype)
elif func_sent=="c2dtlz2":   
    problem = C2DTLZ2(negate=True,dim=args["dim"],num_objectives=args["nobj"]).to(device=device, dtype=dtype)     
elif func_sent=="weldedbeam":
    problem = WeldedBeam(negate=True).to(device=device, dtype=dtype)     
elif func_sent=="srn":
    problem = SRN(negate=False).to(device=device, dtype=dtype) 
elif func_sent=="constr":
    problem = CONSTR(negate=True).to(device=device, dtype=dtype) 
elif func_sent=="bnh":
    problem = BNH(negate=False).to(device=device, dtype=dtype)      
else:
    raise ValueError(f"Invalid argument: --func '{func_sent}'")


if args["ncons"] > 0:
    cons = tf.get_cons()

os.makedirs(args["wd"], exist_ok=True)


#MPI
for REP in range(REPS):
 
    if REP % size == rank:
        print(f"Repetition {REP} on rank {rank}:",flush=True)
        # set up the training points
        torch.manual_seed(manual_seed+REP)

        train_X = torch.rand([args["ntrain"], args["dim"]], dtype=torch.double,device=device)
        train_Y = f(unnormalize(train_X, bounds))

        #Constraint Handling:
        if args["ncons"] > 0:
            cons = tf.get_cons()
            train_Y = torch.column_stack([train_Y, cons(unnormalize(train_X, bounds))])
            #print("training y with constraints:\n",train_Y)

        train_X_full = train_X.clone()
        train_Y_full = train_Y.clone()

        #Treu HV and full train_y even when decoupled for plotting
        True_coupled_train_y = train_Y.clone()
        true_hv=compute_true_hypervolume(True_coupled_train_y,args["ref_point"],nobj=args["nobj"],ncons=args["ncons"],maximize=True) #Taking only objectives for HV computation
        true_hvs=[true_hv]
        
        ### qPOTS ###
        if acquisition_function == "qpots":
            print(f"Using {acquisition_function}",flush=True)
            # fit the GP models
            gps = ModelObject(train_X, train_Y, bounds, args["nobj"], args["ncons"], args["ntrain"], device=device)
            gps.fit_multitask_gp()
            mt_model=gps.models[0]

            hv = hypervolume_from_posterior_mean_mtgp(
                mt_model,
                X=train_X_full,                 # (n, d)
                task_feature=-1,
                ref_point=args["ref_point"], # length K
                maximize=True,
            )
            hvs = [hv]
            times = []
            max_NSGA_iters=10
            NSGA_expansion_counter=0
            Non_invertible_counter=0
            for iter in range(args["iters"]):
                t1 = time.time() # tracking time
                
                temp_timer=time.time()
                for multiplier in range(max_NSGA_iters):
                    print("using Multiplier: ", multiplier+1,flush=True)
                    res, _ = get_model_identified_hv_maximizing_set(mt_model,problem=tf,ref_point=args["ref_point"],train_y=train_Y_full,multiplier=multiplier+1,ncons=args["ncons"])
                    print("res.X.shape[0]:", res.X.shape[0],flush=True)
                    if multiplier>1:
                        NSGA_expansion_counter+=1

                    if res.X.shape[0] >= args["q"]:
                        break
                    print(f"Inner time:{time.time()-temp_timer}")
                else:
                    raise RuntimeError(f"Could not get q candidates after max_tries on rep: {REP} on rank: {rank}")
                

                
                x_new = qmaximin(train_X_full, torch.tensor(res.X), q=args["q"])
                xnew_size=x_new.shape[0]

                
                if partial_sent:
                    print("Using Partial Evaluation",flush=True)
                    tc_i = []
                    y_new = torch.full([xnew_size, args["nobj"]+args["ncons"]], torch.nan, dtype=torch.double) #torch.zeros(q, problem.num_objectives)
                    for i in range(xnew_size):
                        tc = computeTC(x_new[i],mt_model=mt_model)

                        if tc is not None: #tc is None when R is invertible
                            if torch.abs(tc) > torch.tensor(args["threshold"]): #Perform partial eval at x when total correlation above given threshold
                                tc_i.append(torch.abs(tc).item())
                    

                                post = mt_model.posterior(x_new[i].view(-1,args["dim"]))
                                cov  = post.distribution.covariance_matrix.detach()  # 2x2 (materialized)
                                res = argmax_mi_subset_bruteforce(cov, assume_samples=False, base=2.0)
                
                                S = torch.tensor(res["S"], dtype=torch.long, device=y_new.device)
                                fx = f(unnormalize(x_new[i], bounds)).unsqueeze(0) #Function objectives at x_i
                                #print(fx.shape)
                                if args["ncons"] > 0:
                                    fc=cons(unnormalize(x_new[i], bounds).unsqueeze(0)) #Constraints at x_i
                                    #print(fx.shape)
                                    fx=torch.column_stack((fx,fc)).squeeze()#Combining into objectives+cons

                                #print("S",S)
                                #print("y_new",y_new)
                                #print("fx",fx)
                                y_new[i, S] = fx[S]
                                
                            else: #when total correlation is below threshold, perform join evaluation at x
                                
                                tc_i.append(torch.abs(tc).item())
                                fx=f(unnormalize(x_new[i], bounds)).unsqueeze(0)
                                if args["ncons"] > 0:
                                    fc=cons(unnormalize(x_new[i], bounds).unsqueeze(0)) #Constraints at x_i
                                
                                    fx=torch.column_stack((fx,fc))#Combining into objectives+cons
                                y_new[i] = fx
                        else: #When R isnt invertible, perform joint evaluation at x
                            fx=f(unnormalize(x_new[i], bounds)).unsqueeze(0)
                            if args["ncons"] > 0:
                                fc=cons(unnormalize(x_new[i], bounds).unsqueeze(0)) #Constraints at x_i
                                #print(fx.shape)
                                fx=torch.column_stack((fx,fc))#Combining into objectives+cons
                            y_new[i] = fx
                            Non_invertible_counter+=1
                else:
                    print("Using Joint Evaluation",flush=True)
                    fx=f(unnormalize(x_new, bounds))
                    if args["ncons"] > 0:
                        fc=cons(unnormalize(x_new, bounds)) #Constraints at x_i
                
                        fx=torch.column_stack((fx,fc))#Combining into objectives+cons
                        
                    y_new = fx

                train_X_full = torch.row_stack([train_X_full, x_new])
                train_Y_full = torch.row_stack([train_Y_full, y_new])

                #Update true HV using true coupled train_y

                fx=f(unnormalize(x_new, bounds))
                if args["ncons"] > 0:
                    fc=cons(unnormalize(x_new, bounds)) #Constraints at x_i
                    #print(fx.shape)
                    fx=torch.column_stack((fx,fc))#Combining into objectives+cons
                
                coupled_new_y=fx
                True_coupled_train_y = torch.row_stack([True_coupled_train_y, coupled_new_y])
                true_hv=compute_true_hypervolume(True_coupled_train_y,args["ref_point"],nobj=args["nobj"],ncons=args["ncons"],maximize=True) #taking only the objective points for HV computation
                true_hvs.append(true_hv)
                
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

                t2 = time.time()
                times.append(t2 - t1)
                if partial_sent:
                    print(f"iter {iter}, Time: {t2 - t1}, HV: {true_hv}, tc {tc_i}, newx {x_new}, newy {y_new}\n",flush=True) #iter output statement
                else:
                    print(f"iter {iter}, Time: {t2 - t1}, HV: {true_hv}, newx {x_new}, newy {y_new}\n",flush=True) #iter output statement
                if args["mt"]==1:
                    if args["threshold"] is None:
                        if args["partial_info"] == 1:
                            tag="rand"
                        else:
                            tag="joint"
                    else:
                        tag="thresh"
                else:
                    tag="Model_list"
                print(f"saving file on Rep: {REP}, rank: {rank}")
                file_saving_inloop(func_sent=func_sent,tag=tag,train_X_full=train_X_full,train_Y_full=train_Y_full,hvs=hvs,true_hvs=true_hvs,times=times,REP=REP,coupled_y=True_coupled_train_y)
                np.save(f"{args['wd']}/{REP}_{func_sent}_{tag}_exception_handling_NSGA.npy", NSGA_expansion_counter)
                np.save(f"{args['wd']}/{REP}_{func_sent}_{tag}_Non_invertible_counter.npy", Non_invertible_counter)

        ### HVKG ###
        elif acquisition_function == "hvkg":
            from botorch import fit_gpytorch_mll
            from qpots.utils.acq_utils import initialize_model,hypervolume_from_posterior_mean_gp,optimize_HVKG_and_get_obs_decoupled
            from botorch.models.cost import FixedCostModel

            # define the cost model
            print("HVKG:\n",flush=True)
            tag=acquisition_function
            objective_costs = {i: float(cost) for i, cost in enumerate(cost_sent)}
            objective_indices = list(objective_costs.keys())
            objective_costs = {int(k): v for k, v in objective_costs.items()}
            objective_costs_t = torch.tensor(
                [objective_costs[k] for k in sorted(objective_costs.keys())], **tkwargs
            )
            cost_model = FixedCostModel(fixed_cost=objective_costs_t)
            total_cost = {"hvkg": 0.0}

            train_obj_hvkg_list = list(train_Y.split(1, dim=-1))
            train_x_hvkg_list = [train_X] * len(train_obj_hvkg_list)
            mll_hvkg, model_hvkg = initialize_model(train_x_hvkg_list, train_obj_hvkg_list,bounds)
            
            standard_bounds = torch.zeros(2, problem.dim, **tkwargs)
            standard_bounds[1] = 1
            
            cost_hvkg = cost_model(train_X).sum(dim=-1)
            total_cost["hvkg"] += cost_hvkg.sum().item()
            
            # fit the models
            try:
                fit_gpytorch_mll(mll_hvkg)
            except ModelFittingError:
                print("GP fitting failed, restoring previous hyperparameters.")
                if 'prev_hypers' in globals():
                    model_hvkg.load_state_dict(prev_hypers)
                else:
                    print("No previous hyperparameters available. Using current defaults.")
            prev_hypers = model_hvkg.state_dict()
            
            # compute hypervolume
            #hv = get_model_identified_hv_maximizing_set(model=model_hvkg)
            hv = hypervolume_from_posterior_mean_gp(model=model_hvkg,X=train_X_full,ncons=args["ncons"],ref_point=args["ref_point"],maximize=True,)
            print("Initial Hypervolume is: ",true_hv,flush=True)
            hvs_hvkg= [hv]
            times = []
            for iter in range(args["iters"]):
                t1 = time.time()

                # generate candidates
                print("\nHVKG:\n",flush=True)
                (
                    new_x_hvkg,
                    new_obj_hvkg,
                    eval_objective_indices_hvkg,
                ) = optimize_HVKG_and_get_obs_decoupled(
                    model_hvkg,args["q"],problem,cost_model,standard_bounds,objective_indices,ncons=args["ncons"],nobj=args["nobj"],train_x=train_X_full
                )

                #Constraints handled in optimize, new_obj_hvkg has constraints in it
                
                # update training points
                new_x_hvkg_norm=normalize(new_x_hvkg,bounds) #normalizing the new_x to keep train_x normalized 2/23
                for i in eval_objective_indices_hvkg:
                    train_x_hvkg_list[i] = torch.cat([train_x_hvkg_list[i], new_x_hvkg_norm])
                    train_obj_hvkg_list[i] = torch.cat(
                        [train_obj_hvkg_list[i], new_obj_hvkg], dim=0
                    )
                train_X_full=torch.cat([train_X_full,new_x_hvkg_norm])
                #print("train_y_post",train_obj_hvkg_list)

                # update costs
                all_outcome_cost = cost_model(new_x_hvkg)
                new_cost_hvkg = all_outcome_cost[..., eval_objective_indices_hvkg].sum(dim=-1)
                cost_hvkg = torch.cat([cost_hvkg, new_cost_hvkg], dim=0)
                total_cost["hvkg"] += new_cost_hvkg.sum().item()
                # fit models
                mll_hvkg, model_hvkg = initialize_model(train_x_hvkg_list, train_obj_hvkg_list, bounds)
                
                try:
                    fit_gpytorch_mll(mll_hvkg)
                except ModelFittingError:
                    print("GP fitting failed, restoring previous hyperparameters.")
                    if 'prev_hypers' in globals():
                        model_hvkg.load_state_dict(prev_hypers)
                    else:
                        print("No previous hyperparameters available. Using current defaults.")
                prev_hypers = model_hvkg.state_dict()
                
                hv = hypervolume_from_posterior_mean_gp(model=model_hvkg,X=train_X_full,ncons=args["ncons"],ref_point=args["ref_point"],maximize=True,)
                hvs_hvkg.append(hv)

                #Update true HV using true coupled train_y
                fx=f(new_x_hvkg)
                if args["ncons"] > 0:
                    fc=cons(new_x_hvkg) #Constraints at x_i
                    #print(fx.shape)
                    fx=torch.column_stack((fx,fc))#Combining into objectives+cons
                
                coupled_new_y=fx
                True_coupled_train_y = torch.row_stack([True_coupled_train_y, coupled_new_y])
                true_hv=compute_true_hypervolume(True_coupled_train_y,args["ref_point"],nobj=args["nobj"],ncons=args["ncons"],maximize=True)
                true_hvs.append(true_hv)

                t2 = time.time()
                times.append(t2 - t1)
                print(f"iter {iter}, Time: {t2 - t1}, HV: {true_hv}, newx {new_x_hvkg}, newy {new_obj_hvkg}\n",flush=True) #iter output statement
                file_saving_inloop(func_sent=func_sent,tag=tag,train_X_full=train_X_full,train_Y_full=train_obj_hvkg_list,hvs=hvs_hvkg,true_hvs=true_hvs,times=times,REP=REP,coupled_y=True_coupled_train_y)

        #qNEHVI        
        elif acquisition_function == "qnehvi":
            from botorch import fit_gpytorch_mll
            from qpots.utils.acq_utils import initialize_model,hypervolume_from_posterior_mean_gp,optimize_qnehvi_and_get_observation
            from botorch.sampling.normal import SobolQMCNormalSampler

            print("qNEHVI",flush=True)
            tag=acquisition_function
            objective_costs = {i: float(cost) for i, cost in enumerate(cost_sent)}
            objective_indices = list(objective_costs.keys())

            train_obj_qnehvi_list = list(train_Y.split(1, dim=-1))
            train_x_qnehvi_list = [train_X] * len(train_obj_qnehvi_list)

            #print("qnehvi printing",train_obj_qnehvi_list)
            mll_qnehvi, model_qnehvi = initialize_model(train_x_qnehvi_list, train_obj_qnehvi_list,bounds)
            standard_bounds = torch.zeros(2, problem.dim, **tkwargs)
            standard_bounds[1] = 1
            
            # fit the models
            try:
                fit_gpytorch_mll(mll_qnehvi)
            except ModelFittingError:
                print("GP fitting failed, restoring previous hyperparameters.")
                if 'prev_hypers' in globals():
                    model_qnehvi.load_state_dict(prev_hypers)
                else:
                    print("No previous hyperparameters available. Using current defaults.")
            prev_hypers = model_qnehvi.state_dict()
            
            # compute hypervolume
            hv = hypervolume_from_posterior_mean_gp(model=model_qnehvi,X=train_X_full,ncons=args["ncons"],ref_point=args["ref_point"],maximize=True,)
            hvs_qnehvi= [hv]
            print("Initial Hypervolume is: ",true_hv,flush=True)
            times = []
            for iter in range(args["iters"]):
                t1 = time.time()
                qnehvi_sampler = SobolQMCNormalSampler(sample_shape=torch.Size([128]))
                # generate candidates
                new_x_qnehvi, new_obj_qnehvi = optimize_qnehvi_and_get_observation(
                    model_qnehvi, train_x_qnehvi_list[0], qnehvi_sampler, q=args["q"], problem=problem,standard_bounds=standard_bounds,ncons=args["ncons"],nobj=args["nobj"]
                )
                #new_obj_qnehvi does not have constraints added to it yet

                #Constraint Handling
                if args["ncons"] > 0:
                    new_cons=cons(new_x_qnehvi)
                    new_obj_qnehvi=torch.column_stack((new_obj_qnehvi,new_cons))#Combining into objectives+cons

                
                #print("cons new_obj_qnehvi",new_obj_qnehvi)
                
                # update training points
                new_x_qnehvi_norm=normalize(new_x_qnehvi,bounds) #normalizing the new_x to keep train_x normalized 2/23
                for i in range(args["nobj"]+args["ncons"]):
                    train_x_qnehvi_list[i] = torch.cat([train_x_qnehvi_list[i], new_x_qnehvi_norm])
                    train_obj_qnehvi_list[i] = torch.cat(
                        [train_obj_qnehvi_list[i], new_obj_qnehvi[..., i : i + 1]]
                    )
                train_X_full=torch.cat([train_X_full,new_x_qnehvi_norm])
                
                #print("Updated training objs",train_obj_qnehvi_list)
                mll_qnehvi, model_qnehvi = initialize_model(
                    train_x_qnehvi_list, train_obj_qnehvi_list,bounds
                )
                try:
                    fit_gpytorch_mll(mll_qnehvi)
                except ModelFittingError:
                    print("GP fitting failed, restoring previous hyperparameters.")
                    if 'prev_hypers' in globals():
                        model_qnehvi.load_state_dict(prev_hypers)
                    else:
                        print("No previous hyperparameters available. Using current defaults.")
                prev_hypers = model_qnehvi.state_dict()
                
                hv = hypervolume_from_posterior_mean_gp(model=model_qnehvi,X=train_X_full,ncons=args["ncons"],ref_point=args["ref_point"],maximize=True,)

                hvs_qnehvi.append(hv)

                #Update true HV using true coupled train_y
                fx=f(new_x_qnehvi)
                if args["ncons"] > 0:
                    fc=cons(new_x_qnehvi) #Constraints at x_i
                    #print(fx.shape)
                    fx=torch.column_stack((fx,fc))#Combining into objectives+cons
                
                coupled_new_y=fx
                True_coupled_train_y = torch.row_stack([True_coupled_train_y, coupled_new_y])
                true_hv=compute_true_hypervolume(True_coupled_train_y,args["ref_point"],nobj=args["nobj"],ncons=args["ncons"],maximize=True)
                true_hvs.append(true_hv)
                
                t2 = time.time()
                times.append(t2 - t1)
                print(f"iter {iter}, Time: {t2 - t1}, HV: {true_hv}, newx {new_x_qnehvi}, newy {new_obj_qnehvi}\n",flush=True) #iter output statement
                file_saving_inloop(func_sent=func_sent,tag=tag,train_X_full=train_X_full,train_Y_full=train_obj_qnehvi_list,hvs=hvs_qnehvi,true_hvs=true_hvs,times=times,REP=REP,coupled_y=True_coupled_train_y)
        elif acquisition_function == "sobol":
            from botorch import fit_gpytorch_mll
            from qpots.utils.acq_utils import initialize_model,hypervolume_from_posterior_mean_gp,generate_sobol_data

            print("Sobol",flush=True)
            tag=acquisition_function
            objective_costs = {i: float(cost) for i, cost in enumerate(cost_sent)}
            objective_indices = list(objective_costs.keys())

            train_obj_random_list = list(train_Y.split(1, dim=-1))
            train_x_random_list = [train_X] * len(train_obj_random_list)
            mll_random, model_random = initialize_model(train_x_random_list, train_obj_random_list,bounds)
            
            standard_bounds = torch.zeros(2, problem.dim, **tkwargs)
            standard_bounds[1] = 1
            
            # fit the models
            try:
                fit_gpytorch_mll(mll_random)
            except ModelFittingError:
                print("GP fitting failed, restoring previous hyperparameters.")
                if 'prev_hypers' in globals():
                    model_random.load_state_dict(prev_hypers)
                else:
                    print("No previous hyperparameters available. Using current defaults.")
            prev_hypers = model_random.state_dict()
            
            # compute hypervolume
            hv = hypervolume_from_posterior_mean_gp(model=model_random,X=train_X_full,ncons=args["ncons"],ref_point=args["ref_point"],maximize=True,)
            hvs_random= [hv]

            print("Initial Hypervolume is: ",true_hv,flush=True)
            times = []
            for iter in range(args["iters"]):
                t1 = time.time()
                new_x_random, new_obj_random = generate_sobol_data(n=args["q"],problem=problem)
                if args["ncons"] > 0:
                    new_cons=cons(new_x_random)
                    new_obj_random=torch.column_stack((new_obj_random,new_cons))#Combining into objectives+cons
                
                # update training points
                new_x_random_norm=normalize(new_x_random,bounds) #normalizing the new_x to keep train_x normalized 2/23
                for i in objective_indices:
                    train_x_random_list[i] = torch.cat([train_x_random_list[i], new_x_random_norm])
                    train_obj_random_list[i] = torch.cat(
                        [train_obj_random_list[i], new_obj_random[..., i : i + 1]]
                    )
                train_X_full=torch.cat([train_X_full,new_x_random_norm])
                
                mll_random, model_random = initialize_model(
                    train_x_random_list, train_obj_random_list, bounds
                )
                ## Model Fitting
                try:
                    fit_gpytorch_mll(mll_random)
                except ModelFittingError:
                    print("GP fitting failed, restoring previous hyperparameters.")
                    if 'prev_hypers' in globals():
                        model_random.load_state_dict(prev_hypers)
                    else:
                        print("No previous hyperparameters available. Using current defaults.")
                prev_hypers = model_random.state_dict()
                

                #hv = get_model_identified_hv_maximizing_set(model=model_random)
                hv = hypervolume_from_posterior_mean_gp(model=model_random,X=train_X_full,ncons=args["ncons"],ref_point=args["ref_point"],maximize=True,)

                hvs_random.append(hv)

                #Update true HV using true coupled train_y
                fx=f(new_x_random)
                if args["ncons"] > 0:
                    fc=cons(new_x_random) #Constraints at x_i
                    #print(fx.shape)
                    fx=torch.column_stack((fx,fc))#Combining into objectives+cons
                
                coupled_new_y=fx
                True_coupled_train_y = torch.row_stack([True_coupled_train_y, coupled_new_y])
                true_hv=compute_true_hypervolume(True_coupled_train_y,args["ref_point"],nobj=args["nobj"],ncons=args["ncons"],maximize=True)
                true_hvs.append(true_hv)

                t2 = time.time()
                times.append(t2 - t1)
                print(f"iter {iter}, Time: {t2 - t1}, HV: {true_hv}, newx {new_x_random}, newy {new_obj_random}\n",flush=True) #iter output statement
                file_saving_inloop(func_sent=func_sent,tag=tag,train_X_full=train_X_full,train_Y_full=train_obj_random_list,hvs=hvs_random,true_hvs=true_hvs,times=times,REP=REP,coupled_y=True_coupled_train_y)
        #Sobol Decoupled
        elif acquisition_function == "sobol_dec":
            from botorch import fit_gpytorch_mll
            from qpots.utils.acq_utils import initialize_model,hypervolume_from_posterior_mean_gp,generate_sobol_data

            print("Sobol Decoupled",flush=True)
            tag=acquisition_function
            objective_costs = {i: float(cost) for i, cost in enumerate(cost_sent)}
            objective_indices = list(objective_costs.keys())

            train_obj_random_list = list(train_Y.split(1, dim=-1))
            train_x_random_list = [train_X] * len(train_obj_random_list)
            mll_random, model_random = initialize_model(train_x_random_list, train_obj_random_list,bounds)
            
            standard_bounds = torch.zeros(2, problem.dim, **tkwargs)
            standard_bounds[1] = 1
            
            # fit the models
            try:
                fit_gpytorch_mll(mll_random)
            except ModelFittingError:
                print("GP fitting failed, restoring previous hyperparameters.")
                if 'prev_hypers' in globals():
                    model_random.load_state_dict(prev_hypers)
                else:
                    print("No previous hyperparameters available. Using current defaults.")
            prev_hypers = model_random.state_dict()
            
            # compute hypervolume
            hv = hypervolume_from_posterior_mean_gp(model=model_random,X=train_X_full,ncons=args["ncons"],ref_point=args["ref_point"],maximize=True,)
            hvs_random= [hv]

            print("Initial Hypervolume is: ",true_hv,flush=True)
            times = []
            for iter in range(args["iters"]):
                t1 = time.time()
                new_x_random, new_obj_random = generate_sobol_data(n=args["q"],problem=problem)
                if args["ncons"] > 0:
                    new_cons=cons(new_x_random)
                    new_obj_random=torch.column_stack((new_obj_random,new_cons))#Combining into objectives+cons

                rand_tasks = torch.randint(
                    low=0,
                    high=args["nobj"]+args["ncons"],
                    size=(args["q"],)
                )
                # update training points
                k=0
                print("rand_tasks",rand_tasks,flush=True)
                
                new_x_random_norm=normalize(new_x_random,bounds) #normalizing the new_x to keep train_x normalized 2/23
                for i in rand_tasks:
                    train_x_random_list[i] = torch.cat([train_x_random_list[i], new_x_random_norm[k].unsqueeze(0)])
                    
                    train_obj_random_list[i] = torch.cat(
                        [train_obj_random_list[i], new_obj_random[k, i].view(1,1)]
                    )
                    k+=1
                train_X_full=torch.cat([train_X_full,new_x_random_norm])
                
                mll_random, model_random = initialize_model(
                    train_x_random_list, train_obj_random_list, bounds
                )
                try:
                    fit_gpytorch_mll(mll_random)
                except ModelFittingError:
                    print("GP fitting failed, restoring previous hyperparameters.")
                    if 'prev_hypers' in globals():
                        model_random.load_state_dict(prev_hypers)
                    else:
                        print("No previous hyperparameters available. Using current defaults.")
                prev_hypers = model_random.state_dict()
                #hv = get_model_identified_hv_maximizing_set(model=model_random)
                hv = hypervolume_from_posterior_mean_gp(model=model_random,X=train_X_full,ncons=args["ncons"],ref_point=args["ref_point"],maximize=True,)

                hvs_random.append(hv)

                #Update true HV using true coupled train_y

                fx=f(new_x_random)
                if args["ncons"] > 0:
                    fc=cons(new_x_random) #Constraints at x_i
                    #print(fx.shape)
                    fx=torch.column_stack((fx,fc))#Combining into objectives+cons
                
                coupled_new_y=fx
                True_coupled_train_y = torch.row_stack([True_coupled_train_y, coupled_new_y])
                true_hv=compute_true_hypervolume(True_coupled_train_y,args["ref_point"],nobj=args["nobj"],ncons=args["ncons"],maximize=True) #taking only the objective points for HV computation
                true_hvs.append(true_hv)

                t2 = time.time()
                times.append(t2 - t1)
                print(f"iter {iter}, Time: {t2 - t1}, HV: {true_hv}, newx {new_x_random}, newy {new_obj_random}\n",flush=True) #iter output statement
                file_saving_inloop(func_sent=func_sent,tag=tag,train_X_full=train_X_full,train_Y_full=train_obj_random_list,hvs=hvs_random,true_hvs=true_hvs,times=times,REP=REP,coupled_y=True_coupled_train_y)
        
        elif acquisition_function == "pesmo":
            from qpots.utils.acq_utils import *
            from botorch import fit_gpytorch_mll

            print("PESMO decoupled:\n",flush=True)
            tag=acquisition_function

            train_obj_list = list(train_Y.split(1, dim=-1))
            train_x_list = [train_X] * len(train_obj_list)
            mll, model = initialize_model(train_x_list, train_obj_list, bounds)

            # fit the models
            fit_gpytorch_mll(mll)
           
            # compute hypervolume
            #hv = hypervolume_from_posterior_mean_gp(model=model,X=train_X_full,ref_point=args["ref_point"],maximize=True,ncons=args["ncons"],)
            hv=true_hv
            print("Initial Hypervolume is: ",hv,flush=True)
            hvs= [hv]
            times = []

            options={
                        "batch_limit": 5,
                        "maxiter": 200,
                        "with_grad": False, #Trying turning it off for PESMO only (on the botorch documentation)
                    }

            for iter in range(args["iters"]):
                t1 = time.time()
            
                pareto_sets, pareto_fronts, hypercell_bounds = build_pareto_state(model, bounds)
                # PESMO
                choice = choose_competitive_decoupled_candidate(
                    lambda mask: qDecoupledPESMO(
                        model=model,
                        pareto_sets=pareto_sets,
                        maximize=True,
                        X_evaluation_mask=mask,
                    ),
                    bounds=bounds,
                    options=options,
                    num_outputs=model.num_outputs,
                    objective_costs=cost_sent,   # optional cost-aware selection, sent from arg parser
                )
                
                X_next = choice["X"]
                objective_idx = choice["objective_idx"]

                Y_next=f(X_next.squeeze())
                if args["ncons"] > 0:
                    fc=cons(X_next.squeeze()) #Constraints at x_i
                    #print(fx.shape)
                    print(f"Checking sizes: Y_next:{Y_next.shape}, fc:{fc.shape}",flush=True)
                    print(f"Checking actuals: Y_next:{Y_next}, fc:{fc}",flush=True)
                    #Y_next=torch.column_stack((Y_next,fc))#Combining into objectives+cons
                    Y_next=torch.cat([Y_next, fc])#Combining into objectives+cons
                    print(f"Post cat checking size and vals: Y_next:{Y_next.shape}, Y_next:{Y_next}",flush=True)
                    
    
                
                train_x_list[objective_idx] = torch.cat([train_x_list[objective_idx], X_next])
                train_obj_list[objective_idx] = torch.cat(
                    [train_obj_list[objective_idx], Y_next[objective_idx].reshape(1, 1)], dim=0
                ) 
                
                mll, model = initialize_model(train_x_list, train_obj_list, bounds)        
                fit_gpytorch_mll(mll)

                train_X_full=torch.cat([train_X_full,X_next])
                

                #Update true HV using true coupled train_y

                True_coupled_train_y = torch.row_stack([True_coupled_train_y, Y_next])
                true_hv=compute_true_hypervolume(True_coupled_train_y,args["ref_point"],nobj=args["nobj"],ncons=args["ncons"],maximize=True)
                true_hvs.append(true_hv)

                hv = true_hv
                hvs.append(hv)

                t2 = time.time()
                times.append(t2 - t1)
                print(f"iter {iter}, Time: {t2 - t1}, HV: {true_hv}, newx {X_next}, newy {Y_next}, evaluated for objective {objective_idx}\n",flush=True) #iter output statement
                file_saving_inloop(func_sent=func_sent,tag=tag,train_X_full=train_X_full,train_Y_full=train_obj_list,hvs=hvs,true_hvs=true_hvs,times=times,REP=REP,coupled_y=True_coupled_train_y)

#Wait for all to finish, then merge
comm.Barrier()  

#Merging all into master file
if rank == 0:
    train_y_all=[]
    coupled_y_all=[]
    train_x_all=[]
    hv_all=[]
    true_hv_all=[]
    times_all=[]
    NSGA_all=[]
    NonInv_all=[]
    
    for REP in range(REPS):
        train_y=np.load(f"../{REP}_{func_sent}_{tag}_train_y.npy", allow_pickle=True)
        coupled_y=np.load(f"../{REP}_{func_sent}_{tag}_coupled_y.npy", allow_pickle=True)
        train_x=np.load(f"../{REP}_{func_sent}_{tag}_train_x.npy")
        hv=np.load(f"../{REP}_{func_sent}_{tag}_hv.npy")
        true_hv=np.load(f"../{REP}_{func_sent}_{tag}_true_hv.npy")
        times=np.load(f"../{REP}_{func_sent}_{tag}_times.npy")
        
        if acquisition_function == "qpots":
            NSGA_iter=np.load(f"../{REP}_{func_sent}_{tag}_exception_handling_NSGA.npy")
            NonInv_iter=np.load(f"../{REP}_{func_sent}_{tag}_Non_invertible_counter.npy")
            NSGA_all.append(NSGA_iter)
            NonInv_all.append(NonInv_iter)
        
        
        coupled_y_all.append(coupled_y)
        train_y_all.append(train_y)
        train_x_all.append(train_x)
        hv_all.append(hv)
        true_hv_all.append(true_hv)
        times_all.append(times)

    
    np.save(f"../all_{func_sent}_{tag}_train_y.npy", np.array(train_y_all, dtype=object))
    np.save(f"../all_{func_sent}_{tag}_coupled_y.npy", np.array(coupled_y_all, dtype=object))
    np.save(f"../all_{func_sent}_{tag}_train_x.npy", np.array(train_x_all, dtype=object))
    np.save(f"../all_{func_sent}_{tag}_hv.npy", np.array(hv_all, dtype=object))
    np.save(f"../all_{func_sent}_{tag}_true_hv.npy", np.array(true_hv_all, dtype=object))
    np.save(f"../all_{func_sent}_{tag}_times.npy", np.array(times_all, dtype=object))

    if acquisition_function == "qpots":
        np.save(f"../all_{func_sent}_{tag}_exception_handling_NSGA.npy", np.array(NSGA_all, dtype=object))
        np.save(f"../all_{func_sent}_{tag}_Non_invertible_counter.npy", np.array(NonInv_all, dtype=object))