#New File to run the examples multiple times without having to reset parameters manually
import subprocess
import numpy as np
import argparse

"""
call of Multi_Seed_Wrapper from the qPOTS folder:
python -m examples.Multi_Seed_Wrapper --ntrain 20 --iters 100 --q 4 --func "branincurrin" --ref_point -18.0 -6.0 --dim 2 --nobj 2 --ncons 0 --use_mtgp --use_partial --var_thresh 0.5 0.5 --num_test 5 --start_seed 1023
using BC with defaults:
python -m examples.Multi_Seed_Wrapper --num_test 5 --start_seed 1023 --use_mtgp --use_partial --var_thresh 0.5 0.5

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
    parser.add_argument("--var_thresh", type=float, default=None, nargs="+", help="Use variance thresholding or not, leave blank if not using, else give the list of thresholds.")

    # Multi_Seed_Wrapper Options
    parser.add_argument("--num_tests", type=int, default=1, required=True, help="Number of tests to run with different seeds.")
    parser.add_argument("--start_seed", type=int, default=1023, required=True, help="Seed that manual_seed starts with.")
    parser.add_argument("--multi_variance_testing", action="store_true", help="Testing multiple variances at the same seed rather than multi-seeds.")

    return parser.parse_args()

args = arg_parser()

func=args.func
q=args.q
dim=args.dim
nobj=args.nobj
ncons=args.ncons
ntrain=args.ntrain
iters=args.iters
ref_point=args.ref_point


if func=="branincurrin":
    file_name="unconstrained_branin_currin"
    test_function_tag="BC"
elif func=="zdt3":
    file_name="zdt3"
    test_function_tag="ZDT3"
elif func=="discbrake":
    file_name="unconstrained_branin_currin"
    test_function_tag="discbrake"
elif func=="DTLZ1":
    file_name="unconstrained_branin_currin"
    test_function_tag="DTLZ1"
elif func=="DTLZ2":
    file_name="unconstrained_branin_currin"
    test_function_tag="DTLZ2"
elif func=="weldedbeam":
    file_name="unconstrained_branin_currin"
    test_function_tag="weldedbeam"
    

multitask=args.use_mtgp
if multitask:
    mtgp_sending=1
else: 
    mtgp_sending=0
partial_info=args.use_partial

if partial_info:
    partial_sending=1
else:
    partial_sending=0

var_thresh_sending=args.var_thresh
if var_thresh_sending is None:
    var_thresh=False
else:
    var_thresh=True

num_tests=args.num_tests
start_seed=args.start_seed

if multitask is True:
    if partial_info is True:
        if var_thresh is True:
            type_tag="var_thresh"
        else:
            type_tag="rand"
    else:
        type_tag="joint"
else:
    type_tag="Model_list"

multi_variance_testing = args.multi_variance_testing
if multi_variance_testing:
    #variance_list=[[-1.0,-1.0],[-.75,-.75],[-.5,-.5],[-.25,-.25],[0.0,0.0],[.25,.25],[.5,.5],[.75,.75],[1.0,1.0]]
    variance_list=[[0.1,0.1],[.2,.2],[.3,.3],[.4,.4],[0.5,0.5],[.6,.6],[.7,.7],[.8,.8],[0.90,.90]]

train_y_all=[]
train_x_all=[]
hv_all=[]
hv_full_all=[]
times_all=[]
pf_all=[]
if partial_info is True:
    train_y_filled_all=[]
    iteration_tracker_all=[]

i=0
print(num_tests)
print(test_function_tag)
print(type_tag)
for manual_seed in range(num_tests):
    
    if multi_variance_testing:
        curr_var=variance_list[i]
        i+=1
        print(f"######################################## RUNNING Test {manual_seed+1} with variance {curr_var} ########################################")
        subprocess.run(["python", "-m", f"examples.{file_name}", str(start_seed), str(func), str(q), str(dim), str(nobj), str(ncons), str(ntrain), str(iters), str(mtgp_sending), str(partial_sending), str(curr_var), str(ref_point)])
    else:
        print(f"######################################## RUNNING Test {manual_seed+1} with seed {manual_seed+start_seed} ########################################")
        subprocess.run(["python", "-m", f"examples.{file_name}", str(manual_seed+start_seed), str(func), str(q), str(dim), str(nobj), str(ncons), str(ntrain), str(iters), str(mtgp_sending), str(partial_sending), str(var_thresh_sending), str(ref_point)])
    train_y=np.load(f"../{test_function_tag}_{type_tag}_train_y.npy")
    train_x=np.load(f"../{test_function_tag}_{type_tag}_train_x.npy")
    hv=np.load(f"../{test_function_tag}_{type_tag}_hv.npy")
    times=np.load(f"../{test_function_tag}_{type_tag}_times.npy")
    pf=np.load(f"../{test_function_tag}_{type_tag}_pareto_front.npy")
    train_y_all.append(train_y)
    train_x_all.append(train_x)
    hv_all.append(hv)
    times_all.append(times)
    pf_all.append(pf)
    if partial_info is True:
        train_y_filled=np.load(f"../{test_function_tag}_{type_tag}_train_y_filled.npy")
        hv_full=np.load(f"../{test_function_tag}_{type_tag}_hv_full.npy")
        iteration_tracker=np.load(f"../{test_function_tag}_{type_tag}_iteration_tracker.npy")
        train_y_filled_all.append(train_y_filled)
        hv_full_all.append(hv_full)
        iteration_tracker_all.append(iteration_tracker)

    if num_tests>1:
        print(num_tests)
        print(test_function_tag)
        print(type_tag)
        np.save(f"../{test_function_tag}_{type_tag}_train_y_all.npy", np.array(train_y_all, dtype=object))
        np.save(f"../{test_function_tag}_{type_tag}_train_x_all.npy", np.array(train_x_all, dtype=object))
        np.save(f"../{test_function_tag}_{type_tag}_hv_all.npy", np.array(hv_all, dtype=object))
        np.save(f"../{test_function_tag}_{type_tag}_times_all.npy", np.array(times_all, dtype=object))
        np.save(f"../{test_function_tag}_{type_tag}_pareto_front_all.npy", np.array(pf_all, dtype=object))
        if partial_info is True:
            np.save(f"../{test_function_tag}_{type_tag}_train_y_filled_all.npy", np.array(train_y_filled_all, dtype=object))
            np.save(f"../{test_function_tag}_{type_tag}_hv_full_all.npy", np.array(hv_full_all, dtype=object))
            np.save(f"../{test_function_tag}_{type_tag}_iteration_tracker_all.npy", np.array(iteration_tracker_all, dtype=object)) #changed dtype to int 9/16

