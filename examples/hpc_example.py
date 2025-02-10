"""
This file demonstrates how to use an HPC environment for parallelization

Example python command
mpirun python3 -m hpc_example --func branincurrin --rp " -300." " -18." --dim 2 --nobj 2 --reps 10 --ntrain 5 --nystrom 0 --iters 50 --wd "."
"""
import warnings
import os
import torch
import time
import numpy as np
from mpi4py import MPI

from qpots.acquisition import Acquisition
from qpots.model_object import ModelObject
from qpots.function import Function
from qpots.utils.utils import expected_hypervolume, arg_parser
from botorch.utils.transforms import unnormalize

# Set up MPI
warnings.filterwarnings("ignore")
comm_world = MPI.COMM_WORLD
world_size = comm_world.Get_size()
world_rank = comm_world.Get_rank()

# Get args from parser
args = arg_parser() # For a list of arguments see the arg parser, these are command line arguments that must be set when running the code
device = torch.device("cpu")

tf = Function(name=args.func, dim=args.dim, nobj=args.nobj)
f = tf.evaluate
bounds = tf.get_bounds()
args.ref_point = torch.tensor(args.ref_point, dtype=torch.double)
cons = tf.get_cons()

os.makedirs(args.wd, exist_ok=True)

for rep in range(args.reps):
    if rep % world_size == world_rank:
        hvs, times = [], []
        torch.manual_seed(1024 + rep)

        train_x = torch.rand([args.ntrain, args.dim], dtype=torch.double, device=device)
        train_y = f(unnormalize(train_x, bounds)).to(device)
        train_y = torch.column_stack([train_y, cons(unnormalize(train_x, bounds))]).to(device) # Stack constraints on top of objectives

        gps = ModelObject(train_x=train_x,
                          train_y=train_y,
                          bounds=bounds,
                          nobj=args.nobj,
                          ncons=args.ncons,
                          device=device)
        gps.fit_gp()

        acq = Acquisition(tf, gps, device=device, q=args.q)

        for i in range(args.iters):
            t1 = time.time()
            newx = acq.qpots(bounds=bounds, iteration=i, **vars(args))
            t2 = time.time()
            times.append(t2 - t1)

            newy = f(unnormalize(newx.reshape(-1, args.dim), bounds))
            newconsy = cons(unnormalize(newx.reshape(-1, args.dim), bounds))
            newy = torch.column_stack([newy.reshape(args.q, args.nobj),
                                newconsy.reshape(args.q, args.ncons)])
            print(f"Iteration: {i}, Newy: {newy}, Newconsy: {newconsy}")
            hv, pf = expected_hypervolume(gps, ref_point=args.ref_point)
            hvs.append(hv)

            print(f"Iteration: {i}, New candidate: {newx}, Time: {t2 - t1}, HV: {hv}", flush=True)
            train_x = torch.row_stack([train_x, newx.view(-1, args.dim)]).to(device)
            train_y = torch.row_stack([train_y, newy]).to(device)
            gps = ModelObject(train_x, train_y, bounds, args.nobj, args.ncons, device=device)
            gps.fit_gp()

            # Save at each iteration for post processing
            np.save(f"{args.wd}/train_x_{rep}.npy", train_x)
            np.save(f"{args.wd}/train_y_{rep}.npy", train_y)
            np.save(f"{args.wd}/hv_{rep}.npy", hvs)
            np.save(f"{args.wd}/times_{rep}.npy", times)