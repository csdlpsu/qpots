#!/usr/bin/env python3
"""
qPOTS noise ablation on Branin-Currin
-------------------------------------
Install plotting dependencies with ``pip install qpots[examples]``.

Runs qPOTS on the Branin-Currin bi-objective test function across multiple
Gaussian observation noise levels, repetitions, and budgets.

Outputs:
- CSV per noise level with hypervolume (per iteration & per repetition)
- One combined CSV with mean/std aggregations
- A Matplotlib PNG plot of HV vs iteration (mean ± 1 std) for each noise
- A JSON metadata file with the exact settings

To run do:
    python qpots_noise_ablation.py \
        --outdir results/noise_ablation \
        --noise_vars 0.0 1e-6 1e-4 1e-3 \
        --reps 10 --ntrain 20 --iters 40 --q 1 --seed 123

Notes:
- This script uses qPOTS' built-in Branin-Currin function:
  qpots.function.Function(name='branincurrin', nobj=2)
- We compute the hypervolume in the *maximization* sense (qpots default),
  and we set the reference point automatically from a random design.
"""
import argparse
import json
import math
import os
import time
from pathlib import Path
from typing import List, Tuple, Dict

import numpy as np
import pandas as pd
import torch
import warnings

# qPOTS imports
from botorch.utils.transforms import unnormalize
from qpots.acquisition import Acquisition
from qpots.config import DEFAULT_DEVICE, DEFAULT_DTYPE
from qpots.model_object import ModelObject
from qpots.function import Function
from qpots.utils.utils import expected_hypervolume

warnings.filterwarnings("ignore")

# -------------------------- Utilities --------------------------

def set_torch(device: str = "auto", seed: int = 0):
    torch.manual_seed(seed)
    if device == "cuda" and torch.cuda.is_available():
        return torch.device("cuda")
    if device == "cpu":
        return torch.device("cpu")
    return DEFAULT_DEVICE


def auto_ref_point(func: Function, bounds: torch.Tensor, dim: int, nobj: int,
                   n_samples: int = 2048, minimize: bool = False, seed: int = 0) -> torch.Tensor:
    """
    Estimate a reasonable reference point by sampling the design space.
    For maximization HV (default in qPOTS), the ref point should be
    *worse* than all observed values (i.e., componentwise smaller).
    For minimization HV, it should be componentwise larger.
    """
    g = torch.Generator().manual_seed(seed)
    X = torch.rand((n_samples, dim), dtype=DEFAULT_DTYPE, generator=g)
    Y = func.evaluate(unnormalize(X, bounds))  # (n_samples, nobj)
    y_min = Y.min(dim=0).values
    y_max = Y.max(dim=0).values
    margin = 0.05 * (y_max - y_min).clamp_min(1e-12)
    if not minimize:
        # Maximization: set ref below the worst observed with a bit of slack
        ref = (y_min - margin).to(dtype=DEFAULT_DTYPE)
    else:
        # Minimization: set ref above the worst observed with a bit of slack
        ref = (y_max + margin).to(dtype=DEFAULT_DTYPE)
    return ref


def add_gaussian_noise(Y: torch.Tensor, noise_var: float, g: torch.Generator) -> torch.Tensor:
    if noise_var <= 0.0:
        return Y
    std = math.sqrt(noise_var)
    noise = torch.normal(
        mean=0.0,
        std=std,
        size=Y.shape,
        generator=g,
        dtype=Y.dtype,
    )
    return Y + noise


def run_single_rep(rep: int,
                   noise_var: float,
                   args: argparse.Namespace,
                   device: torch.device,
                   tf: Function,
                   bounds: torch.Tensor,
                   ref_point: torch.Tensor) -> Dict:
    """
    Runs one qPOTS replication for a given noise variance.
    Returns a dict with hypervolume trace and metadata.
    """
    dim = args.dim
    nobj = args.nobj

    # Random generators per-rep for reproducibility
    base_seed = args.seed + 1000 * rep + int(noise_var * 1e9) % 1000
    g = torch.Generator().manual_seed(base_seed)

    # Initial random design in [0,1]^d, then unnormalize to real bounds
    train_x = torch.rand([args.ntrain, dim], dtype=DEFAULT_DTYPE, generator=g)
    true_y = tf.evaluate(unnormalize(train_x, bounds))
    train_y = add_gaussian_noise(true_y, noise_var, g)

    # Fit initial multi-objective GPs
    gps = ModelObject(train_x=train_x, train_y=train_y, bounds=bounds,
                      nobj=nobj, ncons=0, device=device)
    gps.fit_gp()

    # Acquisition (qPOTS)
    acq = Acquisition(tf, gps, device=device, q=args.q)

    # Iterations
    hv_trace = []
    iter_times = []

    for it in range(args.iters):
        t1 = time.time()
        # Propose next candidate(s)
        newx = acq.qpots(bounds, it, nystrom=args.nystrom, nychoice=args.nychoice,
                         dim=dim, ngen=args.ngen, q=args.q, iters=args.iters)
        t2 = time.time()

        # Evaluate (true function), then add observation noise
        newy_true = tf.evaluate(unnormalize(newx.reshape(-1, dim), bounds))
        newy = add_gaussian_noise(newy_true, noise_var, g)

        # HV on the *current* GP (before updating with new point) – conventional
        hv_val, _ = expected_hypervolume(gps, ref_point=ref_point, min=args.minimize)
        hv_trace.append(float(hv_val))
        iter_times.append(t2 - t1)

        # Update dataset and refit models
        train_x = torch.row_stack([train_x, newx.view(-1, dim)])
        train_y = torch.row_stack([train_y, newy])
        gps = ModelObject(train_x=train_x, train_y=train_y, bounds=bounds,
                          nobj=nobj, ncons=0, device=device)
        gps.fit_gp()
        acq = Acquisition(tf, gps, device=device, q=args.q)

    # Final HV after last update
    hv_val, _ = expected_hypervolume(gps, ref_point=ref_point, min=args.minimize)
    hv_trace.append(float(hv_val))

    return dict(
        rep=rep,
        noise_var=noise_var,
        hv_trace=hv_trace,  # length = iters + 1 (includes final)
        iter_times=iter_times,
        seed=base_seed,
    )


def aggregate_and_save(all_runs, args: argparse.Namespace, outdir: Path):
    """
    Save per-rep CSV and aggregate mean/std CSV; plot HV traces.
    """
    outdir.mkdir(parents=True, exist_ok=True)

    # Dump raw per-rep traces
    rows = []
    for run in all_runs:
        rep = run["rep"]
        nv = run["noise_var"]
        hv = run["hv_trace"]
        for t, hv_t in enumerate(hv):
            rows.append({
                "noise_var": nv,
                "rep": rep,
                "iteration": t,  # 0..iters (last is final HV after last update)
                "hypervolume": hv_t,
            })
    df = pd.DataFrame(rows)
    raw_csv = outdir / "hv_traces_raw.csv"
    df.to_csv(raw_csv, index=False)

    # Aggregate mean and std over reps at each iteration for each noise
    agg = (
        df.groupby(["noise_var", "iteration"])["hypervolume"]
          .agg(["mean", "std"])
          .reset_index()
          .sort_values(["noise_var", "iteration"])
    )
    agg_csv = outdir / "hv_traces_agg.csv"
    agg.to_csv(agg_csv, index=False)

    # Save metadata
    meta = {
        "args": vars(args),
        "timestamp": time.strftime("%Y-%m-%d %H:%M:%S"),
        "raw_csv": str(raw_csv),
        "agg_csv": str(agg_csv),
    }
    with open(outdir / "metadata.json", "w") as f:
        json.dump(meta, f, indent=2)

    # Plot mean ± std for each noise level
    import matplotlib.pyplot as plt

    # single plot (no subplots); default styles (no color set)
    plt.figure()
    for nv, gdf in agg.groupby("noise_var"):
        iters = gdf["iteration"].values
        mu = gdf["mean"].values
        sd = gdf["std"].values
        plt.plot(iters, mu, label=f"noise_var={nv:g}")
        plt.fill_between(iters, mu - sd, mu + sd, alpha=0.2)

    plt.xlabel("Iteration (including final)")
    plt.ylabel("Hypervolume (expected)")
    plt.title("qPOTS on Branin-Currin: noise ablation")
    plt.legend()
    png_path = outdir / "hv_vs_iteration.png"
    plt.tight_layout()
    plt.savefig(png_path, dpi=200)
    print(f"Saved plot to: {png_path}")


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--outdir", type=str, default="results/qpots_noise_ablation")
    p.add_argument("--device", type=str, choices=["cpu", "cuda"], default="cpu")
    p.add_argument("--seed", type=int, default=1023)
    p.add_argument("--dim", type=int, default=2)
    p.add_argument("--nobj", type=int, default=2)
    p.add_argument("--ntrain", type=int, default=20, help="Initial design size")
    p.add_argument("--iters", type=int, default=40, help="BO iterations (per rep)")
    p.add_argument("--q", type=int, default=1, help="Batch size")
    p.add_argument("--reps", type=int, default=10, help="Repetitions per noise level")
    p.add_argument("--noise_vars", type=float, nargs="+",
                   default=[0.0, 1e-6, 1e-4, 1e-3], help="Gaussian noise variances")
    p.add_argument("--nystrom", type=int, default=0, help="Use Nyström approximation (1=yes, 0=no)")
    p.add_argument("--nychoice", type=str, default="pareto", help="Nyström column selection")
    p.add_argument("--ngen", type=int, default=10, help="NSGA-II generations in qPOTS internals")
    p.add_argument("--minimize", action="store_true", help="Compute HV in minimization sense")
    args = p.parse_args()

    outdir = Path(args.outdir)
    device = set_torch(args.device, seed=args.seed)

    # Define function & bounds
    tf = Function("branincurrin", dim=args.dim, nobj=args.nobj)
    bounds = tf.get_bounds()

    # Build a reference point automatically based on random sampling
    ref_point = auto_ref_point(tf, bounds, args.dim, args.nobj,
                               n_samples=4096, minimize=args.minimize, seed=args.seed)
    print(f"Ref point (computed): {ref_point.detach().cpu().numpy()}  (minimize={args.minimize})")

    all_runs = []
    for nv in args.noise_vars:
        for rep in range(args.reps):
            run = run_single_rep(rep=rep, noise_var=float(nv), args=args,
                                 device=device, tf=tf, bounds=bounds,
                                 ref_point=ref_point)
            all_runs.append(run)
            print(f"[noise_var={nv}] rep={rep} done. Final HV={run['hv_trace'][-1]:.4f}")

    aggregate_and_save(all_runs, args, outdir)
    print(f"Saved outputs under: {outdir.resolve()}")


if __name__ == "__main__":
    main()
