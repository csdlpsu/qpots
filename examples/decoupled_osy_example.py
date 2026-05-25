"""
Decoupled qPOTS-DOE example on the OSY benchmark.

OSY is a 6-dimensional constrained problem with 2 objectives and 6 inequality
constraints. In many real-world settings the objectives and constraints are
measured by separate simulators or laboratory analyses, so evaluating all of
them together at every candidate is wasteful. This example uses qPOTS-DOE to
decide -- at each candidate point -- which oracle subset to query, based on
the total posterior correlation among the multitask GP's tasks.

Key settings
------------
mt=1             -- use a joint MultiTaskGP over all objectives + constraints
partial_info=1   -- enable decoupled oracle selection
threshold=1e-4   -- only decouple when total correlation exceeds this value;
                    set threshold=None for random (unconditional) decoupling
"""

import time
import warnings

import torch
from botorch.utils.transforms import unnormalize

from qpots.acquisition import Acquisition
from qpots.config import DEFAULT_DEVICE, DEFAULT_DTYPE
from qpots.function import Function
from qpots.model_object import ModelObject
from qpots.utils.utils import compute_true_hypervolume, posterior_mean_fill

warnings.filterwarnings("ignore")

# ---------------------------------------------------------------------------
# Problem: OSY -- 6-D input, 2 objectives, 6 constraints
# ---------------------------------------------------------------------------
DIM   = 6
NOBJ  = 2
NCONS = 6

# Reference point for hypervolume (negated objectives, so values are negative)
REF_POINT = torch.tensor([-300.0, -15.0], device=DEFAULT_DEVICE, dtype=DEFAULT_DTYPE)

settings = {
    "ntrain":       10 * DIM,   # 60 initial training points
    "iters":        50,
    "q":            2,          # candidates per iteration
    "wd":           ".",
    "ref_point":    REF_POINT,
    "dim":          DIM,
    "nobj":         NOBJ,
    "ncons":        NCONS,
    "nystrom":      0,
    "nychoice":     "pareto",
    "ngen":         20,
    # --- qPOTS-DOE settings ---
    "mt":           1,          # use MultiTaskGP
    "partial_info": 1,          # enable decoupled oracle selection
    "threshold":    1e-4,       # total-correlation gate (None = always decouple)
}

# ---------------------------------------------------------------------------
# Setup
# ---------------------------------------------------------------------------
test_function = Function("osy", dim=settings["dim"], nobj=settings["nobj"])
evaluate      = test_function.evaluate
get_cons      = test_function.get_cons
bounds        = test_function.get_bounds()
cons          = get_cons()

torch.manual_seed(1023)

train_x = torch.rand(
    settings["ntrain"], settings["dim"],
    device=DEFAULT_DEVICE, dtype=DEFAULT_DTYPE,
)
train_y_obj  = evaluate(unnormalize(train_x, bounds))
train_y_cons = cons(unnormalize(train_x, bounds))
train_y      = torch.column_stack([train_y_obj, train_y_cons])

# Keep a fully-observed copy for hypervolume tracking
full_train_y = train_y.clone()

# ---------------------------------------------------------------------------
# Fit the initial MultiTaskGP (one joint GP over all NOBJ + NCONS tasks)
# ---------------------------------------------------------------------------
gps = ModelObject(
    train_x=train_x,
    train_y=train_y,
    bounds=bounds,
    nobj=settings["nobj"],
    ncons=settings["ncons"],
    ntrain=settings["ntrain"],
    device=DEFAULT_DEVICE,
)
gps.fit_multitask_gp()

acq = Acquisition(test_function, gps, cons=cons, device=DEFAULT_DEVICE, q=settings["q"])

# Initial hypervolume (feasible points only)
hv = compute_true_hypervolume(
    full_train_y,
    ref_point=REF_POINT,
    nobj=NOBJ,
    ncons=NCONS,
    maximize=True,
)
print(f"Initial hypervolume: {hv:.4f}")

# ---------------------------------------------------------------------------
# Optimization loop
# ---------------------------------------------------------------------------
for iteration in range(settings["iters"]):
    t0 = time.time()

    # qPOTS-DOE returns (candidates, task_ids) when partial_info=1.
    # task_ids[i] is a 1-D tensor of oracle indices actually queried for
    # candidate i; un-queried tasks remain NaN in train_y.
    new_x, new_task_ids = acq.qpots(bounds=bounds, iteration=iteration, **settings)

    elapsed = time.time() - t0

    # Evaluate the full oracle (all objectives + constraints) for bookkeeping,
    # then mask out the tasks that were NOT selected by the MI subset rule.
    full_new_y_obj  = evaluate(unnormalize(new_x.reshape(-1, DIM), bounds))
    full_new_y_cons = cons(unnormalize(new_x.reshape(-1, DIM), bounds))
    full_new_y      = torch.column_stack([
        full_new_y_obj.reshape(new_x.shape[0], NOBJ),
        full_new_y_cons.reshape(new_x.shape[0], NCONS),
    ])

    # Build the partially-observed new_y: NaN where the task was not queried.
    new_y = torch.full_like(full_new_y, float("nan"))
    for j in range(new_x.shape[0]):
        cols = new_task_ids[j]
        valid = ~torch.isnan(cols)
        selected = cols[valid].long()
        new_y[j, selected] = full_new_y[j, selected]

    n_queried = (~torch.isnan(new_y)).sum().item()
    n_total   = new_y.numel()
    print(
        f"Iteration {iteration:3d} | "
        f"oracles queried: {n_queried}/{n_total} | "
        f"time: {elapsed:.2f}s | "
        f"HV: {hv:.4f}"
    )

    # Update training data (partially observed rows use NaN for missing tasks)
    train_x = torch.row_stack([train_x, new_x.view(-1, DIM)])
    train_y = torch.row_stack([train_y, new_y])
    full_train_y = torch.row_stack([full_train_y, full_new_y])

    # Hypervolume on the fully-observed data (ground truth for comparison)
    hv = compute_true_hypervolume(
        full_train_y,
        ref_point=REF_POINT,
        nobj=NOBJ,
        ncons=NCONS,
        maximize=True,
    )

    # Refit the MultiTaskGP; NaN entries are handled internally by the model
    gps = ModelObject(
        train_x=train_x,
        train_y=train_y,
        bounds=bounds,
        nobj=settings["nobj"],
        ncons=settings["ncons"],
        ntrain=settings["ntrain"],
        device=DEFAULT_DEVICE,
    )
    gps.fit_multitask_gp()
    acq = Acquisition(test_function, gps, cons=cons, device=DEFAULT_DEVICE, q=settings["q"])

# ---------------------------------------------------------------------------
# Fill NaN entries with MTGP posterior means for downstream analysis
# ---------------------------------------------------------------------------
train_y_filled = posterior_mean_fill(gps)
print("\nOptimization complete.")
print(f"Final hypervolume: {hv:.4f}")
print(f"train_y shape (with NaNs filled): {train_y_filled.shape}")
