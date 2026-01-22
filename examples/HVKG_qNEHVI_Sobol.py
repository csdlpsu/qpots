import os

import torch
from botorch.test_functions.multi_objective import ZDT2
from botorch.models.cost import FixedCostModel
from botorch.models.gp_regression import SingleTaskGP
from botorch.models.model_list_gp_regression import ModelListGP
from botorch.utils.sampling import draw_sobol_samples
from botorch.utils.transforms import normalize, unnormalize
from gpytorch.mlls.sum_marginal_log_likelihood import SumMarginalLogLikelihood
from torch import Tensor
from gpytorch.priors import GammaPrior
from gpytorch.kernels import MaternKernel, ScaleKernel

from botorch.acquisition.multi_objective.monte_carlo import (
    qNoisyExpectedHypervolumeImprovement,
)
from botorch.acquisition.multi_objective.logei import qLogNoisyExpectedHypervolumeImprovement
from botorch.optim.optimize import optimize_acqf

from botorch.acquisition.cost_aware import InverseCostWeightedUtility
from botorch.acquisition.multi_objective.hypervolume_knowledge_gradient import (
    _get_hv_value_function,
    qHypervolumeKnowledgeGradient,
)
from botorch.models.deterministic import GenericDeterministicModel
from botorch.sampling.list_sampler import ListSampler
from botorch.sampling.normal import IIDNormalSampler

import numpy as np
from botorch.utils.multi_objective.box_decompositions.non_dominated import (
    FastNondominatedPartitioning,
)
from botorch.utils.multi_objective.pareto import _is_non_dominated_loop
from gpytorch import settings

import time
import warnings

from botorch import fit_gpytorch_mll
from botorch.exceptions import BadInitialCandidatesWarning
from botorch.sampling.normal import SobolQMCNormalSampler

from botorch.test_functions.multi_objective import (
    BraninCurrin, DTLZ1, DTLZ2, DTLZ3, DTLZ7, GMM, DH1, Penicillin,
    VehicleSafety, CarSideImpact, ConstrainedBraninCurrin,
    ZDT2,ZDT3, DiscBrake, MW7, OSY, WeldedBeam
)

warnings.filterwarnings("ignore", category=BadInitialCandidatesWarning)
warnings.filterwarnings("ignore", category=RuntimeWarning)

#tkwargs
tkwargs = {
    "dtype": torch.double,
    "device": torch.device("cuda" if torch.cuda.is_available() else "cpu"),
}
SMOKE_TEST = os.environ.get("SMOKE_TEST")


#Problem setup
problem = BraninCurrin(negate=True).to(**tkwargs)
ref_point=problem.ref_point
print("Ref Point:\n",ref_point)
#ref_point=torch.tensor([-300.0,-20.0])

# define the cost model
objective_costs = {0: 3.0, 1: 1.0}
objective_indices = list(objective_costs.keys())
objective_costs = {int(k): v for k, v in objective_costs.items()}
objective_costs_t = torch.tensor(
    [objective_costs[k] for k in sorted(objective_costs.keys())], **tkwargs
)
cost_model = FixedCostModel(fixed_cost=objective_costs_t)

#Generate Initial Data
def generate_initial_data(n):
    # generate training data
    train_x = draw_sobol_samples(bounds=problem.bounds, n=n, q=1).squeeze(1)
    train_obj_true = problem(train_x)
    return train_x, train_obj_true

#initializing model
def initialize_model(train_x_list, train_obj_list):
    # define models for objective and constraint
    train_x_list = [normalize(train_x, problem.bounds) for train_x in train_x_list]
    models = []
    for i in range(len(train_obj_list)):
        train_y = train_obj_list[i]
        train_yvar = torch.full_like(train_y, 1e-7)  # noiseless
        models.append(
            SingleTaskGP(
                train_X=train_x_list[i],
                train_Y=train_y,
                train_Yvar=train_yvar,
                covar_module=ScaleKernel(
                    MaternKernel(
                        nu=2.5,
                        ard_num_dims=train_x_list[0].shape[-1],
                        lengthscale_prior=GammaPrior(2.0, 2.0),
                    ),
                    outputscale_prior=GammaPrior(2.0, 0.15),
                )
            )
        )
    model = ModelListGP(*models)
    mll = SumMarginalLogLikelihood(model.likelihood, model)
    return mll, model

BATCH_SIZE = 1
NUM_RESTARTS = 10 if not SMOKE_TEST else 2
RAW_SAMPLES = 512 if not SMOKE_TEST else 4

standard_bounds = torch.zeros(2, problem.dim, **tkwargs)
standard_bounds[1] = 1


def optimize_qnehvi_and_get_observation(model, train_x, sampler):
    """Optimizes the qNEHVI acquisition function, and returns a new candidate and observation."""
    # partition non-dominated space into disjoint rectangles
    acq_func = qLogNoisyExpectedHypervolumeImprovement(
        model=model,
        ref_point=ref_point.tolist(),  # use known reference point
        X_baseline=normalize(train_x, problem.bounds),
        prune_baseline=True,  # prune baseline points that have estimated zero probability of being Pareto optimal
        sampler=sampler,
    )
    # optimize
    candidates, _ = optimize_acqf(
        acq_function=acq_func,
        bounds=standard_bounds,
        q=BATCH_SIZE,
        num_restarts=NUM_RESTARTS,
        raw_samples=RAW_SAMPLES,  # used for intialization heuristic
        options={"batch_limit": 5, "maxiter": 200},
        sequential=True,
    )
    # observe new values
    new_x = unnormalize(candidates.detach(), bounds=problem.bounds)
    new_obj_true = problem(new_x)
    return new_x, new_obj_true

NUM_PARETO = 2 if SMOKE_TEST else 10
NUM_FANTASIES = 2 if SMOKE_TEST else 8
NUM_HVKG_RESTARTS = 1

#Get HV helper
def get_current_value(
    model,
    ref_point,
    bounds,
):
    """Helper to get the hypervolume of the current hypervolume
    maximizing set.
    """
    curr_val_acqf = _get_hv_value_function(
        model=model,
        ref_point=ref_point,
        use_posterior_mean=True,
    )
    _, current_value = optimize_acqf(
        acq_function=curr_val_acqf,
        bounds=bounds,
        q=NUM_PARETO,
        num_restarts=20,
        raw_samples=1024,
        return_best_only=True,
        options={"batch_limit": 5},
    )
    return current_value

#optimize
def optimize_HVKG_and_get_obs_decoupled(model,problem):
    """Utility to initialize and optimize HVKG."""
    cost_aware_utility = InverseCostWeightedUtility(cost_model=cost_model)

    current_value = get_current_value(
        model=model,
        ref_point=ref_point,
        bounds=standard_bounds,
    )

    acq_func = qHypervolumeKnowledgeGradient(
        model=model,
        ref_point=ref_point,  # use known reference point
        num_fantasies=NUM_FANTASIES,
        num_pareto=NUM_PARETO,
        current_value=current_value,
        cost_aware_utility=cost_aware_utility,
    )

    # optimize acquisition functions and get new observations
    objective_vals = []
    objective_candidates = []
    for objective_idx in objective_indices:
        # set evaluation index to only condition on one objective
        # this could be multiple objectives
        X_evaluation_mask = torch.zeros(
            1,
            len(objective_indices),
            dtype=torch.bool,
            device=standard_bounds.device,
        )
        X_evaluation_mask[0, objective_idx] = 1
        acq_func.X_evaluation_mask = X_evaluation_mask
        candidates, vals = optimize_acqf(
            acq_function=acq_func,
            num_restarts=NUM_HVKG_RESTARTS,
            raw_samples=RAW_SAMPLES,
            bounds=standard_bounds,
            q=BATCH_SIZE,
            sequential=False,
            options={"batch_limit": 5},
        )
        objective_vals.append(vals.view(-1))
        objective_candidates.append(candidates)
    best_objective_index = torch.cat(objective_vals, dim=-1).argmax().item()
    eval_objective_indices = [best_objective_index]
    candidates = objective_candidates[best_objective_index]
    vals = objective_vals[best_objective_index]
    # observe new values
    new_x = unnormalize(candidates.detach(), bounds=problem.bounds)
    new_obj = problem(new_x)
    new_obj = new_obj[..., eval_objective_indices]
    return new_x, new_obj, eval_objective_indices

from botorch.utils.multi_objective.box_decompositions import FastNondominatedPartitioning
from botorch.utils.multi_objective.hypervolume import Hypervolume
from botorch.utils.multi_objective.pareto import is_non_dominated

def hypervolume_from_posterior_mean_gp(
    model,
    X: torch.Tensor,                      # (n, d) base features (no task column)
    *,
    ref_point,
    maximize,
) -> torch.Tensor:
    model.eval()

    # Infer K (#objectives/tasks) from ref_point
    if not torch.is_tensor(ref_point):
        ref_point = torch.tensor(ref_point)

    # Move to model device/dtype
    p = next(model.parameters())
    device, dtype = p.device, p.dtype
    X = X.to(device=device, dtype=dtype)
    ref_point = ref_point.to(device=device, dtype=dtype).view(-1)
    K = ref_point.numel()

    # Build long-format inputs and get posterior mean for each (x, task)
    print("Pulling the Posterior")
    post = model.posterior(X)
    mean_flat = post.mean.squeeze(-1)  # (n*K,)

    n = X.shape[0]
    Y_mean = mean_flat.view(K, n).transpose(0, 1).contiguous()  # (n, K)
    #print("Y_mean",Y_mean)
    #print("ref_point",ref_point)
    # Hypervolume assumes maximization. If minimizing, negate both.
    if not maximize:
        Y_mean = -Y_mean
        ref_point = -ref_point

    # Non-dominated subset of posterior means
    nd_mask = is_non_dominated(Y_mean)
    pareto_Y = Y_mean[nd_mask]

    if pareto_Y.numel() == 0:
        # Shouldn't happen unless n=0, but keep it safe:
        return torch.zeros((), device=device, dtype=dtype)

    hv = Hypervolume(ref_point=ref_point).compute(pareto_Y)
    return hv


########## Optimization Loop ##############

args = dict(
    {
        
        "iters": 10,
        "ref_point": ref_point,
        
    }
)

MC_SAMPLES = 128 if not SMOKE_TEST else 16
COST_BUDGET = 90 if not SMOKE_TEST else 54
torch.manual_seed(0)
verbose = True
N_INIT = 10 * problem.dim
print("N_INIT",N_INIT)

total_cost = {"hvkg": 0.0, "qnehvi": 0.0, "random": 0.0}


# call helper functions to generate initial training data and initialize model
torch.manual_seed(1023)
#train_x_hvkg, train_obj_hvkg = generate_initial_data(n=N_INIT)
train_x_hvkg = torch.rand([N_INIT, problem.dim], dtype=torch.double)
train_obj_hvkg = problem(train_x_hvkg)
train_X_full=train_x_hvkg
train_obj_hvkg_list = list(train_obj_hvkg.split(1, dim=-1))
train_x_hvkg_list = [train_x_hvkg] * len(train_obj_hvkg_list)
mll_hvkg, model_hvkg = initialize_model(train_x_hvkg_list, train_obj_hvkg_list)
train_obj_random_list = train_obj_hvkg_list
train_x_random_list = train_x_hvkg_list
train_x_qnehvi_list, train_obj_qnehvi_list = (
    train_x_hvkg_list,
    train_obj_hvkg_list,
)
cost_hvkg = cost_model(train_x_hvkg).sum(dim=-1)
total_cost["hvkg"] += cost_hvkg.sum().item()
cost_qnehvi = cost_hvkg
cost_random = cost_hvkg
total_cost["qnehvi"] = total_cost["hvkg"]
total_cost["random"] = total_cost["hvkg"]
mll_qnehvi, model_qnehvi = initialize_model(train_x_qnehvi_list, train_obj_qnehvi_list)
mll_random, model_random = initialize_model(train_x_random_list, train_obj_random_list)
# fit the models
fit_gpytorch_mll(mll_hvkg)
fit_gpytorch_mll(mll_qnehvi)
fit_gpytorch_mll(mll_random)
# compute hypervolume
#hv = get_model_identified_hv_maximizing_set(model=model_hvkg)
hv = hypervolume_from_posterior_mean_gp(model=model_hvkg,X=train_X_full,ref_point=torch.tensor([-300.0,-20.0]),maximize=True,)
hvs_hvkg, hvs_qnehvi, hvs_random = [hv], [hv], [hv]

# run N_BATCH rounds of BayesOpt after the initial random batch
iteration = 0
acquisition_function = "hvkg"
#acquisition_function = "qnehvi"
#acquisition_function = "sobol"


#while any(v < COST_BUDGET for v in total_cost.values()):
for iter in range(args["iters"]):
    t0 = time.monotonic()
    if acquisition_function == "hvkg":
        # generate candidates
        print("\nHVKG:\n")
        (
            new_x_hvkg,
            new_obj_hvkg,
            eval_objective_indices_hvkg,
        ) = optimize_HVKG_and_get_obs_decoupled(
            model_hvkg,problem=problem
        )
        # update training points
        for i in eval_objective_indices_hvkg:
            train_x_hvkg_list[i] = torch.cat([train_x_hvkg_list[i], new_x_hvkg])
            train_obj_hvkg_list[i] = torch.cat(
                [train_obj_hvkg_list[i], new_obj_hvkg], dim=0
            )
        train_X_full=torch.cat([train_X_full,new_x_hvkg])
        print("train_X_full",train_X_full)
        # update costs
        all_outcome_cost = cost_model(new_x_hvkg)
        new_cost_hvkg = all_outcome_cost[..., eval_objective_indices_hvkg].sum(dim=-1)
        cost_hvkg = torch.cat([cost_hvkg, new_cost_hvkg], dim=0)
        total_cost["hvkg"] += new_cost_hvkg.sum().item()
        # fit models
        mll_hvkg, model_hvkg = initialize_model(train_x_hvkg_list, train_obj_hvkg_list)
        
        fit_gpytorch_mll(mll_hvkg)
        #hv = get_model_identified_hv_maximizing_set(model=model_hvkg)
        hv = hypervolume_from_posterior_mean_gp(model=model_hvkg,X=train_X_full,ref_point=torch.tensor([-300.0,-20.0]),maximize=True,)

        hvs_hvkg.append(hv)

        t1 = time.monotonic()
        print(f"Iter: {iter}, Hypervolume: {hv}, time = {t1-t0:>4.2f}")

    if acquisition_function == "qnehvi":
        print("qNEHVI")
        qnehvi_sampler = SobolQMCNormalSampler(sample_shape=torch.Size([MC_SAMPLES]))
        # generate candidates
        new_x_qnehvi, new_obj_qnehvi = optimize_qnehvi_and_get_observation(
            model_qnehvi, train_x_qnehvi_list[0], qnehvi_sampler
        )
        # update training points
        for i in objective_indices:
            train_x_qnehvi_list[i] = torch.cat([train_x_qnehvi_list[i], new_x_qnehvi])
            train_obj_qnehvi_list[i] = torch.cat(
                [train_obj_qnehvi_list[i], new_obj_qnehvi[..., i : i + 1]]
            )
        train_X_full=torch.cat([train_X_full,new_x_qnehvi])
        # update costs
        new_cost_qnehvi = cost_model(new_x_qnehvi).sum(dim=-1)
        cost_qnehvi = torch.cat([cost_qnehvi, new_cost_qnehvi], dim=0)
        total_cost["qnehvi"] += new_cost_qnehvi.sum().item()
        # fit models
        mll_qnehvi, model_qnehvi = initialize_model(
            train_x_qnehvi_list, train_obj_qnehvi_list
        )
        fit_gpytorch_mll(mll_qnehvi)
        #hv = get_model_identified_hv_maximizing_set(model=model_qnehvi)
        hv = hypervolume_from_posterior_mean_gp(model=model_qnehvi,X=train_X_full,ref_point=torch.tensor([-300.0,-20.0]),maximize=True,)

        hvs_qnehvi.append(hv)

        t1 = time.monotonic()
        print(f"Iter: {iter}, Hypervolume: {hv}, time = {t1-t0:>4.2f}")

    if acquisition_function == "sobol":
        # generate candidates
        print("Sobol")
        new_x_random, new_obj_random = generate_initial_data(n=BATCH_SIZE)
        # update training points
        for i in objective_indices:
            train_x_random_list[i] = torch.cat([train_x_random_list[i], new_x_random])
            train_obj_random_list[i] = torch.cat(
                [train_obj_random_list[i], new_obj_random[..., i : i + 1]]
            )
        train_X_full=torch.cat([train_X_full,new_x_random])
        # update costs
        new_cost_random = cost_model(new_x_random).sum(dim=-1)
        cost_random = torch.cat([cost_random, new_cost_random], dim=0)
        total_cost["random"] += new_cost_random.sum().item()
        # fit models
        mll_random, model_random = initialize_model(
            train_x_random_list, train_obj_random_list
        )
        fit_gpytorch_mll(mll_random)
        #hv = get_model_identified_hv_maximizing_set(model=model_random)
        hv = hypervolume_from_posterior_mean_gp(model=model_random,X=train_X_full,ref_point=torch.tensor([-300.0,-20.0]),maximize=True,)

        hvs_random.append(hv)

        t1 = time.monotonic()
        print(f"Iter: {iter}, Hypervolume: {hv}, time = {t1-t0:>4.2f}")

    t1 = time.monotonic()
    """
    if verbose:
        print(
            f"\nBatch {iteration:>2}: Costs (random, qHVKG, qNEHVI) = "
            f"({total_cost['random']:>4.2f}, {total_cost['hvkg']:>4.2f}, {total_cost['qnehvi']:>4.2f}). "
        )
        print(
            f"\nHypervolume (random, qHVKG, qNEHVI) = "
            f"({hvs_random[-1]:>4.2f}, {hvs_hvkg[-1]:>4.2f}, {hvs_qnehvi[-1]:>4.2f}), "
            f"time = {t1-t0:>4.2f}.",
            end="",
        )
    else:
        print(".", end="")
    iteration += 1
    """

########## Plotting ############
from matplotlib import pyplot as plt


log_hv_difference_hvkg = np.log10(problem.max_hv - np.asarray(hvs_hvkg))
log_hv_difference_qnehvi = np.log10(problem.max_hv - np.asarray(hvs_qnehvi))
log_hv_difference_rnd = np.log10(problem.max_hv - np.asarray(hvs_random))

fig, ax = plt.subplots(1, 1, figsize=(8, 6))
running_cost_random = np.cumsum(cost_random.cpu().numpy()[N_INIT-1:])
running_cost_qnehvi = np.cumsum(cost_qnehvi.cpu().numpy()[N_INIT-1:])
running_cost_hvkg = np.cumsum(cost_hvkg.cpu().numpy()[N_INIT-1:])
ax.errorbar(
    running_cost_random,
    log_hv_difference_rnd[: len(running_cost_random)],
    label="Sobol",
    linewidth=1.5,
    ls="--",
    marker="s",
)
ax.errorbar(
    running_cost_qnehvi,
    log_hv_difference_qnehvi[: len(running_cost_qnehvi)],
    label="qNEHVI",
    linewidth=1.5,
    ls="--",
    marker="o"
)
ax.errorbar(
    running_cost_hvkg,
    log_hv_difference_hvkg[: len(running_cost_hvkg)],
    label="HVKG",
    linewidth=1.5,
    ls="--",
    marker="d"
)
ax.set(
    xlabel="Cost",
    ylabel="Log Hypervolume Difference",
)
ax.legend(loc="upper right")