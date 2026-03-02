import torch
from botorch.models.cost import FixedCostModel
from botorch.models.gp_regression import SingleTaskGP
from botorch.models.model_list_gp_regression import ModelListGP
from botorch.utils.sampling import draw_sobol_samples
from botorch.utils.transforms import normalize, unnormalize
from gpytorch.mlls.sum_marginal_log_likelihood import SumMarginalLogLikelihood
from torch import Tensor
from gpytorch.priors import GammaPrior
from gpytorch.kernels import MaternKernel, ScaleKernel
from botorch.utils.multi_objective.hypervolume import Hypervolume
from botorch.utils.multi_objective.pareto import is_non_dominated

from botorch.acquisition.cost_aware import InverseCostWeightedUtility
from botorch.acquisition.multi_objective.hypervolume_knowledge_gradient import (
    _get_hv_value_function,
    qHypervolumeKnowledgeGradient,
)
from botorch.acquisition.objective import ConstrainedMCObjective

from botorch.optim.optimize import optimize_acqf

from botorch.acquisition.multi_objective.logei import qLogNoisyExpectedHypervolumeImprovement
from botorch.acquisition.multi_objective.objective import IdentityMCMultiOutputObjective, FeasibilityWeightedMCMultiOutputObjective 

#initializing model
def initialize_model(train_x_list, train_obj_list, bounds):
    # define models for objective and constraint
    train_x_list = [normalize(train_x, bounds) for train_x in train_x_list]
    models = []
    for i in range(len(train_obj_list)): #assumes constraints are already put into train_obj_list
        train_y = train_obj_list[i]
        train_yvar = torch.full_like(train_y, 1e-7)  # noiseless
        models.append(
            SingleTaskGP(
                train_X=train_x_list[i],
                train_Y=train_y,
                train_Yvar=train_yvar,
                #covar_module=ScaleKernel(MaternKernel(nu=2.5,ard_num_dims=train_x_list[0].shape[-1],lengthscale_prior=GammaPrior(2.0, 2.0),),outputscale_prior=GammaPrior(2.0, 0.15), )
            )
        )
    model = ModelListGP(*models)
    mll = SumMarginalLogLikelihood(model.likelihood, model)
    return mll, model

def hypervolume_from_posterior_mean_gp(
    model,
    X: torch.Tensor,                      # (n, d) base features (no task column)
    ncons,
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
    #print("Pulling the Posterior")
    post = model.posterior(X)
    Y_mean = post.mean
    if ncons>0:
        ind_feasible = (Y_mean[..., -ncons :] >= 0).all(dim=-1)
        Y_mean[~ind_feasible.squeeze(), -ncons :] = -1e12  # Penalize infeasible points
        Y_mean = Y_mean[...,:-ncons]

    
    
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

#Get HV helper
def get_current_value(
    model,
    ref_point,
    bounds,
    mcobjective=None
):
    """Helper to get the hypervolume of the current hypervolume
    maximizing set.
    """
    curr_val_acqf = _get_hv_value_function(
        model=model,
        ref_point=ref_point,
        use_posterior_mean=True,
        objective=mcobjective
    )
    
    _, current_value = optimize_acqf(
        acq_function=curr_val_acqf,
        bounds=bounds,
        q=10, #From their implementation
        num_restarts=20,
        raw_samples=1024,
        return_best_only=True,
        options={"batch_limit": 5},
    )
    return current_value
#optimize Hypervolume Knowledge Gradient decoupled
def optimize_HVKG_and_get_obs_decoupled(model,q,problem,cost_model,standard_bounds,objective_indices,nobj,ncons,train_x):
    """Utility to initialize and optimize HVKG."""
    cost_aware_utility = InverseCostWeightedUtility(cost_model=cost_model)

    if ncons > 0:
        #Setting Constrained MC Objective
        identity_objective=IdentityMCMultiOutputObjective(outcomes=list(range(nobj)))
        mc_objective=FeasibilityWeightedMCMultiOutputObjective(model=model,X_baseline= train_x,constraint_idcs=list(range(-ncons, 0)), objective=identity_objective)

        current_value = get_current_value(
            model=model,
            ref_point=problem.ref_point,
            bounds=standard_bounds,
            mcobjective=mc_objective
        )

        acq_func = qHypervolumeKnowledgeGradient(
            model=model,
            ref_point=problem.ref_point,  # use known reference point from the problem for stability
            num_fantasies=8, #From their implementation
            num_pareto=10, #From their implementation
            objective=mc_objective,
            current_value=current_value,
            cost_aware_utility=cost_aware_utility,
        )

    else:
        current_value = get_current_value(
            model=model,
            ref_point=problem.ref_point,
            bounds=standard_bounds,
        )
        acq_func = qHypervolumeKnowledgeGradient(
            model=model,
            ref_point=problem.ref_point,  # use known reference point from the problem for stability
            num_fantasies=8, #From their implementation
            num_pareto=10, #From their implementation
            current_value=current_value,
            cost_aware_utility=cost_aware_utility,
        )

    # optimize acquisition functions and get new observations
    objective_vals = []
    objective_candidates = []
    #print("objective_indices",objective_indices)
    for objective_idx in range(nobj+ncons): #Generates an x location and a value for each objective
        # set evaluation index to only condition on one objective
        # this could be multiple objectives
        print("Objective ",objective_idx)
        X_evaluation_mask = torch.zeros(
            q,
            nobj+ncons,
            dtype=torch.bool,
            device=standard_bounds.device,
        )
        X_evaluation_mask[:, objective_idx] = 1 #Setting the evaluation of only the selected objective (1=True)
        #print("X_evaluation_mask",X_evaluation_mask)
        acq_func.X_evaluation_mask = X_evaluation_mask
        
        candidates, vals = optimize_acqf(
            acq_function=acq_func,
            num_restarts=1, #From their implementation
            raw_samples=512,#From their implementation
            bounds=standard_bounds,
            q=q,
            sequential=False,
            #sequential=True,
            options={"batch_limit": 5},
        )
        print("candidates:",candidates,"vals",vals)
        objective_vals.append(vals.view(-1))
        objective_candidates.append(candidates)
    best_objective_index = torch.cat(objective_vals, dim=-1).argmax().item()#Picking the objective as max vals
    eval_objective_indices = [best_objective_index] #Choosing index to evaluate objective at
    candidates = objective_candidates[best_objective_index] #choosing the corresponding candidate point, x, at the best vals
    vals = objective_vals[best_objective_index]
    # observe new values
    #new_x_norm=candidates.clone()
    new_x = unnormalize(candidates.detach(), bounds=problem.bounds)
    new_obj = problem(new_x)
    #print("new_obj in HVKG",new_obj)
    #print("new_obj in HVKG shape",new_obj.shape)
    if ncons>0:
        new_con = problem._evaluate_slack_true(new_x) #Getting Constraint
        #print(new_con)
        #print(new_con.shape)
        new_obj = torch.column_stack([new_obj,new_con])
        #print("new_obj w/ cons in HVKG",new_obj)
    new_obj = new_obj[..., eval_objective_indices]
    return new_x, new_obj, eval_objective_indices

# qNEHVI
def optimize_qnehvi_and_get_observation(model, train_x, sampler, q, problem,standard_bounds,ncons,nobj ):
    """Optimizes the qNEHVI acquisition function, and returns a new candidate and observation."""
    # partition non-dominated space into disjoint rectangles
    
    if ncons>0:
        #print(f"ncons: {ncons}")
        acq_func = qLogNoisyExpectedHypervolumeImprovement(
            model=model,
            ref_point=problem.ref_point.tolist(),  # use known reference point
            X_baseline=normalize(train_x, problem.bounds),
            prune_baseline=True,  # prune baseline points that have estimated zero probability of being Pareto optimal
            sampler=sampler,
            objective=IdentityMCMultiOutputObjective(outcomes=list(range(nobj))),
            constraints=[lambda Z, i=i: -Z[..., i] for i in range(-ncons, 0)], # qNEHVI expects negative constraints to be feasible, but we consider positive constraints to be feasible, so negating the constraint lambda functions
        )
    else:
        acq_func = qLogNoisyExpectedHypervolumeImprovement(
            model=model,
            ref_point=problem.ref_point.tolist(),  # use known reference point
            X_baseline=normalize(train_x, problem.bounds),
            prune_baseline=True,  # prune baseline points that have estimated zero probability of being Pareto optimal
            sampler=sampler,
        )

    # optimize
    candidates, _ = optimize_acqf(
        acq_function=acq_func,
        bounds=standard_bounds,
        q=q,
        num_restarts=10,
        raw_samples=512,  # used for intialization heuristic
        options={"batch_limit": 5, "maxiter": 200},
        sequential=True,
    )
    # observe new values
    new_x = unnormalize(candidates.detach(), bounds=problem.bounds)
    new_obj_true = problem(new_x)
    return new_x, new_obj_true

#Generate sobol Data
def generate_sobol_data(n,problem):
    # generate training data
    train_x = draw_sobol_samples(bounds=problem.bounds, n=n, q=1).squeeze(1)
    train_obj_true = problem(train_x)
    return train_x, train_obj_true