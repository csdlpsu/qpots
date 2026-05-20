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

###PESMO Imports:


from math import pi
from typing import Callable, Sequence


from torch import Tensor

from botorch.acquisition.multi_objective.predictive_entropy_search import (
    qMultiObjectivePredictiveEntropySearch,
    _augment_factors_with_cached_factors,
    _compute_log_determinant,
    _initialize_predictive_matrices,
    _safe_update_omega,
    _update_damping,
    _update_marginals,
)
from botorch.acquisition.multi_objective.max_value_entropy_search import (
    qLowerBoundMultiObjectiveMaxValueEntropySearch,
)
from botorch.acquisition.multi_objective.joint_entropy_search import (
    qLowerBoundMultiObjectiveJointEntropySearch,
    _compute_entropy_monte_carlo,
    _compute_entropy_noiseless,
    _compute_entropy_upper_bound,
)
from botorch.acquisition.multi_objective.utils import (
    compute_sample_box_decomposition,
    sample_optimal_points,
)

from botorch.models.utils import check_no_nans

from botorch.posteriors.gpytorch import GPyTorchPosterior
from botorch.utils.transforms import (
    average_over_ensemble_models,
    concatenate_pending_points,
    t_batch_mode_transform,
)
from botorch.acquisition.multi_objective.utils import (
    compute_sample_box_decomposition,
    random_search_optimizer,
    sample_optimal_points,
)

#initializing model
def initialize_model(train_x_list, train_obj_list, bounds):
    """
    Used for HVKG, qNEHVI, Sobol models

    Initialize a multi-output Gaussian Process model (ModelListGP) for
    objectives and (optionally) constraints using independent SingleTaskGPs.

    Each entry in `train_obj_list` corresponds to one GP model. Inputs are
    normalized to [0, 1]^d using the provided bounds.

    Parameters
    ----------
    train_x_list : list of torch.Tensor
        List of input tensors, one per objective or constraint. Each tensor has
        shape ``n_i x d``, where ``n_i`` is the number of training points for
        output ``i`` and ``d`` is the input dimension. Different outputs may
        use different input sets.
        Note: This allows different datasets per output (i.e., not necessarily shared X).

    train_obj_list : list of torch.Tensor
        List of output tensors (objectives and/or constraints), one per model.
        Each tensor has shape (n_i, 1).

        Objectives and constraints should already be combined into this list.
        Constraints are modeled the same way as objectives here.

    bounds : torch.Tensor
        Tensor of shape ``2 x d`` specifying lower and upper bounds for each
        input dimension. ``bounds[0]`` contains lower bounds and ``bounds[1]``
        contains upper bounds.

    Returns
    -------
    mll : SumMarginalLogLikelihood
        Marginal log likelihood object used for training the ModelListGP.

    model : ModelListGP
        A container of independent SingleTaskGP models, one per objective/constraint.

    Assumptions
    ----------- 
    - `train_x_list` and `train_obj_list` have the same length.
    - Each (train_x_list[i], train_obj_list[i]) pair is aligned.
    - Outputs are already properly shaped (n_i, 1).

    Potential Extensions
    --------------------
    - Use a MultiTaskGP instead of ModelListGP to model correlations between outputs.
    - Learn noise instead of fixing it (remove `train_Yvar`).
    - Add kernel priors or ARD lengthscales for better performance in higher dimensions.
    """
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
    """
    Compute the hypervolume of the posterior mean Pareto front from a GP model.

    This function evaluates the GP posterior mean at a set of input points,
    filters infeasible solutions (if constraints are present), extracts the
    non-dominated subset, and computes the hypervolume indicator.

    Parameters
    ----------
    model : ModelListGP or compatible GP model
        A trained BoTorch model. Must support ``posterior(X)`` and return a
        posterior with ``mean`` of shape ``n x m``, where ``n`` is the number
        of input points and ``m`` is the number of outputs.

    X : torch.Tensor
        Candidate design points with shape ``n x d``.
        These are the design points at which the posterior mean is evaluated.

    ncons : int
        Number of constraint outputs included in the model.
        Assumes that the last `ncons` outputs correspond to constraints.

        A point is feasible if all constraint values are nonnegative.
        Infeasible points are penalized and excluded from Pareto computation.

    ref_point : array-like or torch.Tensor
        Reference point for hypervolume computation.
        Must have shape (m_obj,), where m_obj = number of objectives
        (i.e., total outputs minus constraints).

    maximize : bool
        Whether the objectives are being maximized. If ``False``, objectives
        and the reference point are negated to convert the calculation into a
        maximization problem.

    Returns
    -------
    hv : torch.Tensor (scalar)
        The computed hypervolume of the non-dominated posterior mean set.

    Notes
    -----
    This uses the posterior mean only, not posterior samples. It assumes
    constraints are ordered as the final ``ncons`` outputs. If no nondominated
    points are available, it returns a scalar zero tensor.
    """
    model.eval()

    # Move to model device/dtype
    p = next(model.parameters())
    device, dtype = p.device, p.dtype
    X = X.to(device=device, dtype=dtype)
    ref_point = torch.as_tensor(ref_point, device=device, dtype=dtype).view(-1)
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
    """
    Generate an initial Sobol design and evaluate the optimization problem.

    Parameters
    ----------
    n : int
        Number of Sobol points to draw.
    problem : callable
        BoTorch-style test problem with a ``bounds`` attribute and a callable
        interface that evaluates inputs in the original design space.

    Returns
    -------
    tuple[torch.Tensor, torch.Tensor]
        ``(train_x, train_obj_true)`` where ``train_x`` has shape ``n x d`` and
        ``train_obj_true`` contains the corresponding objective values.
    """
    # generate training data
    train_x = draw_sobol_samples(bounds=problem.bounds, n=n, q=1).squeeze(1)
    train_obj_true = problem(train_x)
    return train_x, train_obj_true


# -------------------- PESMO Decoupled Utils --------------------

##Helpers
def build_pareto_state(
    model,
    bounds: Tensor,
    *,
    num_pareto_samples: int = 8,
    num_pareto_points: int = 8,
    maximize: bool = True,
) -> tuple[Tensor, Tensor, Tensor]:
    """Sample Pareto sets/fronts and compute hypercell bounds."""
    optimizer_kwargs = {
        "pop_size": 4000,
        "max_tries": 10,
        }   
    pareto_sets, pareto_fronts = sample_optimal_points(
        model=model,
        bounds=bounds,
        num_samples=num_pareto_samples,
        num_points=num_pareto_points,
        maximize=maximize,
        optimizer=random_search_optimizer, #addition 4/19
        optimizer_kwargs= optimizer_kwargs #{"pop_size": 4096}  # try 4k–10k
    )
    hypercell_bounds = compute_sample_box_decomposition(
        pareto_fronts=pareto_fronts,
        maximize=maximize,
    )
    return pareto_sets, pareto_fronts, hypercell_bounds


def choose_competitive_decoupled_candidate(
    make_acqf: Callable[[Tensor], object],
    bounds: Tensor,
    options: dict,
    num_outputs: int,
    *,
    objective_costs: Sequence[float] | Tensor | None = None,
    num_restarts: int = 10,
    raw_samples: int = 512,
) -> dict[str, Tensor | int]:
    """Optimize one acquisition per outcome and pick the best (x, outcome) pair.

    `make_acqf(mask)` must return an acquisition function that accepts a
    one-hot `1 x M` evaluation mask.
    """
    if objective_costs is None:
        objective_costs_t = torch.ones(num_outputs, dtype=bounds.dtype, device=bounds.device)
    else:
        objective_costs_t = torch.as_tensor(
            objective_costs, dtype=bounds.dtype, device=bounds.device
        )
    best: dict[str, Tensor | int] | None = None

    for m in range(num_outputs):
        mask = torch.zeros(1, num_outputs, dtype=torch.bool, device=bounds.device)
        mask[0, m] = True
        acqf = make_acqf(mask)

        X_m, acq_val_m = optimize_acqf(
            acq_function=acqf,
            bounds=bounds,
            q=1,
            num_restarts=num_restarts,
            raw_samples=raw_samples,
            options=options,
            
        )
        score_m = acq_val_m.squeeze() / objective_costs_t[m]
        if best is None or bool(score_m > best["score"]):
            best = {
                "X": X_m.detach(),
                "objective_idx": int(m),
                "acq_value": acq_val_m.detach(),
                "score": score_m.detach(),
            }

    if best is None:
        raise RuntimeError("Failed to select a candidate.")
    return best

def _validate_mask(mask: Tensor | None, num_outputs: int) -> None:
    """
    Validate a decoupled-evaluation mask used by competitive acquisition rules.

    Parameters
    ----------
    mask : torch.Tensor or None
        Boolean tensor of shape ``q x num_outputs``. Each row marks which model
        outputs would be observed for the associated candidate. ``None`` means
        all outputs are observed.
    num_outputs : int
        Total number of model outputs, including objectives and constraints.

    Raises
    ------
    ValueError
        If ``mask`` has the wrong shape or contains a row with no selected
        output.
    """
    if mask is None:
        return
    if mask.ndim != 2 or mask.shape[-1] != num_outputs:
        raise ValueError(
            f"Expected X_evaluation_mask with shape q x {num_outputs}, got {tuple(mask.shape)}."
        )
    if not mask.any(dim=-1).all():
        raise ValueError("Each row of X_evaluation_mask must select at least one output.")


class _CompetitiveDecouplingMixin:
    """Minimal mask plumbing for competitive decoupling.

    This intentionally supports only q=1 when a mask is provided. That is the
    standard 'pick one x and one outcome to evaluate' setting used in the
    decoupled HVKG tutorial.
    """

    X_evaluation_mask: Tensor | None

    def _set_X_evaluation_mask(self, X_evaluation_mask: Tensor | None) -> None:
        """
        Set and validate the output-observation mask for the next acquisition call.

        Parameters
        ----------
        X_evaluation_mask : torch.Tensor or None
            Boolean ``q x M`` mask selecting the outputs to observe. Passing
            ``None`` restores coupled evaluation of all outputs.
        """
        _validate_mask(X_evaluation_mask, self.model.num_outputs)
        self.X_evaluation_mask = X_evaluation_mask

    def _selected_output_indices(self, q: int) -> list[int]:
        """
        Return output indices selected by the current decoupling mask.

        Parameters
        ----------
        q : int
            Acquisition batch size. Masked competitive decoupling currently
            supports only ``q=1`` because it compares one candidate-output pair
            at a time.

        Returns
        -------
        list[int]
            Selected output indices. If no mask is set, all outputs are
            returned.
        """
        if self.X_evaluation_mask is None:
            return list(range(self.model.num_outputs))
        if q != 1:
            raise NotImplementedError(
                "This minimal example supports competitive decoupling only (q=1)."
            )
        return self.X_evaluation_mask[0].nonzero(as_tuple=False).view(-1).tolist()

    def _selected_output_mask(self, q: int, *, dtype: torch.dtype, device: torch.device) -> Tensor:
        """
        Build a numeric one-dimensional mask for selected model outputs.

        Parameters
        ----------
        q : int
            Acquisition batch size forwarded to ``_selected_output_indices``.
        dtype : torch.dtype
            Desired dtype for the returned mask.
        device : torch.device
            Desired device for the returned mask.

        Returns
        -------
        torch.Tensor
            Tensor of length ``model.num_outputs`` with ones at selected
            outputs and zeros elsewhere.
        """
        idx = self._selected_output_indices(q)
        mask = torch.zeros(self.model.num_outputs, dtype=dtype, device=device)
        mask[idx] = 1.0
        return mask


##PESMO decoupled
class qDecoupledPESMO(_CompetitiveDecouplingMixin, qMultiObjectivePredictiveEntropySearch):
    """
    Predictive entropy search acquisition with competitive decoupled outputs.

    This subclass adapts BoTorch's
    ``qMultiObjectivePredictiveEntropySearch`` so that a single output can be
    scored at a candidate point. It is useful for decoupled multiobjective
    experiments where evaluating every objective or constraint at every point
    is unnecessary or has different cost.

    Notes
    -----
    The implementation is intentionally narrow: when an evaluation mask is set,
    it supports the common competitive setting ``q=1`` and compares one
    candidate-output pair at a time.
    """

    def __init__(
        self,
        *args,
        X_evaluation_mask: Tensor | None = None,
        **kwargs,
    ) -> None:
        """
        Initialize the decoupled PESMO acquisition function.

        Parameters
        ----------
        *args
            Positional arguments forwarded to BoTorch's
            ``qMultiObjectivePredictiveEntropySearch``.
        X_evaluation_mask : torch.Tensor or None, optional
            Optional boolean mask of shape ``q x M`` selecting which outputs
            are observed for each candidate. ``None`` means all outputs are
            observed.
        **kwargs
            Keyword arguments forwarded to the BoTorch base acquisition.
        """
        super().__init__(*args, **kwargs)
        self._set_X_evaluation_mask(X_evaluation_mask)

    def _masked_logdet(self, cov: Tensor, q: int) -> Tensor:
        """
        Compute the PES log-determinant term for selected outputs only.

        Parameters
        ----------
        cov : torch.Tensor
            Predictive covariance tensor produced by BoTorch's PES utilities.
        q : int
            Batch size represented in ``cov``.

        Returns
        -------
        torch.Tensor
            Log-determinant contribution averaged over Pareto samples and
            restricted to the selected output mask when one is active.
        """
        if self.X_evaluation_mask is None:
            return _compute_log_determinant(cov=cov, q=q)
        log_det_cov = torch.logdet(cov[..., 0:q, 0:q])
        check_no_nans(log_det_cov)
        weights = self._selected_output_mask(
            q=q, dtype=log_det_cov.dtype, device=log_det_cov.device
        )
        return (log_det_cov * weights).sum(dim=-1).mean(dim=-1)

    def _compute_information_gain(self, X: Tensor) -> Tensor:
        """
        Evaluate the approximate PESMO information gain at candidate points.

        Parameters
        ----------
        X : torch.Tensor
            Candidate tensor with BoTorch acquisition shape
            ``batch_shape x q x d``.

        Returns
        -------
        torch.Tensor
            Acquisition value for each batch element. Larger values indicate
            higher expected information gain about the Pareto-optimal set.
        """
        tkwargs = {"dtype": X.dtype, "device": X.device}
        batch_shape = X.shape[0:-2]
        q = X.shape[-2]
        M = self.model.num_outputs

        if M > 1 or isinstance(self.model, ModelListGP):
            N = len(self.model.train_inputs[0][0])
        else:
            N = len(self.model.train_inputs[0])
        P = self.pareto_sets.shape[-2]
        num_pareto_samples = self.num_pareto_samples

        new_shape = batch_shape + torch.Size([num_pareto_samples]) + X.shape[-2:]
        expanded_X = X.unsqueeze(-3).expand(new_shape)
        expanded_ps = self.pareto_sets.expand(X.shape[0:-2] + self.pareto_sets.shape)
        aug_X = torch.cat([expanded_X, expanded_ps], dim=-2)

        (
            pred_nat_mean,
            pred_nat_cov,
            pred_mean,
            pred_cov,
        ) = _initialize_predictive_matrices(
            X=aug_X,
            model=self.model,
            observation_noise=True,
            jitter=self.test_jitter,
            natural=True,
        )
        pred_f_mean = pred_mean[..., 0:M, :]
        pred_f_nat_mean = pred_nat_mean[..., 0:M, :]
        pred_f_cov = pred_cov[..., 0:M, :, :]
        pred_f_nat_cov = pred_nat_cov[..., 0:M, :, :]

        (_, _, _, pred_cov_noise) = _initialize_predictive_matrices(
            X=aug_X,
            model=self.model,
            observation_noise=True,
            jitter=self.test_jitter,
            natural=False,
        )
        pred_f_cov_noise = pred_cov_noise[..., 0:M, :, :]
        observation_noise = pred_f_cov_noise - pred_f_cov

        omega_f_nat_mean = torch.zeros(
            batch_shape + torch.Size([num_pareto_samples, M, q + P, P, 2]), **tkwargs
        )
        omega_f_nat_cov = torch.zeros(
            batch_shape + torch.Size([num_pareto_samples, M, q + P, P, 2, 2]), **tkwargs
        )

        omega_f_nat_mean, omega_f_nat_cov = _safe_update_omega(
            mean_f=pred_f_mean,
            cov_f=pred_f_cov,
            omega_f_nat_mean=omega_f_nat_mean,
            omega_f_nat_cov=omega_f_nat_cov,
            N=q,
            P=P,
            M=M,
            maximize=self.maximize,
            jitter=self.test_jitter,
        )
        omega_f_nat_mean, omega_f_nat_cov = _augment_factors_with_cached_factors(
            q=q,
            N=N,
            omega_f_nat_mean=omega_f_nat_mean,
            cached_omega_f_nat_mean=self._omega_f_nat_mean,
            omega_f_nat_cov=omega_f_nat_cov,
            cached_omega_f_nat_cov=self._omega_f_nat_cov,
        )
        nat_mean_f, nat_cov_f = _update_marginals(
            pred_f_nat_mean=pred_f_nat_mean,
            pred_f_nat_cov=pred_f_nat_cov,
            omega_f_nat_mean=omega_f_nat_mean,
            omega_f_nat_cov=omega_f_nat_cov,
            N=q,
            P=P,
        )
        damping = torch.ones(batch_shape + torch.Size([num_pareto_samples, M]), **tkwargs)
        _, cholesky_nat_cov_f_new = _update_damping(
            nat_cov=pred_f_nat_cov,
            nat_cov_new=nat_cov_f,
            damping_factor=damping,
            jitter=self.test_jitter,
        )
        cov_f_new = torch.cholesky_inverse(cholesky_nat_cov_f_new)
        check_no_nans(cov_f_new)

        log_det_pred_f_cov_noise = self._masked_logdet(cov=pred_f_cov_noise, q=q)
        log_det_cov_f = self._masked_logdet(cov=cov_f_new + observation_noise, q=q)
        q_pes_f = log_det_pred_f_cov_noise - log_det_cov_f
        check_no_nans(q_pes_f)
        return 0.5 * q_pes_f

    @concatenate_pending_points
    @t_batch_mode_transform()
    @average_over_ensemble_models
    def forward(self, X: Tensor) -> Tensor:
        """
        Evaluate the acquisition function at ``X``.

        Parameters
        ----------
        X : torch.Tensor
            Candidate set in BoTorch acquisition format.

        Returns
        -------
        torch.Tensor
            Information-gain acquisition value for each batch element.
        """
        return self._compute_information_gain(X)
    
