import torch
from torch import Tensor
from botorch.utils.multi_objective.box_decompositions import FastNondominatedPartitioning
from botorch.utils.multi_objective.hypervolume import Hypervolume
from botorch.utils.multi_objective.pareto import is_non_dominated
from botorch.utils.transforms import standardize
from sklearn.neighbors import KernelDensity
from scipy.spatial.distance import cdist
import argparse
from typing import Tuple
from qpots.model_object import ModelObject
from qpots.function import Function
import numpy as np

from itertools import combinations
from typing import Iterable, Optional, Sequence, Tuple, Union, Dict, Any


from botorch.utils.multi_objective.pareto import _is_non_dominated_loop
from gpytorch import settings

from pymoo.core.problem import Problem
from pymoo.algorithms.moo.nsga2 import NSGA2
from pymoo.optimize import minimize
# from pymoo.util.termination.max_gen import MaximumGenerationTermination
from botorch.utils.transforms import normalize, unnormalize

from botorch.models import MultiTaskGP
from botorch.models.transforms.outcome import Standardize
from botorch.fit import fit_gpytorch_mll
from gpytorch.mlls import ExactMarginalLogLikelihood
from botorch.sampling import SobolQMCNormalSampler

def unstandardize_ignore_nan(Y: Tensor, train_y: Tensor, correction: int = 1) -> Tensor:
    """
    Reverse the standardization of output `Y` using the mean and standard deviation 
    computed from the training data, works when NaN values are in the train_y tensor.

    Parameters
    ----------
    Y : torch.Tensor
        The standardized output tensor.
    train_y : torch.Tensor
        The training output data used to compute the mean and standard deviation.

    Returns
    -------
    torch.Tensor
        The unstandardized output tensor.
    """
    mean = torch.nanmean(train_y, dim=0)
    std = torch.from_numpy(np.nanstd(train_y.cpu().numpy(), axis=0, ddof=1)).to(train_y)
    return Y * std + mean

def get_model_identified_hv_maximizing_set(
    model,
    problem,
    ref_point,
    train_y, #adding train_y to pass for unsandardizing
    multiplier=1,
    max_gen=100,
    ncons=0,
    

    
):
    """
    Construct a Pymoo optimization problem that uses a GP posterior sample
    to identify a hypervolume-maximizing set via NSGA-II.

    This function defines an inner Pymoo `Problem` class whose objective
    evaluations are derived from samples of the GP posterior. It is intended
    to be used with evolutionary algorithms (e.g., NSGA-II) to approximate
    the Pareto set that maximizes hypervolume under the learned model.

    Parameters
    ----------
    model : ModelListGP or compatible GP model
        A trained BoTorch model. Must support `.posterior(X)` and return
        a posterior over all outputs (objectives + constraints).

    problem : object
        A problem definition containing:
            - problem.dim  : int, input dimension (d)
            - problem.nobj : int, number of objectives (m)

    ref_point : torch.Tensor
        Reference point for hypervolume computation.
        Used externally (not directly inside `_evaluate`), but defines
        dtype/device and objective dimensionality consistency.

    train_y : torch.Tensor
        Training outputs used to standardize the GP model.
        Required here to *unstandardize* posterior samples so that
        optimization is performed in the original objective space.

    multiplier : int, optional (default=1)
        Scales the population size:
            population_size = 100 * dim * multiplier

    max_gen : int, optional (default=100)
        Maximum number of generations for the evolutionary algorithm.
        (Used externally when running NSGA-II.)

    ncons : int, optional (default=0)
        Number of constraint outputs in the model.
        Assumes constraints are the last `ncons` outputs.

        Constraint convention:
        - Feasible if constraint values >= 0
        - Infeasible points are penalized in objective space

    Returns
    -------
    PosteriorMeanPymooProblem : pymoo.core.problem.Problem
        A configured Pymoo problem instance that can be passed to
        NSGA-II or other evolutionary algorithms.

    Internal Class: PosteriorMeanPymooProblem
    ----------------------------------------
    Defines the optimization problem evaluated by NSGA-II.

    Key Features
    ------------
    - Decision variables:
        x ∈ [0, 1]^d (assumes normalized design space)

    - Objective evaluation:
        1. Convert numpy input to torch tensor
        2. Evaluate GP posterior at X
        3. Draw a single sample using Sobol QMC sampler
        4. Unstandardize outputs using `train_y`
        5. Apply constraint handling (if any)
        6. Return NEGATED objectives (since Pymoo minimizes)

    - Stochastic evaluation:
        Uses a fixed-seed SobolQMCNormalSampler for reproducibility.

    Constraint Handling
    -------------------
    - Constraints are assumed to be the last `ncons` outputs.
    - Feasibility condition: all constraints >= 0
    - Infeasible points are penalized by assigning a large negative
      value (-1e12) to their objective values before minimization.

    Important Conventions
    ---------------------
    - Pymoo performs minimization → objectives are negated (`out["F"] = -f`)
    - Model operates in standardized space → outputs are unstandardized
    - Sampling (not mean) is used → enables exploration of model uncertainty

    Notes
    -----
    - Although named "PosteriorMean...", this actually uses posterior samples,
      not the mean.
    - Hypervolume is not computed directly here; instead, NSGA-II approximates
      the Pareto front, which can later be evaluated using HV.
    - The Sobol sampler ensures deterministic behavior across evaluations
      when using the same seed.

    """
    tkwargs = {
        "dtype": ref_point.dtype,
        "device": ref_point.device,
    }
    dim = problem.dim
    population_size=100*dim*multiplier
    seed=2429+multiplier

    class PosteriorMeanPymooProblem(Problem):
        def __init__(self):
            super().__init__(
                n_var=dim,
                n_obj=problem.nobj,
                #n_ieq_constr=ncons, #2/9 Newline
                type_var=np.double,
            )
            self.xl = np.zeros(dim)
            self.xu = np.ones(dim)
            self.sampler = SobolQMCNormalSampler(
                sample_shape=torch.Size([1]), 
                seed=seed
            ) # Sampler for consistency when calling _evaluate()

        def _evaluate(self, x, out, *args, **kwargs):
            X = torch.from_numpy(x).to(**tkwargs)

            #wihout Sampler
            #y_std = model.posterior(X).sample().reshape(-1,problem.nobj+ncons)

            #With Sampler (Better)
            #"""
            with torch.no_grad():
                posterior = model.posterior(X)
                y_std = self.sampler(posterior)  # shape: [1, N, m]
                y_std = y_std.squeeze(0)
                #print("y_std\n",y_std[:5,:])
            #"""
            
            #unstandardizing posterior based on the sent train_y (should be the SAME as the one you use to train your GP)
            y = unstandardize_ignore_nan(y_std, train_y.to(**tkwargs))   

            ## Constraint Handling
            if ncons > 0:

                #penalizing constraint violation
                ind_feasible = (y[..., -ncons :] >= 0).all(dim=-1)
                y[~ind_feasible.squeeze(), : problem.nobj] = -1e12  # Penalize infeasible points
                f = y[..., : problem.nobj]

            else:
                f=y
                
            out["F"] = -f.cpu().numpy()

    pymoo_problem = PosteriorMeanPymooProblem()
    algorithm = NSGA2(
        pop_size=population_size,
        eliminate_duplicates=True,
    )
    res = minimize(
        pymoo_problem,
        algorithm,
        ("n_gen", max_gen),
        pop_size=population_size,
        seed=seed,
        verbose=False,
    )
    X = torch.tensor(
        res.X,
        **tkwargs,
    )
    X = unnormalize(X, problem.get_bounds().to(tkwargs["device"]))
    Y = torch.tensor(-res.F, **tkwargs) #problem(X)
    # compute HV
    partitioning = FastNondominatedPartitioning(ref_point=ref_point, Y=Y)
    return res, partitioning.compute_hypervolume().item()

def _unravel_index(flat_idx: int, shape):
    """Like numpy.unravel_index(flat_idx, shape), but without numpy."""
    idx = []
    for s in reversed(shape):
        flat_idx, r = divmod(flat_idx, s)
        idx.append(r)
    return tuple(reversed(idx))

def qmaximin(train_X, X, *, q: int = 1, return_index: bool = False, return_distance: bool = False):
    """
    Greedy *sequential* maximin (a.k.a. farthest-point sampling):
        x_{n+1} = argmax_{x in X}   min_{y in train_X} ||x - y||
        x_{n+2} = argmax_{x in X}   min_{y in train_X ∪ {x_{n+1}}} ||x - y||
        ...
        x_{n+q} = ...

    train_X: tensor/array of shape (..., d)
    X:       tensor/array of shape (..., d)   (same last dimension d)
    q:       number of points to pick (unique picks, up to len(X))

    Returns:
      - by default: best_points of shape (k, d) where k = min(q, num_candidates)
      - optionally indices: list[tuple] of length k (indices into X.shape[:-1])
      - optionally distances: tensor of shape (k,) giving the maximin distance
        at the time each point was selected.
    """
    train_X = torch.as_tensor(train_X)
    X = torch.as_tensor(X)

    if q < 1:
        raise ValueError("q must be >= 1")
    if train_X.shape[-1] != X.shape[-1]:
        raise ValueError(f"Last dimension mismatch: train_X has {train_X.shape[-1]}, X has {X.shape[-1]}")
    if X.numel() == 0:
        raise ValueError("X is empty (no candidate points).")

    # Put everything on X's device for distance calculations
    if train_X.device != X.device:
        train_X = train_X.to(X.device)

    d = X.shape[-1]
    X_flat = X.reshape(-1, d)
    train_flat = train_X.reshape(-1, d)

    n_cand = X_flat.shape[0]
    k = min(q, n_cand)

    # Work in float for distance computations
    Xf = X_flat.float()
    x_norm2 = (Xf * Xf).sum(dim=1)  # precompute ||x||^2 for all candidates

    # Initial min distance to the existing training set
    if train_flat.shape[0] == 0:
        # No observed points yet: all candidates start as "inf" far away;
        # the first pick will be arbitrary (the first argmax).
        min_dist = torch.full((n_cand,), float("inf"), device=X.device, dtype=Xf.dtype)
    else:
        trainf = train_flat.float()
        min_dist = torch.cdist(Xf, trainf).min(dim=1).values  # (n_cand,)

    selected = torch.empty((k,), dtype=torch.long, device=X.device)
    selected_d = torch.empty((k,), dtype=min_dist.dtype, device=X.device)

    for t in range(k):
        best = int(min_dist.argmax().item())
        selected[t] = best
        selected_d[t] = min_dist[best]

        # Prevent re-selecting the same candidate
        min_dist[best] = -float("inf")

        # Update: min_dist[x] = min(min_dist[x], ||x - x_best||)
        if t < k - 1:
            v = Xf[best]            # (d,)
            v_norm2 = x_norm2[best] # scalar
            # ||x - v||^2 = ||x||^2 + ||v||^2 - 2 x·v
            dist2 = x_norm2 + v_norm2 - 2.0 * (Xf @ v)
            dist2 = dist2.clamp_min_(0.0)
            dist_new = torch.sqrt(dist2)
            min_dist = torch.minimum(min_dist, dist_new)

    best_points = X_flat[selected]  # return in original dtype

    out = (best_points,)
    if return_index:
        out += ([ _unravel_index(int(i), X.shape[:-1]) for i in selected.tolist() ],)
    if return_distance:
        out += (selected_d,)

    return out[0] if len(out) == 1 else out

#New function for jitter
def cholesky_with_jitter(R, initial_jitter=1e-6, max_jitter=1e-2, factor=10.0):
    """
    Attempt Cholesky decomposition of R.
    If it fails, increase diagonal jitter iteratively until success.
    
    Args:
        R: (n, n) correlation/covariance matrix
        initial_jitter: starting jitter on diagonal
        max_jitter: maximum allowed jitter
        factor: factor to increase jitter each attempt
    
    Returns:
        L: Cholesky factor
        final_jitter: the jitter that worked
    """
    jitter = initial_jitter
    while True:
        try:
            L = torch.linalg.cholesky(R)
            return L
        except torch._C._LinAlgError:
            print("Fail to invert, increasing jitter")
            jitter *= factor
            if jitter > max_jitter:
                raise RuntimeError(f"Cholesky failed even with jitter={jitter:.3e}")

def corr_and_total_correlation(
    cov: torch.Tensor,
    jitter: float = 1e-6,
    eps: float = 1e-12,
):
    """
    Compute task correlation matrix R and total correlation TC from an inter-task covariance matrix.

    Args:
        cov: (..., m, m) symmetric covariance matrix across m tasks (e.g., posterior Cov[f(x)]).
        jitter: diagonal jitter added to R before Cholesky/logdet for numerical stability.
        eps: clamp floor for variances to avoid divide-by-zero.

    Returns:
        R:  (..., m, m) correlation matrix.
        TC: (...) total correlation in nats, TC = -0.5 * logdet(R).
    """
    # Standard deviations from diagonal
    var = torch.diagonal(cov, dim1=-2, dim2=-1).clamp_min(eps)  # (..., m)
    #print("var",var)
    #print("cov",cov)
    std = var.sqrt()

    # Correlation matrix
    R = cov / (std.unsqueeze(-1) * std.unsqueeze(-2))

    # Stabilize and compute logdet via Cholesky: logdet(R) = 2 * sum(log(diag(L)))
    m = R.shape[-1]
    eye = torch.eye(m, device=R.device, dtype=R.dtype).expand(R.shape[:-2] + (m, m))
    
    Rj = R + jitter * eye

    #print("Rj: \n",Rj)
    try: #Runs when Rj is positive definite
        L = torch.linalg.cholesky(Rj)
        logdet = 2.0 * torch.log(torch.diagonal(L, dim1=-2, dim2=-1)).sum(dim=-1)
        TC = -0.5 * logdet
    except RuntimeError: #If not, run coupled evaluation at this location (TC=None)
        print("R was not invertible, performing coupled evaluation")
        TC=None 

    return R, TC

def computeTC(x,mt_model):
    """
    Compute the total correlation (TC) from the posterior covariance
    of a multi-output GP model at a given input.

    This function evaluates the model posterior at a given point,
    extracts the covariance matrix of the joint output distribution,
    and computes the total correlation (a measure of statistical
    dependence between outputs).

    Parameters
    ----------
    x : torch.Tensor
        Input tensor. Can be of shape (..., d) or flat.
        Internally reshaped to (-1, d_eff), where:
            - d_eff = dim - 1
        (assumes last column may correspond to a task index or is excluded)

    mt_model : MultiTaskGP or compatible model
        A trained multi-output GP model that supports `.posterior(X)`
        and returns a multivariate normal distribution with a full
        covariance matrix across outputs.

    Returns
    -------
    tc_ : torch.Tensor
        Scalar tensor representing the total correlation of the
        posterior output distribution at input `x`.

    Notes
    -----
    - The covariance matrix reflects dependencies between outputs
      (e.g., tasks in a MultiTaskGP).
    - Total correlation (TC) is a multivariate generalization of mutual
      information:
          TC = sum of marginal entropies − joint entropy
      It is zero if outputs are independent and positive otherwise.
    - Assumes `corr_and_total_correlation(cov)` returns:
          (correlation_matrix, total_correlation)

    Assumptions
    -----------
    - `mt_model.train_inputs[0]` exists and has shape (..., d_total)
    - The effective feature dimension is `dim - 1`
      (e.g., last column may be a task index)
    - Posterior covariance is fully materialized and small enough
      to fit in memory
    """
    dim=mt_model.train_inputs[0].shape[-1]
    post = mt_model.posterior(x.view(-1,dim-1))
    cov  = post.distribution.covariance_matrix  # 2x2 (materialized)

    _, tc_ = corr_and_total_correlation(cov)
    return tc_
def _stable_logdet(A: np.ndarray, jitter: float = 1e-10, max_tries: int = 8) -> float:
    """
    Numerically stable log(det(A)) for (near) PSD matrices by adding diagonal jitter if needed.
    Raises if it cannot make the matrix numerically positive definite.
    """
    A = np.asarray(A, dtype=float)
    if A.ndim != 2 or A.shape[0] != A.shape[1]:
        raise ValueError(f"A must be square; got shape {A.shape}")

    n = A.shape[0]
    eye = np.eye(n, dtype=float)

    j = float(jitter)
    for _ in range(max_tries):
        sign, ld = np.linalg.slogdet(A + j * eye)
        if sign > 0 and np.isfinite(ld):
            return float(ld)
        j *= 10.0

    raise np.linalg.LinAlgError(
        "Could not compute a positive logdet even after adding jitter. "
        "Covariance may be indefinite or extremely ill-conditioned."
    )


def mutual_information_split_gaussian(
    cov: np.ndarray,
    S: Sequence[int],
    *,
    jitter: float = 1e-10,
    base: float = np.e,
) -> float:
    """
    Compute I(Y_S ; Y_{Sc}) under a multivariate Gaussian with covariance `cov`.

    Parameters
    ----------
    cov : (K, K) array
        Covariance matrix of Y.
    S : sequence of ints
        Indices in the subset S.
    jitter : float
        Diagonal jitter used for stable log-determinants.
    base : float
        Log base. Use base=2.0 for bits, base=np.e for nats.

    Returns
    -------
    mi : float
        Mutual information I(Y_S ; Y_{Sc}) in the chosen log base.
    """
    cov = np.asarray(cov, dtype=float)
    if cov.ndim != 2 or cov.shape[0] != cov.shape[1]:
        raise ValueError(f"cov must be square; got shape {cov.shape}")
    K = cov.shape[0]

    S = sorted(set(int(i) for i in S))
    if any(i < 0 or i >= K for i in S):
        raise ValueError(f"S contains out-of-range indices for K={K}: {S}")

    Sc = [i for i in range(K) if i not in set(S)]
    if len(S) == 0 or len(Sc) == 0:
        return 0.0

    cov_S = cov[np.ix_(S, S)]
    cov_Sc = cov[np.ix_(Sc, Sc)]

    ld_S = _stable_logdet(cov_S, jitter=jitter)
    ld_Sc = _stable_logdet(cov_Sc, jitter=jitter)
    ld_all = _stable_logdet(cov, jitter=jitter)

    mi_nats = 0.5 * (ld_S + ld_Sc - ld_all)

    if base == np.e:
        return float(mi_nats)
    else:
        return float(mi_nats / np.log(base))


def argmax_mi_subset_bruteforce(
    cov_or_samples: np.ndarray,
    *,
    subset_size: Optional[int] = None,
    jitter: float = 1e-10,
    base: float = np.e,
    assume_samples: Optional[bool] = None,
    deduplicate_complements: bool = True,
    return_all_scores: bool = False,
) -> Dict[str, Any]:
    """
    Brute-force search over subsets S to maximize I(Y_S ; Y_{Sc}) (Gaussian assumption).

    You can pass either:
      - cov_or_samples as a (K,K) covariance matrix, OR
      - cov_or_samples as (N,K) samples (rows=samples), from which we estimate covariance.

    Parameters
    ----------
    cov_or_samples : np.ndarray
        (K,K) covariance OR (N,K) samples.
    subset_size : int or None
        If provided, restrict search to subsets with |S| == subset_size.
        If None, searches all non-trivial subsets (excluding empty/full).
    jitter, base : see above
    assume_samples : bool or None
        If None, auto-detect: (K,K) => covariance; otherwise => samples.
    deduplicate_complements : bool
        If subset_size is None, avoid evaluating both S and Sc by restricting to |S| <= floor(K/2).
        This is safe because I(S;Sc) == I(Sc;S).
    return_all_scores : bool
        If True, returns a dict mapping subset tuples -> MI score (can be large: O(2^K)).

    Returns
    -------
    result : dict with keys
        - "S": tuple of indices for the best subset
        - "Sc": tuple of indices for the complement
        - "mi": best MI value
        - "cov": covariance used
        - optionally "scores": dict[(tuple)->float] if return_all_scores=True
    """
    X = np.asarray(cov_or_samples, dtype=float)

    if assume_samples is None:
        assume_samples = not (X.ndim == 2 and X.shape[0] == X.shape[1])

    if assume_samples:
        if X.ndim != 2:
            raise ValueError(f"Samples must be 2D (N,K); got shape {X.shape}")
        # sample covariance (rowvar=False => columns are variables)
        cov = np.cov(X, rowvar=False, bias=False)
    else:
        cov = X

    cov = np.asarray(cov, dtype=float)
    if cov.ndim != 2 or cov.shape[0] != cov.shape[1]:
        raise ValueError(f"Covariance must be (K,K); got shape {cov.shape}")

    K = cov.shape[0]
    idx = list(range(K))

    if subset_size is not None:
        if not (1 <= subset_size <= K - 1):
            raise ValueError(f"subset_size must be in [1, K-1]; got {subset_size} for K={K}")
        sizes = [subset_size]
    else:
        # all non-trivial subsets
        max_size = (K // 2) if deduplicate_complements else (K - 1)
        sizes = list(range(1, max_size + 1))

    best_S: Optional[Tuple[int, ...]] = None
    best_mi = -np.inf
    scores = {} if return_all_scores else None

    for m in sizes:
        for S in combinations(idx, m):
            mi = mutual_information_split_gaussian(cov, S, jitter=jitter, base=base)
            if return_all_scores:
                scores[tuple(S)] = mi
            if mi > best_mi:
                best_mi = mi
                best_S = tuple(S)

    if best_S is None:
        # This only happens if K < 2.
        best_S = tuple()
        best_mi = 0.0

    best_S_set = set(best_S)
    best_Sc = tuple(i for i in idx if i not in best_S_set)

    out: Dict[str, Any] = {"S": best_S, "Sc": best_Sc, "mi": float(best_mi), "cov": cov}
    if return_all_scores:
        out["scores"] = scores
    
    return out

def fit_mtgp(train_X, train_Y,d,n_train,device,dtype):
    """
    Outdated
    """
    mt_model_kind = "MultiTaskGP(task_feature)"

    task_feature = d  # last column
    
    train_X0 = torch.cat(
        [train_X, torch.zeros(n_train, 1, device=device, dtype=dtype)], dim=-1
    )
    train_X1 = torch.cat(
        [train_X, torch.ones(n_train, 1, device=device, dtype=dtype)], dim=-1
    )
    train_X_mt = torch.cat([train_X0, train_X1], dim=0)
    train_Y_mt = torch.cat([train_Y[:, [0]], train_Y[:, [1]]], dim=0)
    
    mt_model = MultiTaskGP(train_X_mt, train_Y_mt, task_feature=task_feature, outcome_transform=Standardize(m=1), rank=1,
    ).to(device=device, dtype=dtype)
    
    mll_mt = ExactMarginalLogLikelihood(mt_model.likelihood, mt_model)
    fit_gpytorch_mll(mll_mt);

    return mt_model

def wide_to_long_mt(
    x: torch.Tensor,          # (q, d) WITHOUT task feature
    y: torch.Tensor,          # (q, K) with NaNs allowed
    task_feature: int,        # index of task feature in the AUGMENTED input (d+1 dims)
) -> tuple[torch.Tensor, torch.Tensor]:
    """
    Convert wide (q x d, q x K) into MultiTaskGP long format:
      X_long: (n_obs, d+1) with a task index inserted at task_feature
      Y_long: (n_obs, 1)

    We drop NaNs in y.
    """
    if x.ndim != 2 or y.ndim != 2:
        raise ValueError(f"Expected x,y to be 2D. Got x.ndim={x.ndim}, y.ndim={y.ndim}.")

    q, d = x.shape
    q2, K = y.shape
    if q2 != q:
        raise ValueError(f"x and y must have same first dim. Got {q} and {q2}.")

    aug_dim = d + 1
    tf = task_feature if task_feature >= 0 else aug_dim + task_feature
    if not (0 <= tf < aug_dim):
        raise ValueError(f"task_feature={task_feature} is invalid for augmented dim {aug_dim}.")

    Xs, Ys = [], []
    for k in range(K):
        obs_mask = torch.isfinite(y[:, k])  # True where not NaN/inf
        if obs_mask.any():
            xk = x[obs_mask]  # (n_k, d)
            tk = torch.full(
                (xk.shape[0], 1),
                float(k),              # integer-valued, stored in float tensor
                device=x.device,
                dtype=x.dtype,
            )
            # Insert task column at position tf
            Xk = torch.cat([xk[..., :tf], tk, xk[..., tf:]], dim=-1)  # (n_k, d+1)
            yk = y[obs_mask, k].unsqueeze(-1)                         # (n_k, 1)

            Xs.append(Xk)
            Ys.append(yk)

    if len(Xs) == 0:
        raise ValueError("No non-NaN observations in y_new; nothing to add.")

    X_long = torch.cat(Xs, dim=0)
    Y_long = torch.cat(Ys, dim=0)
    return X_long, Y_long


def update_mtgp_with_new_data(
    mt_model: MultiTaskGP,
    x_new: torch.Tensor,      # (q, d)
    y_new: torch.Tensor,      # (q, K) with NaNs
    task_feature: int,
    rank: int = 1,
    refit_hyperparams: bool = True,
) -> MultiTaskGP:
    """
    Update an existing MultiTaskGP with new (x_new, y_new) where y_new has NaNs
    for unevaluated objectives.

    Recommended approach:
      1) Convert new data to long format (drop NaNs)
      2) Recover existing training data in *raw* outcome space
      3) Concatenate
      4) Rebuild a new MultiTaskGP so Standardize(m=1) is re-fit on the expanded data
      5) Warm-start hypers from old mt_model (excluding outcome_transform buffers)
      6) Optionally refit hypers
    """
    # Put new tensors on same device/dtype as the model
    p = next(mt_model.parameters())
    device, dtype = p.device, p.dtype
    x_new = x_new.to(device=device, dtype=dtype)
    y_new = y_new.to(device=device, dtype=dtype)

    # 1) wide -> long (drops NaNs)
    X_new_mt, Y_new_mt = wide_to_long_mt(x_new, y_new, task_feature=task_feature)

    # 2) get old training data from model
    X_old_mt = mt_model.train_inputs[0]   # (N_old, d+1)

    Y_old = mt_model.train_targets        # often (N_old,) in gpytorch
    if Y_old.ndim == 1:
        Y_old = Y_old.unsqueeze(-1)       # -> (N_old, 1)

    # If there is an outcome_transform, train_targets are typically in transformed space.
    # Untransform them back to raw space so we can rebuild properly.
    otf = getattr(mt_model, "outcome_transform", None)
    if otf is not None:
        Y_old_raw, _ = otf.untransform(Y_old)
    else:
        Y_old_raw = Y_old

    # 3) concatenate raw training data
    X_upd = torch.cat([X_old_mt, X_new_mt], dim=0)
    Y_upd = torch.cat([Y_old_raw, Y_new_mt], dim=0)
    
    # 4) rebuild model so Standardize gets re-fit on (X_upd, Y_upd)
    # Warm-start hypers from the previous model, but drop outcome_transform buffers.
    old_sd = mt_model.state_dict()
    old_sd = {k: v for k, v in old_sd.items() if not k.startswith("outcome_transform.")}

    mt_model_upd = MultiTaskGP(
        train_X=X_upd,
        train_Y=Y_upd,
        task_feature=task_feature,
        outcome_transform=Standardize(m=1),
        rank=rank,
    ).to(device=device, dtype=dtype)

    mt_model_upd.load_state_dict(old_sd, strict=False)

    # 5) optionally refit hyperparameters
    if refit_hyperparams:
        mll = ExactMarginalLogLikelihood(mt_model_upd.likelihood, mt_model_upd)
        fit_gpytorch_mll(mll)

    return mt_model_upd