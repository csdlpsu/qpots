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

def get_model_identified_hv_maximizing_set(
    model,
    problem,
    ref_point,
    population_size=250,
    max_gen=100,
    
):
    """Optimize the posterior mean using NSGA-II."""
    tkwargs = {
        "dtype": ref_point.dtype,
        "device": ref_point.device,
    }
    dim = problem.dim

    class PosteriorMeanPymooProblem(Problem):
        def __init__(self):
            super().__init__(
                n_var=dim,
                n_obj=problem.nobj,
                type_var=np.double,
            )
            self.xl = np.zeros(dim)
            self.xu = np.ones(dim)

        def _evaluate(self, x, out, *args, **kwargs):
            #torch.manual_seed(2439) #Setting same seed to see if its the same now? for sampling for posterior optimization
            X = torch.from_numpy(x).to(**tkwargs)
            y = model.posterior(X).sample().reshape(-1,problem.nobj)
            #print("evaluate post sample:\n",y)                    
            out["F"] = -y.cpu().numpy()

    pymoo_problem = PosteriorMeanPymooProblem()
    algorithm = NSGA2(
        pop_size=population_size,
        eliminate_duplicates=True,
    )
    res = minimize(
        pymoo_problem,
        algorithm,
        # termination=MaximumGenerationTermination(max_gen),
        pop_size=200,
        seed=2430,
        verbose=False,
    )
    X = torch.tensor(
        res.X,
        **tkwargs,
    )
    X = unnormalize(X, problem.get_bounds())
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

    print("Rj: \n",Rj)
    try: #Runs when Rj is positive definite
        L = torch.linalg.cholesky(Rj)
        logdet = 2.0 * torch.log(torch.diagonal(L, dim1=-2, dim2=-1)).sum(dim=-1)
        TC = -0.5 * logdet
    except RuntimeError: #If not, run coupled evaluation at this location (TC=None)
        print("R was not invertible, performing coupled evaluation")
        TC=None 

    return R, TC

def computeTC(x,mt_model):
    dim=mt_model.train_inputs[0].shape[-1]
    #print("dim",dim)
    #print("x",x)
    #print("x viewed",x.view(-1,dim-1))
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
    
    mt_model_kind = "MultiTaskGP(task_feature)"
    # MultiTaskGP expects a single output (N x 1) and a task feature in X.
    # We'll stack the observations for the two objectives:
    #
    # train_X_mt: (2*n_train, d+1), last column is task index (0 or 1)
    # train_Y_mt: (2*n_train, 1)
    task_feature = d  # last column
    
    train_X0 = torch.cat(
        [train_X, torch.zeros(n_train, 1, device=device, dtype=dtype)], dim=-1
    )
    train_X1 = torch.cat(
        [train_X, torch.ones(n_train, 1, device=device, dtype=dtype)], dim=-1
    )
    train_X_mt = torch.cat([train_X0, train_X1], dim=0)
    train_Y_mt = torch.cat([train_Y[:, [0]], train_Y[:, [1]]], dim=0)
    
    # NOTE: Do not Normalize the task column. We'll normalize only the first d dims by pre-normalizing.
    # Here train_X is already in [0,1]^d, so we skip additional normalization for simplicity.
    #print("train_X_mt:\n",train_X_mt)
    #print("train_Y_mt:\n",train_Y_mt)
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
    #print("train_X_mt:\n",X_upd)
    #print("train_Y_mt:\n",Y_upd)

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