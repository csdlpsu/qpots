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
from qpots.utils.tc_utils import qmaximin 
from qpots.config import as_tensor, get_device, get_dtype
import numpy as np

from itertools import combinations
from typing import Iterable, Optional, Sequence, Tuple, Union, Dict, Any

def unstandardize(Y: Tensor, train_y: Tensor) -> Tensor:
    """
    Reverse the standardization of output `Y` using the mean and standard deviation 
    computed from the training data.

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
    mean = train_y.mean(dim=0)
    std = train_y.std(dim=0)
    return Y * std + mean

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
    std = torch.from_numpy(np.nanstd(train_y.detach().cpu().numpy(), axis=0, ddof=1)).to(train_y)
    return Y * std + mean

#1/9/26
def mtgp_posterior_mean_hypervolume(gps: ModelObject, ref_point: Tensor | None = None):
    """
        Compute the expected hypervolume based ONLY on MTGP model predictions.

        Parameters
        ----------
        gps : ModelObject
            The multi-objective GP models.
        ref_point : torch.Tensor, optional
            Reference point for hypervolume calculation.
        Returns
        -------
        
        """
    model=gps.models[0]
    train_x=gps.train_x
    ref_point = as_tensor([-300.0, -18.0], device=train_x.device, dtype=train_x.dtype) if ref_point is None else ref_point.to(train_x)
    
    task_ids = torch.arange(gps.nobj, device=train_x.device, dtype=train_x.dtype).repeat_interleave(train_x.shape[0]).unsqueeze(-1)
    train_x_extended = train_x.repeat(gps.nobj, 1)
    train_x_mt = torch.cat([train_x_extended,task_ids], dim=-1)  # (n*num_tasks, d+1)
    post = model.posterior(train_x_mt)
    model_y=post.mean.reshape(train_x.shape[0],gps.nobj)

    bd1 = FastNondominatedPartitioning(ref_point, model_y)
    hypervolume = bd1.compute_hypervolume()
    return hypervolume

def expected_hypervolume(
    gps: ModelObject, ref_point: Tensor | None = None, min: bool = False
) -> Tuple[float, Tensor]:
    """
    Compute the expected hypervolume and Pareto front based on GP model predictions.

    Parameters
    ----------
    gps : ModelObject
        The multi-objective GP models.
    ref_point : torch.Tensor, optional
        Reference point for hypervolume calculation.
    min : bool, optional
        If `True`, minimizes the objectives instead of maximizing them.

    Returns
    -------
    tuple
        - hypervolume_value (float): The computed hypervolume.
        - pareto_front (torch.Tensor): The Pareto front tensor.
    """
    train_y_filled=posterior_mean_fill(gps)
    configured_dtype = getattr(gps, "dtype", None)
    runtime_dtype = (
        configured_dtype
        if isinstance(configured_dtype, torch.dtype)
        else get_dtype()
    )
    runtime_device = train_y_filled.device
    train_y_filled = train_y_filled.to(device=runtime_device, dtype=runtime_dtype)
    ref_point = as_tensor(
        [-300.0, -18.0],
        device=runtime_device,
        dtype=runtime_dtype,
    ) if ref_point is None else torch.as_tensor(ref_point, device=runtime_device, dtype=runtime_dtype)
    
    nan_mask = ~torch.isnan(train_y_filled[..., :gps.nobj]).any(dim=1)
    
    if min:
        if gps.ncons > 0:
            is_feas = (gps.train_y[..., -gps.ncons:] >= 0).all(dim=-1)
            is_feas_obj = gps.train_y[is_feas]
            pareto_mask = is_non_dominated(is_feas_obj, maximize=False)
            pareto_front = is_feas_obj[pareto_mask]
            hv_calculator = Hypervolume(ref_point=-1 * as_tensor([0.335, 0.335], device=gps.train_y.device, dtype=gps.train_y.dtype))
            hypervolume_value = hv_calculator.compute(-1 * pareto_front)
            return hypervolume_value, pareto_front
        else:
            pareto_mask = is_non_dominated(gps.train_y, maximize=False)
            pareto_front = gps.train_y[pareto_mask]
            hv_calculator = Hypervolume(ref_point=-1 * as_tensor([0.335, 0.335], device=gps.train_y.device, dtype=gps.train_y.dtype))
            hypervolume_value = hv_calculator.compute(-1 * pareto_front)
            return hypervolume_value, pareto_front
    else:
        if gps.ncons > 0:
            is_feas = (train_y_filled[..., -gps.ncons:] >= 0).all(dim=-1)
            valid_mask = is_feas & nan_mask
            Y_valid = train_y_filled[valid_mask]

            bd1 = FastNondominatedPartitioning(ref_point, Y_valid[..., :gps.nobj])
            return bd1.compute_hypervolume(), bd1.pareto_Y
        else:
            Y_valid = train_y_filled[nan_mask, :gps.nobj]
            
            bd1 = FastNondominatedPartitioning(ref_point, Y_valid)
            return bd1.compute_hypervolume(), bd1.pareto_Y

def gen_filtered_cands(
    gps: ModelObject, cands: Tensor, ref_point: Tensor | None = None, kernel_bandwidth: float = 0.05
) -> Tensor:
    """
    Generate filtered candidate points based on the current Pareto front using Kernel Density Estimation (KDE).

    Parameters
    ----------
    gps : ModelObject
        The multi-objective GP models.
    cands : torch.Tensor
        Candidate points to filter.
    ref_point : torch.Tensor, optional
        Reference point for the Pareto front.
    kernel_bandwidth : float, optional
        Bandwidth for the KDE filter.

    Returns
    -------
    torch.Tensor
        Filtered candidate points.
    """
    cands = cands.to(gps.train_x)
    ref_point = as_tensor([0.0, 0.0], device=gps.train_y.device, dtype=gps.train_y.dtype) if ref_point is None else ref_point.to(gps.train_y)
    bd1 = FastNondominatedPartitioning(ref_point, gps.train_y)
    nPareto = bd1.pareto_Y.shape[0]

    # Find Pareto-optimal indices
    ind = torch.tensor([(gps.train_y == bd1.pareto_Y[j]).nonzero()[0, 0] for j in range(nPareto)], device=gps.train_x.device)
    x_nd = gps.train_x[ind]

    # Fit KDE to Pareto points
    kde = KernelDensity(kernel="gaussian", bandwidth=kernel_bandwidth).fit(x_nd.detach().cpu().numpy())

    # Filter candidates using KDE sampling
    U = torch.log(torch.rand(cands.shape[0], device=cands.device, dtype=cands.dtype))
    w = kde.score_samples(cands.detach().cpu().numpy())
    M = w.max()
    cands_fil = cands[w > U.detach().cpu().numpy() * M]

    return cands_fil

def select_candidates(
    gps: ModelObject, pareto_set: np.ndarray, device: torch.device, q: int = 1, seed: int = None
) -> Tensor:
    """
    Select candidates from the Pareto-optimal set.

    Parameters
    ----------
    gps : ModelObject
        Gaussian Process models.
    pareto_set : numpy.ndarray
        Pareto-optimal set of solutions.
    device : torch.device
        Device to store the selected candidates.
    q : int, optional
        Number of candidates to select. Defaults to 1.
    seed : int, optional
        Random seed for sampling.

    Returns
    -------
    torch.Tensor
        Selected candidate points.
    """
    if seed is not None:
        torch.manual_seed(seed)
    print("Sample Pareto Set, X*, is of shape: ",pareto_set.shape)
    device = get_device(device)
    dtype_attr = getattr(gps, "dtype", None)
    dtype = dtype_attr if isinstance(dtype_attr, torch.dtype) else gps.train_x.dtype
    D = cdist(pareto_set, gps.train_x.detach().cpu().numpy())
    selected_indices = D.min(axis=-1).argsort()[-q:]
    selected_candidates = torch.as_tensor(pareto_set[selected_indices], device=device, dtype=dtype)
    return selected_candidates

def select_candidates_partial_info(gps: ModelObject, pareto_set: np.ndarray, device: torch.device, q: int = 1, seed: int = None, thresh: float = None, rescaling: str = "_"
) -> Tensor:
    """
    Select candidates from the Pareto-optimal set using partial information.

    This function selects `q` candidate points from a given Pareto-optimal set and assigns
    tasks for evaluation. Task selection can be either random or based on a variance threshold.
    If `thresh` is provided, tasks with posterior variance above the threshold are chosen.
    Otherwise, tasks are selected randomly. Any candidates with all-NaN task assignments
    are removed.

    Parameters
    ----------
    gps : ModelObject
        Object containing Gaussian Process models, including training data and number of objectives/constraints.
    pareto_set : numpy.ndarray
        Pareto-optimal set of solutions. Shape `[num_points, num_variables]`.
    device : torch.device
        Device to store the selected candidates (CPU or GPU).
    q : int, optional
        Number of candidates to select. Defaults to 1.
    seed : int, optional
        Random seed for reproducibility when selecting tasks randomly or applying variance thresholding.
    thresh : float, optional
        Variance threshold for selecting tasks. If None, tasks are chosen randomly.
    rescaling : str, optional
        Method to rescale variance for thresholding. Options are:
        - "_" : no rescaling (raw variance)
        - "std" : standardize to mean 0, std 1
        - "norm" : normalize to [0, 1]

    Returns
    -------
    Tuple[torch.Tensor, torch.Tensor]
        - `selected_candidates`: Tensor of shape `[num_selected, num_variables]` containing the chosen candidate points.
        - `task_ids`: Tensor of shape `[num_selected, num_tasks]` indicating which tasks are selected for each candidate.
    """
    print("Sample Pareto Set, X*, is of shape: ",pareto_set.shape)
    device = get_device(device)
    dtype_attr = getattr(gps, "dtype", None)
    dtype = dtype_attr if isinstance(dtype_attr, torch.dtype) else gps.train_x.dtype
    D = cdist(pareto_set, gps.train_x.detach().cpu().numpy())
    selected_indices = D.min(axis=-1).argsort()[-q:]
    selected_candidates = torch.as_tensor(pareto_set[selected_indices], device=device, dtype=dtype)
    
    if pareto_set.shape[0]<=q:
        print(f"WARNING Pareto Set from NSGA-II is smaller than number of batch points: {pareto_set.shape[0]}")

    model=gps.models[0]
    num_inputs=selected_candidates.shape[0]
    
    num_outputs=gps.nobj+gps.ncons 
    if thresh is None:
        print("Random Task Choice:")
    
        task_ids = torch.full((num_inputs, num_outputs), float("nan"), device=device, dtype=dtype)
        tasks_stacked = torch.arange(num_outputs, device=device, dtype=dtype).repeat(num_inputs, 1)

        mask = torch.randint(0, 2, (num_inputs, num_outputs), device=device).bool()
        task_ids[mask] = tasks_stacked[mask]

        # remove rows that are all NaN
        nan_mask = ~torch.isnan(task_ids).all(dim=1)
        task_ids = task_ids[nan_mask]
        selected_candidates = selected_candidates[nan_mask]

    else:
        print("Variance Thresholding Task Choice")
        if seed is not None:
            torch.manual_seed(seed)

        task_ids=torch.arange(end=num_outputs, device=device, dtype=dtype).repeat_interleave(num_inputs).reshape(-1,1)
        new_x_mt=torch.cat([selected_candidates.repeat(num_outputs,1),task_ids],dim=-1)
        rand_y_mt=torch.rand(num_inputs,num_outputs, device=device, dtype=dtype).T.reshape(-1, 1)
    
        new_model = model.condition_on_observations(X=new_x_mt, Y=rand_y_mt) #Fantasizing
        new_variance = new_model.posterior(selected_candidates).variance
        
        task_ids = torch.full_like(new_variance,float('nan'))
        
        if rescaling == "std": #scaling variance to a mean of 0 std of 1 for thresholding only
            standardized_variance=standardize(new_variance)
            print("Standardized_Variance: ",standardized_variance)
            mask = standardized_variance>thresh.unsqueeze(0) 
        elif rescaling == "norm": #scaling variance between 0 and 1 for thresholding only
            normalized_variance=minmax_scale(new_variance)
            print("Normalized_Variance: ",normalized_variance)
            mask = normalized_variance>thresh.unsqueeze(0) 
        else:
            mask = new_variance>thresh.unsqueeze(0)
        
        tasks_stacked = torch.arange(num_outputs, device=device, dtype=dtype).repeat(num_inputs, 1)
        task_ids[mask]=tasks_stacked[mask]
        # Removing any empty rows
        nan_mask = ~torch.isnan(task_ids).all(dim=1)
        task_ids = task_ids[nan_mask]
        selected_candidates = selected_candidates[nan_mask]
 
    return selected_candidates, task_ids


#Total Correlation work: 12/31
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
    std = var.sqrt()

    # Correlation matrix
    with torch.no_grad():
        R = cov / (std.unsqueeze(-1) * std.unsqueeze(-2))

    # Stabilize and compute logdet via Cholesky: logdet(R) = 2 * sum(log(diag(L)))
    m = R.shape[-1]
    eye = torch.eye(m, device=R.device, dtype=R.dtype).expand(R.shape[:-2] + (m, m))
    Rj = R + jitter * eye

    L = torch.linalg.cholesky(Rj)
    logdet = 2.0 * torch.log(torch.diagonal(L, dim1=-2, dim2=-1)).sum(dim=-1)

    with torch.no_grad():
        TC = -0.5 * logdet
    return R, TC


## Mutual Information: ##
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

# Mutual Info
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

# Brute Force Maximjze MI for set
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
    Brute-force search over output subsets under a Gaussian assumption.

    You can pass either:
      - cov_or_samples as a (K,K) covariance matrix, OR
      - cov_or_samples as (N,K) samples (rows=samples), from which we estimate covariance.

    Parameters
    ----------
    cov_or_samples : np.ndarray
        (K,K) covariance OR (N,K) samples.
    subset_size : int or None
        If provided, restrict search to subsets whose size equals
        ``subset_size``.
        If None, searches all non-trivial subsets (excluding empty/full).
    jitter, base : see above
    assume_samples : bool or None
        If None, auto-detect: (K,K) => covariance; otherwise => samples.
    deduplicate_complements : bool
        If ``subset_size`` is ``None``, avoid evaluating both a subset and its
        complement by restricting the search to subsets of size at most
        ``floor(K / 2)``. This is safe because the split mutual information is
        symmetric.
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
    if cov_or_samples.shape[-1]==2:
        best_S=torch.randint(0,2,[1]).item()
        best_Sc=1-best_S
        out: Dict[str, Any] = {"S": best_S, "Sc": best_Sc}
        print("K=2, exiting", out)
        return out
    
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
    #print("BEST_S",best_S)
    if best_S is None:
        # This only happens if K < 2.
        best_S = tuple()
        best_mi = 0.0

    best_S_set = set(best_S)
    best_Sc = tuple(i for i in idx if i not in best_S_set)

    out: Dict[str, Any] = {"S": best_S, "Sc": best_Sc, "mi": float(best_mi), "cov": cov}
    if return_all_scores:
        out["scores"] = scores
    print(out)
    return out


def select_candidates_total_correlation(gps: ModelObject, pareto_set: np.ndarray, device: torch.device, q: int = 1, seed: int = None, thresh: float = None
) -> Tensor:
    """
    Select Pareto-set candidates and choose which outputs to evaluate.

    Candidate locations are chosen from the sampled Pareto set using the same
    maximin-distance heuristic as ``select_candidates``. If ``thresh`` is
    ``None``, each selected location receives a random subset of outputs. If a
    threshold is provided, the posterior output covariance at each candidate is
    converted to total correlation. Highly coupled candidates are assigned the
    subset of outputs that maximizes Gaussian mutual information; weakly
    coupled candidates are assigned all outputs.

    Parameters
    ----------
    gps : ModelObject
        Fitted model container. This routine expects ``gps.models[0]`` to be a
        multi-output model whose posterior exposes the joint covariance across
        outputs.
    pareto_set : numpy.ndarray
        Candidate Pareto-optimal design points returned by NSGA-II, with shape
        ``n_pareto x d``.
    device : torch.device
        Device used for the returned candidate tensor.
    q : int, optional
        Maximum number of design points to select.
    seed : int, optional
        Reserved for reproducible random task selection.
    thresh : float, optional
        Total-correlation threshold. ``None`` enables random task selection;
        otherwise candidates with ``abs(TC) > thresh`` use the mutual
        information split and candidates below the threshold evaluate all
        outputs.

    Returns
    -------
    tuple[torch.Tensor, torch.Tensor]
        Selected candidates and a task-id matrix. Task entries contain the
        output index to evaluate or ``NaN`` when that output is skipped.

    Notes
    -----
    The feasibility/objective convention follows the rest of qPOTS: objectives
    and constraints are modeled together, with constraints stored after the
    objective columns.
    """
    #Old Maxmin distance
    #"""
    print("Sample Pareto Set, X*, is of shape: ",pareto_set.shape)
    device = get_device(device)
    dtype_attr = getattr(gps, "dtype", None)
    dtype = dtype_attr if isinstance(dtype_attr, torch.dtype) else gps.train_x.dtype
    D = cdist(pareto_set, gps.train_x.detach().cpu().numpy())
    selected_indices = D.min(axis=-1).argsort()[-q:]
    selected_candidates = torch.as_tensor(pareto_set[selected_indices], device=device, dtype=dtype)
    #"""
    #New Maxmin distance, 1/15, in tc_utils.py: 
    #selected_candidates = qmaximin(gps.train_x, torch.tensor(pareto_set), q=q)

    
    if pareto_set.shape[0]<=q:
        print(f"WARNING Pareto Set from NSGA-II is smaller than number of batch points: {pareto_set.shape[0]}")

    model=gps.models[0]
    dim=selected_candidates.shape[1]
    num_outputs=gps.nobj+gps.ncons 
    if thresh is None:
        print("Random Task Choice:")
    
        task_ids = torch.full((q, num_outputs), float("nan"), device=device, dtype=dtype)
        tasks_stacked = torch.arange(num_outputs, device=device, dtype=dtype).repeat(q, 1)

        mask = torch.randint(0, 2, (q, num_outputs), device=device).bool()
        task_ids[mask] = tasks_stacked[mask]

        # remove rows that are all NaN
        nan_mask = ~torch.isnan(task_ids).all(dim=1)
        task_ids = task_ids[nan_mask]
        selected_candidates = selected_candidates[nan_mask]

    else:
        print("\nTotal Correlation Thresholding Task Choice")
        
        #Method From Dr. R.
        tc_i = []
        eval_list=[]
        for i in range(selected_candidates.shape[0]): #was range q, but NSGA-II front was getting too small????? BAD problem
            post = model.posterior(selected_candidates[i].view(-1,dim))
            with torch.no_grad():
                cov  = post.distribution.covariance_matrix.detach()  # 2x2 (materialized)

            _, tc = corr_and_total_correlation(cov)
  
           
            tc_i.append(torch.abs(tc).item())
            if torch.abs(tc) > thresh:
                res = argmax_mi_subset_bruteforce(cov, assume_samples=False, base=2.0)

                eval_subset=res["S"]
                if isinstance(eval_subset, int):
                    eval_subset = [eval_subset]

                print("res['S']:", res["S"], type(res["S"]))
                eval_list.append([
                    i if i in eval_subset else -1
                    for i in range(num_outputs)
                ])

            else:

                eval_list.append(list(range(num_outputs)))
        eval_tensor=torch.as_tensor(eval_list, device=device, dtype=dtype)
            
        print("All q TC's: \n",tc_i,"\n")

        #Selecting the tasks
        #eval_subset is the MI identifies
      
        tasks_stacked = torch.arange(num_outputs, device=device, dtype=dtype).repeat(selected_candidates.shape[0], 1) #was using q instead of selected_candidates.shape[0], but NSGA-II issue
        print("tasks_stacked:\n",tasks_stacked)
        task_ids = torch.full_like(tasks_stacked,float('nan'))
        print("eval_tensor:\n",eval_tensor)
        mask=tasks_stacked == eval_tensor
        task_ids[mask]=tasks_stacked[mask]
    return selected_candidates, task_ids


def _augment_X_with_tasks(
    X: torch.Tensor,  # (n, d) WITHOUT the task feature
    num_tasks: int,
    task_feature: int,
) -> torch.Tensor:
    """
    Build long-format ``MultiTaskGP`` inputs from task-free design points.

    Parameters
    ----------
    X : torch.Tensor
        Base design matrix with shape ``n x d`` and no task feature.
    num_tasks : int
        Number of task/output indices to append.
    task_feature : int
        Position where the task index should be inserted in the augmented
        ``d + 1`` dimensional input. Negative indices follow Python indexing.

    Returns
    -------
    torch.Tensor
        Augmented tensor with shape ``(n * num_tasks) x (d + 1)``. Rows are
        grouped by task index.

    Raises
    ------
    ValueError
        If ``X`` is not two-dimensional or ``task_feature`` is invalid for the
        augmented input dimension.
    """
    if X.ndim != 2:
        raise ValueError(f"Expected X to be 2D (n x d). Got shape {tuple(X.shape)}.")

    n, d = X.shape
    aug_dim = d + 1
    tf = task_feature if task_feature >= 0 else aug_dim + task_feature
    if not (0 <= tf < aug_dim):
        raise ValueError(f"task_feature={task_feature} invalid for augmented dim {aug_dim}.")

    # Repeat X in blocks: task 0 has rows [0:n], task 1 has rows [n:2n], etc.
    X_rep = X.repeat(num_tasks, 1)  # (n*num_tasks, d)

    # Task ids aligned with the blocks above
    task_ids = (
        torch.arange(num_tasks, device=X.device, dtype=X.dtype)
        .repeat_interleave(n)
        .unsqueeze(-1)
    )  # (n*num_tasks, 1)

    # Insert task column at index tf
    X_aug = torch.cat([X_rep[..., :tf], task_ids, X_rep[..., tf:]], dim=-1)  # (n*num_tasks, d+1)
    return X_aug

def hypervolume_from_posterior_mean_mtgp(
    mt_model,
    X: torch.Tensor,                      # (n, d) base features (no task column)
    task_feature: int,
    *,
    ref_point,
    maximize,
) -> torch.Tensor:
    """
    Compute hypervolume of the *posterior mean* vectors produced by a MultiTaskGP.

    Args:
        mt_model: fitted MultiTaskGP (task encoded as input feature).
        X: (n, d) points at which to compute posterior mean for each task.
        task_feature: index where the task column lives in the augmented input (d+1).
        ref_point: reference point in objective space, length K (K inferred from this).
        maximize: if False, treats objectives as minimization (internally negates).

    Returns:
        A scalar tensor: hypervolume of the non-dominated posterior-mean outcomes.
    """
    mt_model.eval()

    # Move to model device/dtype
    p = next(mt_model.parameters())
    device, dtype = p.device, p.dtype
    X = X.to(device=device, dtype=dtype)
    ref_point = torch.as_tensor(ref_point, device=device, dtype=dtype).view(-1)
    K = ref_point.numel()

    # Build long-format inputs and get posterior mean for each (x, task)
    X_aug = _augment_X_with_tasks(X, num_tasks=K, task_feature=task_feature)  # (n*K, d+1)
    post = mt_model.posterior(X_aug)
    mean_flat = post.mean.squeeze(-1)  # (n*K,)

    n = X.shape[0]
    Y_mean = mean_flat.view(K, n).transpose(0, 1).contiguous()  # (n, K)

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

def compute_true_hypervolume(
    Y: torch.Tensor,                      # (n, d) base features (no task column)
    ref_point,
    nobj,
    ncons,
    maximize,
) -> torch.Tensor:
    """
    Compute true hypervolume .

    Args:
        Y: (n, d) evaluated  y values.
        ref_point: reference point in objective space, length K (K inferred from this).
        maximize: if False, treats objectives as minimization (internally negates).

    Returns:
        A scalar tensor: hypervolume of the non-dominated posterior-mean outcomes.
    """
    #####NEW 1/30
    
    #Checking Constraint Feasibility
    if ncons > 0:
        is_feas = (Y[..., -ncons:] >= 0).all(dim=-1)
        Y = Y[is_feas]

    #Taking only the objectives
    Y=Y[...,:nobj]

    ref_point = torch.as_tensor(ref_point, device=Y.device, dtype=Y.dtype)

    # Hypervolume assumes maximization. If minimizing, negate both.
    if not maximize:
        Y = -Y
        ref_point = -ref_point

    # Non-dominated subset of posterior means
    nd_mask = is_non_dominated(Y)
    pareto_Y = Y[nd_mask]

    if pareto_Y.numel() == 0:
        # Shouldn't happen unless n=0, but keep it safe:
        return torch.zeros((), device=Y.device, dtype=Y.dtype)

    hv = Hypervolume(ref_point=ref_point).compute(pareto_Y)
    return hv

def minmax_scale(x, dim=None,eps=1e-12):
    """
    Rescale a tensor to the ``[0, 1]`` interval.
    
    Parameters
    ----------
    x : torch.Tensor
        Input tensor of any shape.
    dim : int or None, optional
        Dimension along which to compute the minimum and maximum. If ``None``,
        scale globally across the whole tensor.
    eps : float, optional
        Small denominator floor that prevents division by zero when all values
        along a slice are equal.

    Returns
    -------
    torch.Tensor
        Tensor with the same shape as ``x`` and values scaled to ``[0, 1]`` up
        to numerical precision.
    """
    x_min = x.min(dim=dim, keepdim=True).values if dim is not None else x.min()
    x_max = x.max(dim=dim, keepdim=True).values if dim is not None else x.max()
    
    return (x - x_min) / (x_max - x_min + eps)

def posterior_mean_fill(gps: ModelObject):
    """
    Fill missing values in training targets using the GP posterior mean.

    After partial-information updates, `gps.train_y` may contain NaN entries,
    which makes it impossible to compute metrics like the true Pareto front directly.
    This function replaces all NaN entries with the corresponding posterior mean
    predicted by the MultiTaskGP model for that input and task.

    Parameters
    ----------
    gps : ModelObject
        Object containing the MultiTaskGP model(s) and training data, including
        `train_x`, `train_y`, number of objectives (`nobj`), and constraints (`ncons`).

    Returns
    -------
    torch.Tensor
        A tensor of the same shape as `gps.train_y` with NaNs replaced by the
        posterior mean for each missing entry.
    """
    mtgp=gps.models[0]
    full_train_y = gps.train_y.clone().detach()
    for m in range(gps.nobj+gps.ncons):
        missing_mask = torch.isnan(full_train_y[:, m])
        if missing_mask.any():
            X_missing = gps.train_x[missing_mask]
            task_idx = torch.full((X_missing.shape[0], 1), m, dtype=torch.long, device=gps.train_x.device)
            posterior = mtgp.posterior(torch.cat([X_missing, task_idx], dim=-1))
            full_train_y[missing_mask, m] = posterior.mean[:, m]
            
    return full_train_y

def arg_parser():
    """
    Parses command-line arguments for the multi-objective Bayesian optimization script.

    This function provides default values for each argument, allowing customization for 
    different optimization setups, including high-performance computing (HPC) environments.

    Returns
    -------
    argparse.Namespace
        Parsed command-line arguments.
    """
    parser = argparse.ArgumentParser(description="Multi-objective Bayesian Optimization")

    # Experiment settings
    parser.add_argument("--ntrain", type=int, default=20, help="Number of initial training points.")
    parser.add_argument("--iters", type=int, default=20, help="Number of optimization iterations.")
    parser.add_argument("--reps", type=int, default=20, help="Number of repetitions.")
    parser.add_argument("--q", type=int, default=1, help="Batch size for sampled points per iteration.")
    parser.add_argument("--wd", type=str, default=".", help="Working directory for saving results.")

    # Function and optimization settings
    parser.add_argument("--func", type=str, default=None, help="Test function for optimization. If using a custom function, do not specify.")  # HPC
    parser.add_argument("--ref_point", type=float, nargs="+", required=True, help="Reference point for hypervolume calculation.")  # HPC
    parser.add_argument("--dim", type=int, required=True, help="Dimensionality of the input space.")
    parser.add_argument("--nobj", type=int, default=2, help="Number of objectives.")
    parser.add_argument("--ncons", type=int, default=0, help="Number of constraints.")

    # Acquisition function settings
    parser.add_argument("--acq", type=str, default="TS", help="Acquisition function to use.")  # HPC
    parser.add_argument("--nystrom", type=int, default=0, help="Use Nystrom approximation with filtered candidates.")
    parser.add_argument("--nychoice", type=str, default="pareto", help="Method for Nystrom selection: 'pareto' or 'random'.")
    parser.add_argument("--ngen", type=int, default=10, help="Number of generations for NSGA-II.")

    return parser.parse_args()
