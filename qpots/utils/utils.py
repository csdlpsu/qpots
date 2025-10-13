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
import numpy as np


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
    std = torch.from_numpy(np.nanstd(train_y.cpu().numpy(), axis=0, ddof=1)).to(train_y)
    return Y * std + mean

def expected_hypervolume(
    gps: ModelObject, ref_point: Tensor = torch.tensor([-300.0, -18.0]), min: bool = False
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
    
    nan_mask = ~torch.isnan(train_y_filled[..., :gps.nobj]).any(dim=1)
    
    if min:
        if gps.ncons > 0:
            is_feas = (gps.train_y[..., -gps.ncons:] >= 0).all(dim=-1)
            is_feas_obj = gps.train_y[is_feas]
            pareto_mask = is_non_dominated(is_feas_obj, maximize=False)
            pareto_front = is_feas_obj[pareto_mask]
            hv_calculator = Hypervolume(ref_point=-1 * torch.tensor([0.335, 0.335]))
            hypervolume_value = hv_calculator.compute(-1 * pareto_front)
            return hypervolume_value, pareto_front
        else:
            pareto_mask = is_non_dominated(gps.train_y, maximize=False)
            pareto_front = gps.train_y[pareto_mask]
            hv_calculator = Hypervolume(ref_point=-1 * torch.tensor([0.335, 0.335]))
            hypervolume_value = hv_calculator.compute(-1 * pareto_front)
            return hypervolume_value, pareto_front
    else:
        if gps.ncons > 0:
            is_feas = (train_y_filled[..., -gps.ncons:] >= 0).all(dim=-1)
            valid_mask = is_feas & nan_mask
            Y_valid = train_y_filled[valid_mask]

            bd1 = FastNondominatedPartitioning(ref_point.double(), Y_valid[..., :gps.nobj].double())
            return bd1.compute_hypervolume(), bd1.pareto_Y
        else:
            Y_valid = train_y_filled[nan_mask, :gps.nobj]
            
            bd1 = FastNondominatedPartitioning(ref_point.double(), Y_valid.double())
            return bd1.compute_hypervolume(), bd1.pareto_Y

def gen_filtered_cands(
    gps: ModelObject, cands: Tensor, ref_point: Tensor = torch.tensor([0.0, 0.0]), kernel_bandwidth: float = 0.05
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
    bd1 = FastNondominatedPartitioning(ref_point.double(), gps.train_y)
    nPareto = bd1.pareto_Y.shape[0]

    # Find Pareto-optimal indices
    ind = torch.tensor([(gps.train_y == bd1.pareto_Y[j]).nonzero()[0, 0] for j in range(nPareto)])
    x_nd = gps.train_x[ind]

    # Fit KDE to Pareto points
    kde = KernelDensity(kernel="gaussian", bandwidth=kernel_bandwidth).fit(x_nd)

    # Filter candidates using KDE sampling
    U = torch.log(torch.rand(cands.shape[0]))
    w = kde.score_samples(cands)
    M = w.max()
    cands_fil = cands[w > U.numpy() * M]

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

    D = cdist(pareto_set, gps.train_x.numpy())
    selected_indices = D.min(axis=-1).argsort()[-q:]
    selected_candidates = torch.from_numpy(pareto_set[selected_indices]).to(torch.double).to(device)
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

    D = cdist(pareto_set, gps.train_x.numpy())
    selected_indices = D.min(axis=-1).argsort()[-q:]
    selected_candidates = torch.from_numpy(pareto_set[selected_indices]).to(torch.double).to(device)
    
    if pareto_set.shape[0]<=q:
        print("WARNING Pareto Set from NSGA-II is smaller than number of batch points")

    model=gps.models[0]
    num_inputs=selected_candidates.shape[0]
    
    num_outputs=gps.nobj+gps.ncons 
    if thresh is None:
        print("Random Task Choice:")
    
        task_ids = torch.full((num_inputs, num_outputs), float("nan")).double()
        tasks_stacked = torch.arange(num_outputs).repeat(num_inputs, 1).double()

        mask = torch.randint(0, 2, (num_inputs, num_outputs)).bool()
        task_ids[mask] = tasks_stacked[mask].double()

        # remove rows that are all NaN
        nan_mask = ~torch.isnan(task_ids).all(dim=1)
        task_ids = task_ids[nan_mask]
        selected_candidates = selected_candidates[nan_mask]

    else:
        print("Variance Thresholding Task Choice")
        if seed is not None:
            torch.manual_seed(seed)

        task_ids=torch.arange(end=num_outputs).repeat_interleave(num_inputs).reshape(-1,1)
        new_x_mt=torch.cat([selected_candidates.repeat(num_outputs,1),task_ids],dim=-1).double()
        rand_y_mt=torch.rand(num_inputs,num_outputs).double().T.reshape(-1, 1).double() 
    
        new_model = model.condition_on_observations(X=new_x_mt.double(), Y=rand_y_mt.double()) #Fantasizing
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
        
        tasks_stacked = torch.arange(num_outputs).repeat(num_inputs, 1).double()
        task_ids[mask]=tasks_stacked[mask]
        # Removing any empty rows
        nan_mask = ~torch.isnan(task_ids).all(dim=1)
        task_ids = task_ids[nan_mask]
        selected_candidates = selected_candidates[nan_mask]
 
    return selected_candidates, task_ids


def minmax_scale(x, dim=None,eps=1e-12):
    """
    Rescale a tensor to [0, 1] along the each column (task).
    
    Args:
        x: input tensor (any shape)
        dim: dimension along which to compute min/max. 
             If None, scales over the entire tensor.
        eps: small number to avoid division by zero
        
    Returns:
        scaled tensor of same shape as x
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
            
            full_train_y[missing_mask, m] = unstandardize_ignore_nan(posterior.mean,gps.train_y)[:, m]
            
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
