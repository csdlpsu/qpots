import torch
from torch import Tensor
from botorch.utils.multi_objective.box_decompositions import FastNondominatedPartitioning
from botorch.utils.multi_objective.hypervolume import Hypervolume
from botorch.utils.multi_objective.pareto import is_non_dominated
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
            is_feas = (gps.train_y[..., -gps.ncons:] >= 0).all(dim=-1)
            is_feas_obj = gps.train_y[is_feas]
            bd1 = FastNondominatedPartitioning(ref_point.double(), is_feas_obj.double()[..., : gps.nobj])
            return bd1.compute_hypervolume(), bd1.pareto_Y
        else:
            bd1 = FastNondominatedPartitioning(ref_point.double(), gps.train_y[..., : gps.nobj].double())
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
