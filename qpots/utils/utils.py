import torch
from botorch.utils.multi_objective.box_decompositions import FastNondominatedPartitioning
from botorch.utils.multi_objective.hypervolume import Hypervolume
from botorch.utils.multi_objective.hypervolume import Hypervolume
from botorch.utils.multi_objective.pareto import is_non_dominated
from sklearn.neighbors import KernelDensity
from scipy.spatial.distance import cdist
import argparse


def unstandardize(Y, train_y):
    """
    Reverse the standardization process for the output Y using the mean and standard deviation
    from the training data.

    Parameters:
        Y (torch.Tensor): The standardized output tensor.
        train_y (torch.Tensor): The training output data used to compute the mean and std dev.

    Returns:
        torch.Tensor: The unstandardized output tensor.
    """
    mean = train_y.mean(dim=0)
    std = train_y.std(dim=0)
    return Y * std + mean

def expected_hypervolume(gps, ref_point=torch.tensor([-300., -18.]), min=False):
    """
    Compute the expected hypervolume and Pareto front based on the GP model predictions.

    Parameters:
        gps: The multi-objective GP models.
        ref_point (torch.Tensor): Reference point for hypervolume calculation.

    Returns:
        tuple: (Hypervolume value, Pareto front tensor).
    """
    # Compute hypervolume using Pareto front with feasible points if constraints exist
    if min:
        if gps.ncons > 0:
            is_feas = (gps.train_y[..., -gps.ncons:] >= 0).all(dim=-1)
            is_feas_obj = gps.train_y[is_feas]
            pareto_mask = is_non_dominated(is_feas_obj, maximize=False) 
            pareto_front = is_feas_obj[pareto_mask]
            hv_calculator = Hypervolume(ref_point=-1*torch.tensor([0.335, 0.335]))
            hypervolume_value = hv_calculator.compute(-1*pareto_front)
            return hypervolume_value, pareto_front
        else:
            pareto_mask = is_non_dominated(gps.train_y, maximize=False) 
            pareto_front = gps.train_y[pareto_mask]
            hv_calculator = Hypervolume(ref_point=-1*torch.tensor([0.335, 0.335]))
            hypervolume_value = hv_calculator.compute(-1*pareto_front)
            return hypervolume_value, pareto_front
    else:
        if gps.ncons > 0:
            is_feas = (gps.train_y[..., -gps.ncons:] >= 0).all(dim=-1)
            is_feas_obj = gps.train_y[is_feas]
            bd1 = FastNondominatedPartitioning(ref_point.double(), is_feas_obj.double()[..., :gps.nobj])
            return bd1.compute_hypervolume(), bd1.pareto_Y
        else:
            bd1 = FastNondominatedPartitioning(ref_point.double(), gps.train_y[..., :gps.nobj].double())
            return bd1.compute_hypervolume(), bd1.pareto_Y

def gen_filtered_cands(gps, cands, ref_point=torch.tensor([0., 0.]), kernel_bandwidth=0.05):
    """
    Generate filtered candidate points based on the current Pareto front using Kernel Density Estimation (KDE).

    Parameters:
        gps: The multi-objective GP models.
        cands (torch.Tensor): Candidate points to filter.
        ref_point (torch.Tensor): Reference point for the Pareto front.
        kernel_bandwidth (float): Bandwidth for the KDE filter.

    Returns:
        torch.Tensor: Filtered candidate points.
    """
    # Compute the Pareto front based on the training data
    bd1 = FastNondominatedPartitioning(ref_point.double(), gps.train_y)
    nPareto = bd1.pareto_Y.shape[0]

    # Find the indices of Pareto optimal points in the training data
    ind = torch.tensor([(gps.train_y == bd1.pareto_Y[j]).nonzero()[0, 0] for j in range(nPareto)])
    x_nd = gps.train_x[ind]

    # Fit Kernel Density Estimation (KDE) using the Pareto optimal points
    kde = KernelDensity(kernel='gaussian', bandwidth=kernel_bandwidth).fit(x_nd)

    # Filter candidates based on the KDE
    U = torch.log(torch.rand(cands.shape[0]))
    w = kde.score_samples(cands)
    M = w.max()
    cands_fil = cands[w > U.numpy() * M]
    
    return cands_fil
    
def select_candidates(gps, pareto_set, device, q=1, seed=None):
        """
        Select candidates from the Pareto set.

        Parameters:
            gps: Gaussian Process models.
            pareto_set: Pareto optimal set of solutions.
            k: Number of samples to select.
            seed: Random seed for sampling.

        Returns:
            torch.Tensor: Selected candidates.
        """
        if seed is not None:
            torch.manual_seed(seed)

        D = cdist(pareto_set, gps.train_x.numpy())
        selected_indices = D.min(axis=-1).argsort()[-q:]
        selected_candidates = torch.from_numpy(pareto_set[selected_indices]).to(torch.double).to(device)
        return selected_candidates

def arg_parser():
    """
    This function contains all the necessary command line arguments. Each one has a default so it is not always necessary to specify them
    unless the user desires a different value of the argument.

    Arguments marked HPC are helpful for an HPC environment
    """
    # Argument parsing
    parser = argparse.ArgumentParser(description="Multi-objective Bayesian Optimization")
    parser.add_argument('--ntrain', type=int, default=20, help='Number of initial points')
    parser.add_argument('--iters', type=int, default=20, help='Number of iterations')
    parser.add_argument('--reps', type=int, default=20, help='Number of repetitions')
    parser.add_argument('--q', type=int, default=1, help='Number of points to sample at each iteration')
    parser.add_argument('--wd', type=str, default=".", help='Working directory to save results')
    parser.add_argument('--func', type=str, default=None, help='Test function to optimize. If using a custom function then don not specify') # HPC
    parser.add_argument('--ref_point', type=float, nargs='+', required=True, help='Reference point for hypervolume calculation') # HPC
    parser.add_argument('--dim', type=int, required=True, help='Input dimensionality')
    parser.add_argument('--nobj', type=int, default=2, help='Number of objectives')
    parser.add_argument('--ncons',type=int, default=0, help='Number of constraints')
    parser.add_argument('--acq', type=str, default="TS", help='Acquisition function') # HPC
    parser.add_argument('--nystrom', type=int, default=0, help='Use Nystrom approximation with filtered candidates')
    parser.add_argument('--nychoice', type=str, default='pareto', help='Pick a nystrom choice of m, pareto or random')
    parser.add_argument('--ngen', type=int, default=10, help='Number of generations for NSGA-II')
    args = parser.parse_args()

    return args
