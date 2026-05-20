import torch
import numpy as np
from botorch.test_functions.multi_objective import CarSideImpact
from qpots.config import as_tensor

def carside(x):
    """
    Evaluate the BoTorch car-side-impact benchmark.

    Parameters
    ----------
    x : array-like
        Candidate vehicle-design points.

    Returns
    -------
    numpy.ndarray
        Objective values from ``CarSideImpact.evaluate_true``.
    """
    X = as_tensor(x)
    problem = CarSideImpact().to(device=X.device, dtype=X.dtype)
    result = problem.evaluate_true(X)
    return result.detach().cpu().numpy()
