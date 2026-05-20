import torch
import numpy as np
from botorch.test_functions.multi_objective import VehicleSafety
from qpots.config import as_tensor

def vehiclesafety(x):
    """
    Evaluate the BoTorch vehicle-safety benchmark.

    Parameters
    ----------
    x : array-like
        Candidate vehicle-design points.

    Returns
    -------
    numpy.ndarray
        Objective values from ``VehicleSafety.evaluate_true``.
    """
    X = as_tensor(x)
    problem = VehicleSafety().to(device=X.device, dtype=X.dtype)
    result = problem.evaluate_true(X)
    return result.detach().cpu().numpy()
