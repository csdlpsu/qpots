import torch
import numpy as np
from botorch.test_functions.multi_objective import VehicleSafety

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
    X = torch.tensor(x, dtype=torch.float32)
    problem = VehicleSafety()
    result = problem.evaluate_true(X)
    return result.numpy()
