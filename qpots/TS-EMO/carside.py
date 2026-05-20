import torch
import numpy as np
from botorch.test_functions.multi_objective import CarSideImpact

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
    X = torch.tensor(x, dtype=torch.float32)
    problem = CarSideImpact()
    result = problem.evaluate_true(X)
    return result.numpy()
