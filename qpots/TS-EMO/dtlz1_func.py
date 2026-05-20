import torch
import numpy as np
from botorch.test_functions import DTLZ1
from qpots.config import as_tensor

def dtlz1(x, dim):
    """
    Evaluate the two-objective DTLZ1 benchmark.

    Parameters
    ----------
    x : array-like
        Candidate points with ``dim`` design variables.
    dim : int
        Input dimensionality passed to BoTorch's ``DTLZ1`` constructor.

    Returns
    -------
    numpy.ndarray
        True two-objective DTLZ1 values.
    """
    X = as_tensor(x)

    problem = DTLZ1(int(dim), num_objectives=2).to(device=X.device, dtype=X.dtype)
    result = problem.evaluate_true(X)

    return result.detach().cpu().numpy()
    
