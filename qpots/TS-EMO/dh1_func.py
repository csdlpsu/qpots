import torch
import numpy as np
from botorch.test_functions.multi_objective import DH1
from qpots.config import as_tensor


def dh1_eval(x, dim):
    """
    Evaluate the DH1 multiobjective benchmark.

    Parameters
    ----------
    x : array-like
        Candidate points with ``dim`` design variables.
    dim : int
        Input dimensionality passed to BoTorch's ``DH1`` constructor.

    Returns
    -------
    numpy.ndarray
        True DH1 objective values.
    """
    X = as_tensor(x)

    problem = DH1(int(dim)).to(device=X.device, dtype=X.dtype)

    result = problem.evaluate_true(X)

    return result.detach().cpu().numpy()
