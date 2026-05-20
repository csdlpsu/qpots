import torch
import numpy as np
from botorch.test_functions.multi_objective import Penicillin
from qpots.config import as_tensor


def Penicillin_evaluate(x):
    """
    Evaluate the BoTorch Penicillin simulator benchmark.

    Parameters
    ----------
    x : array-like
        Candidate bioprocess design points.

    Returns
    -------
    torch.Tensor
        True Penicillin benchmark objective values.
    """
    X = as_tensor(x)

    problem = Penicillin().to(device=X.device, dtype=X.dtype)

    result = problem.evaluate_true(X)

    return result

    
