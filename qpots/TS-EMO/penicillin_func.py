import torch
import numpy as np
from botorch.test_functions.multi_objective import Penicillin


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
    X = torch.tensor(x, dtype=torch.float32)

    problem = Penicillin()

    result = problem.evaluate_true(X)

    return result

    
