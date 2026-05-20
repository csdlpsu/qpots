import torch
import numpy as np
from botorch.test_functions.multi_objective import DTLZ7


def dtlz7(x, dim):
    """
    Evaluate the two-objective DTLZ7 benchmark.

    Parameters
    ----------
    x : array-like
        Candidate points with ``dim`` design variables.
    dim : int
        Input dimensionality passed to BoTorch's ``DTLZ7`` constructor.

    Returns
    -------
    numpy.ndarray
        True two-objective DTLZ7 values.
    """
    X = torch.tensor(x, dtype=torch.float32)

    problem = DTLZ7(int(dim), num_objectives=2)
    result = problem.evaluate_true(X)

    return result.numpy()
