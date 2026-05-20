import torch
import numpy as np
from botorch.test_functions.multi_objective import DTLZ3


def dtlz3(x, dim):
    """
    Evaluate the six-objective DTLZ3 benchmark for TS-EMO experiments.

    Parameters
    ----------
    x : array-like
        Candidate points with ``dim`` design variables.
    dim : int
        Input dimensionality passed to BoTorch's ``DTLZ3`` constructor.

    Returns
    -------
    numpy.ndarray
        Negated DTLZ3 objective values. The negation keeps the benchmark
        aligned with the maximization convention used elsewhere in qPOTS.
    """
    X = torch.tensor(x, dtype=torch.float32)

    problem = DTLZ3(int(dim), num_objectives=6)

    result = problem.evaluate_true(X)
    # DTLZ3 negate=True doesn't work, negate here for results
    return -1*result.numpy()
