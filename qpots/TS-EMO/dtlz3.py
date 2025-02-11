import torch
import numpy as np
from botorch.test_functions.multi_objective import DTLZ3


def dtlz3(x, dim):
    X = torch.tensor(x, dtype=torch.float32)

    problem = DTLZ3(int(dim), num_objectives=6)

    result = problem.evaluate_true(X)
    # DTLZ3 negate=True doesn't work, negate here for results
    return -1*result.numpy()
