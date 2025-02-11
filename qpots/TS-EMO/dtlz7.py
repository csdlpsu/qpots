import torch
import numpy as np
from botorch.test_functions.multi_objective import DTLZ7


def dtlz7(x, dim):
    X = torch.tensor(x, dtype=torch.float32)

    problem = DTLZ7(int(dim), num_objectives=2)
    result = problem.evaluate_true(X)

    return result.numpy()
