import torch
import numpy as np
from botorch.test_functions.multi_objective import DH1


def dh1_eval(x, dim):
    X = torch.tensor(x, dtype=torch.float32)

    problem = DH1(int(dim))

    result = problem.evaluate_true(X)

    return result.numpy()
