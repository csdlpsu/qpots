import torch
import numpy as np
from botorch.test_functions.multi_objective import Penicillin


def Penicillin_evaluate(x):
    X = torch.tensor(x, dtype=torch.float32)

    problem = Penicillin()

    result = problem.evaluate_true(X)

    return result

    
