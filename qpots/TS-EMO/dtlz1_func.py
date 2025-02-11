import torch
import numpy as np
from botorch.test_functions import DTLZ1

def dtlz1(x, dim):
    X = torch.tensor(x, dtype=torch.float32)

    problem = DTLZ1(int(dim), num_objectives=2)
    result = problem.evaluate_true(X)

    return result.numpy()
    
