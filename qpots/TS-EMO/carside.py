import torch
import numpy as np
from botorch.test_functions.multi_objective import CarSideImpact

def carside(x):
    X = torch.tensor(x, dtype=torch.float32)
    problem = CarSideImpact()
    result = problem.evaluate_true(X)
    return result.numpy()
