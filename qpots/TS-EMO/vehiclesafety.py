import torch
import numpy as np
from botorch.test_functions.multi_objective import VehicleSafety

def vehiclesafety(x):
    X = torch.tensor(x, dtype=torch.float32)
    problem = VehicleSafety()
    result = problem.evaluate_true(X)
    return result.numpy()
