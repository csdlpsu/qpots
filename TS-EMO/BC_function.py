# Assuming BC_function.py contains the necessary imports and the BraninCurrin class definition

import torch
import numpy as np
from botorch.test_functions import BraninCurrin

def BC_evaluate(x):
    # Convert numpy array input to a PyTorch tensor
    X = torch.tensor(x, dtype=torch.float32)
    
    # Instantiate BraninCurrin problem
    problem = BraninCurrin()
    
    # Evaluate the problem with the given inputs
    result = problem.evaluate_true(X)
    
    # Convert the result back to a numpy array and return
    return result.numpy()
