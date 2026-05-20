# Assuming BC_function.py contains the necessary imports and the BraninCurrin class definition

import torch
import numpy as np
from botorch.test_functions import BraninCurrin
from qpots.config import as_tensor

def BC_evaluate(x):
    """
    Evaluate the BoTorch Branin-Currin benchmark for the MATLAB TS-EMO bridge.

    Parameters
    ----------
    x : array-like
        Candidate design points supplied by MATLAB or NumPy with shape
        ``n x 2``.

    Returns
    -------
    numpy.ndarray
        True Branin-Currin objective values evaluated at ``x``.
    """
    # Convert numpy array input to a PyTorch tensor
    X = as_tensor(x)
    
    # Instantiate BraninCurrin problem
    problem = BraninCurrin().to(device=X.device, dtype=X.dtype)
    
    # Evaluate the problem with the given inputs
    result = problem.evaluate_true(X)
    
    # Convert the result back to a numpy array and return
    return result.detach().cpu().numpy()
