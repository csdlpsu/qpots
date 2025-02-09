from unittest.mock import Mock
import pytest
import torch
import numpy as np
import os
import sys
import warnings

warnings.filterwarnings('ignore')
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from pymoo.core.result import Result
from qpots.utils.pymoo_problem import PyMooFunction, nsga2

def test_pymoo_problem_init():
    func = Mock()
    n_var = 2
    n_obj = 2
    xl = 0.
    xu = 1.

    problem = PyMooFunction(func=func, n_var=n_var, n_obj=n_obj, xl=xl, xu=xu)

    assert problem.func == func
    assert problem.n_var == n_var
    assert problem.n_obj == n_obj
    assert np.all(problem.xl == xl)
    assert np.all(problem.xu == xu)


def test_nsga2():
    def func(X):
        f1 = X[:, 0]**2 + X[:, 1]**2 + 2*X[:, 0]*X[:, 1]**2
        f2 = (X[:, 0] - 1)**2 + (X[:, 1] - 1)**2
        return -1*torch.stack([f1, f2], dim=-1)
    n_var = 2
    n_obj = 2
    xl = 0.
    xu = 1.

    pop_size = 100
    ngen = 100

    problem = PyMooFunction(func=func, n_var=n_var, n_obj=n_obj, xl=xl, xu=xu)
    res = nsga2(problem=problem, ngen=ngen, pop_size=pop_size, seed=1023, callback=None)
    assert isinstance(res, Result), "res didn't return a Result object"
    assert isinstance(res.X, np.ndarray), "res.X is not an np.ndarray"
    assert isinstance(res.F, np.ndarray), "res.F is not an np.ndarray"
    assert res.X.shape == (pop_size, n_var), "Shape in res.X is mismatched from expected"
    assert res.F.shape == (pop_size, n_obj), "Shape in res.F is mismatched from expected"
    assert np.all(res.X >= xl) and np.all(res.X <= xu), "Out of bounds"

def test_nsga2_with_callback():
    mock_callback = Mock()

    def func(X):
        f1 = X[:, 0]**2 + X[:, 1]**2 + 2*X[:, 0]*X[:, 1]**2
        f2 = (X[:, 0] - 1)**2 + (X[:, 1] - 1)**2
        return -1*torch.stack([f1, f2], dim=-1)
    n_var = 2
    n_obj = 2
    xl = 0.
    xu = 1.

    pop_size = 100
    ngen = 100

    problem = PyMooFunction(func=func, n_var=n_var, n_obj=n_obj, xl=xl, xu=xu)
    res = nsga2(problem=problem, ngen=ngen, pop_size=pop_size, seed=1023, callback=mock_callback)

    assert mock_callback.call_count > 0, "Callback function was never called"
    assert np.all(res.X >= xl) and np.all(res.X <= xu), "Out of bounds"

def test_pymoo_problem_evaluate():
    func = Mock(return_value=torch.tensor([[1.0, 2.0]]))  # Mock function
    problem = PyMooFunction(func=func, n_var=2, n_obj=2, xl=0., xu=1.)

    out = {}
    x_sample = np.array([[0.5, 0.5]])  # Sample input
    problem._evaluate(x_sample, out)

    func.assert_called_once()  # Ensure func is called
    assert "F" in out, "Output dictionary should have 'F'"
    assert out["F"].shape == (1, 2), "Function output shape mismatch"




