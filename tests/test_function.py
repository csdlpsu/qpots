from unittest.mock import Mock
import pytest
import torch
import os
import sys
import warnings

warnings.filterwarnings('ignore')
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from botorch.test_functions.multi_objective import BraninCurrin, OSY
from qpots.function import Function

@pytest.fixture()
def mock_func():
    func = Mock()
    func.cons = Mock()
    func.bounds = Mock()
    func.name = 'branincurrin'
    func.dim = 2
    func.nobj = 2
    return func

@pytest.fixture()
def custom_function():
    def custom_func(X):
        f1 = X[:, 0]**2 + X[:, 1]**2 + 2*X[:, 0]*X[:, 1]**2
        f2 = (X[:, 0] - 1)**2 + (X[:, 1] - 1)**2
        return -1*torch.stack([f1, f2], dim=-1)
    
    return custom_func

def test_function_init(mock_func):
    name = mock_func.name 
    dim = mock_func.dim
    nobj = mock_func.nobj
    custom_func = mock_func.custom_func
    bounds = mock_func.bounds
    cons = mock_func.cons

    f = Function(
        name=name,
        dim=dim,
        nobj=nobj,
        custom_func=custom_func,
        bounds=bounds,
        cons=cons
    )

    assert f.name == name
    assert f.dim == dim
    assert f.nobj == nobj
    assert f.custom_func == custom_func
    assert f.bounds == bounds
    assert f.cons == cons

def test_func_map_properly_initialized():
    obj = Function('branincurrin', dim=2, nobj=2)
    assert isinstance(obj.func, BraninCurrin), "Function name does not match function instance"
    assert obj.bounds.shape == (2,2), "Bounds are not correct shape"

def test_func_map_with_cons():
    obj = Function('osy')
    assert isinstance(obj.func, OSY), "Function name does not match function instance"
    assert obj.bounds.shape == (2, 6), "Bound are not correct shape"
    assert hasattr(obj, 'cons'), "Cons not initialized"

def test_func_map_evaluate():
    obj = Function('branincurrin', dim=2, nobj=2)
    x = torch.tensor([[0.2, 0.4]])
    res = obj.evaluate(x)
    assert isinstance(res, torch.Tensor), "Result is not a Tensor object"
    assert res.shape == torch.Size([1, obj.dim]), "Shape does not match expected shape"

def test_func_map_with_cons():
    obj = Function('osy', dim=6, nobj=2)
    cons = obj.get_cons()
    x = torch.tensor([[0.2, 0.4, 0.3, 0.2, 0.5, 0.6]])
    res = obj.evaluate(x)
    assert isinstance(res, torch.Tensor), "Result is not a Tensor object"
    assert res.shape == torch.Size([1, obj.nobj]), "Shape does not match expected shape"
    assert isinstance(cons(x), torch.Tensor), "Constraint does not return a Tensor object"
    assert cons(x).shape == torch.Size([1, 6]) # This problem has 6 constraints so 6 "constraint outputs"

def test_custom_function(custom_function):
    x = torch.tensor([[0.2, 0.4]])
    bounds = torch.tensor(([0.0, 0.0], [1.0, 1.0]))
    cons = None
    obj = Function(name=None, dim=2, nobj=2, custom_func=custom_function, bounds=bounds, cons=cons)
    assert isinstance(obj.evaluate(x), torch.Tensor), "Result not a Tensor"
    assert obj.evaluate(x).shape == torch.Size([1, obj.nobj]), "Shapes do not match"

def test_custom_function_with_cons(custom_function):
    x = torch.tensor([0.2, 0.4])
    bounds = torch.tensor(([0.0, 0.0], [1.0, 1.0]))
    def cons(x):
        con1 = x[0] + x[1]*x[0]
        return -1*torch.stack([con1], dim=-1)
    obj = Function(name=None, dim=2, nobj=2, custom_func=custom_function, bounds=bounds, cons=cons)
    cons_func = obj.get_cons()
    assert isinstance(cons_func(x), torch.Tensor), "Tensor not returned from custom constraint function"
    assert cons_func(x).shape == torch.Size([1]), "Size is not properly returned"

def test_custom_function_missing_bounds(custom_function):
    with pytest.raises(ValueError, match="Custom functions must specify bounds."):
        Function(name=None, dim=2, nobj=2, custom_func=custom_function, bounds=None, cons=None)

def test_invalid_function_name():
    with pytest.raises(ValueError, match="Unknown test function 'invalid_name'."):
        Function(name="invalid_name", dim=2, nobj=2)

def test_get_bounds():
    obj = Function('branincurrin', dim=2, nobj=2)
    assert isinstance(obj.get_bounds(), torch.Tensor), "Returned bounds are not a Tensor object"
    assert torch.equal(obj.get_bounds(), obj.bounds), "get_bounds() does not return the correct bounds"

def test_get_bounds_with_custom_function(custom_function):
    bounds = torch.tensor([[0.0, 0.0], [1.0, 1.0]])
    obj = Function(dim=2, nobj=2, custom_func=custom_function, bounds=bounds, cons=None)
    assert isinstance(obj.get_bounds(), torch.Tensor), "Returned bounds are not a Tensor object"
    assert torch.equal(obj.get_bounds(), obj.bounds), "get_bounds() does not return the correct bounds"

def test_get_cons_without_constraints():
    obj = Function('branincurrin', dim=2, nobj=2)
    assert obj.get_cons() is None, "get_cons() should return None for unconstrained functions"

def test_func_map_batch_evaluate():
    obj = Function('branincurrin', dim=2, nobj=2)
    X = torch.tensor([[0.2, 0.4], [0.5, 0.1], [0.9, 0.7]])
    res = obj.evaluate(X)
    assert isinstance(res, torch.Tensor), "Result is not a Tensor"
    assert res.shape == torch.Size([3, obj.nobj]), "Batch evaluation shape mismatch"

def test_custom_function_with_batch_evaluate(custom_function):
    bounds = torch.tensor([[0.0, 0.0], [1.0, 1.0]])
    obj = Function(dim=2, nobj=2, custom_func=custom_function, bounds=bounds, cons=None)
    X = torch.tensor([[0.2, 0.4], [0.5, 0.1], [0.9, 0.7]])
    res = obj.evaluate(X)
    assert isinstance(res, torch.Tensor), "Result is not a Tensor"
    assert res.shape == torch.Size([3, obj.nobj]), "Batch evaluation shape mismatch"
    