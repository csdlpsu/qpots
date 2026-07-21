from unittest.mock import Mock

import pytest
import torch
from botorch.test_functions.multi_objective import OSY, BraninCurrin

from qpots.benchmark_registry import available_benchmarks, create_benchmark
from qpots.function import EvaluationResult, Function


@pytest.fixture()
def mock_func():
    func = Mock()
    func.cons = Mock()
    func.bounds = Mock()
    func.name = "branincurrin"
    func.dim = 2
    func.nobj = 2
    return func


@pytest.fixture()
def custom_function():
    def custom_func(X):
        f1 = X[:, 0] ** 2 + X[:, 1] ** 2 + 2 * X[:, 0] * X[:, 1] ** 2
        f2 = (X[:, 0] - 1) ** 2 + (X[:, 1] - 1) ** 2
        return -1 * torch.stack([f1, f2], dim=-1)

    return custom_func


def test_function_init(mock_func):
    name = mock_func.name
    dim = mock_func.dim
    nobj = mock_func.nobj

    def custom_func(x):
        return torch.column_stack((x[:, 0], x[:, 1]))

    bounds = torch.tensor([[0.0, 0.0], [1.0, 1.0]])
    cons = mock_func.cons

    f = Function(name=name, dim=dim, nobj=nobj, custom_func=custom_func, bounds=bounds, cons=cons)

    assert f.name == name
    assert f.dim == dim
    assert f.nobj == nobj
    assert f.custom_func == custom_func
    assert torch.equal(f.bounds, bounds.to(dtype=torch.float64))
    assert f.cons == cons


def test_func_map_properly_initialized():
    obj = Function("branincurrin", dim=2, nobj=2)
    assert isinstance(obj.func, BraninCurrin), "Function name does not match function instance"
    assert obj.bounds.shape == (2, 2), "Bounds are not correct shape"


def test_constrained_func_map_is_initialized():
    obj = Function("osy")
    assert isinstance(obj.func, OSY), "Function name does not match function instance"
    assert obj.bounds.shape == (2, 6), "Bound are not correct shape"
    assert hasattr(obj, "cons"), "Cons not initialized"


def test_func_map_evaluate():
    obj = Function("branincurrin", dim=2, nobj=2)
    x = torch.tensor([[0.2, 0.4]])
    res = obj.evaluate(x)
    assert isinstance(res, torch.Tensor), "Result is not a Tensor object"
    assert res.shape == torch.Size([1, obj.dim]), "Shape does not match expected shape"


def test_constrained_func_map_evaluates_constraints():
    obj = Function("osy", dim=6, nobj=2)
    cons = obj.get_cons()
    x = torch.tensor([[0.2, 0.4, 0.3, 0.2, 0.5, 0.6]])
    res = obj.evaluate(x)
    assert isinstance(res, torch.Tensor), "Result is not a Tensor object"
    assert res.shape == torch.Size([1, obj.nobj]), "Shape does not match expected shape"
    assert isinstance(cons(x), torch.Tensor), "Constraint does not return a Tensor object"
    assert cons(x).shape == torch.Size(
        [1, 6]
    )  # This problem has 6 constraints so 6 "constraint outputs"


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
        con1 = x[0] + x[1] * x[0]
        return -1 * torch.stack([con1], dim=-1)

    obj = Function(name=None, dim=2, nobj=2, custom_func=custom_function, bounds=bounds, cons=cons)
    cons_func = obj.get_cons()
    assert isinstance(cons_func(x), torch.Tensor), (
        "Tensor not returned from custom constraint function"
    )
    assert cons_func(x).shape == torch.Size([1]), "Size is not properly returned"


def test_custom_function_missing_bounds(custom_function):
    with pytest.raises(ValueError, match="Custom functions must specify bounds."):
        Function(name=None, dim=2, nobj=2, custom_func=custom_function, bounds=None, cons=None)


def test_invalid_function_name():
    with pytest.raises(ValueError, match="Unknown test function 'invalid_name'."):
        Function(name="invalid_name", dim=2, nobj=2)


def test_get_bounds():
    obj = Function("branincurrin", dim=2, nobj=2)
    assert isinstance(obj.get_bounds(), torch.Tensor), "Returned bounds are not a Tensor object"
    assert torch.equal(obj.get_bounds(), obj.bounds), (
        "get_bounds() does not return the correct bounds"
    )


def test_get_bounds_with_custom_function(custom_function):
    bounds = torch.tensor([[0.0, 0.0], [1.0, 1.0]])
    obj = Function(dim=2, nobj=2, custom_func=custom_function, bounds=bounds, cons=None)
    assert isinstance(obj.get_bounds(), torch.Tensor), "Returned bounds are not a Tensor object"
    assert torch.equal(obj.get_bounds(), obj.bounds), (
        "get_bounds() does not return the correct bounds"
    )


def test_get_cons_without_constraints():
    obj = Function("branincurrin", dim=2, nobj=2)
    assert obj.get_cons() is None, "get_cons() should return None for unconstrained functions"


def test_func_map_batch_evaluate():
    obj = Function("branincurrin", dim=2, nobj=2)
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


# ---------------------------------------------------------------------------
# Tests added for improved coverage
# ---------------------------------------------------------------------------


@pytest.mark.parametrize(
    "name,dim,nobj",
    [
        ("dtlz1", 5, 2),
        ("dtlz2", 5, 2),
        ("dtlz3", 5, 2),
        ("dtlz7", 5, 2),
        ("dh1", 4, 2),
        ("gmm", 2, 2),
        ("penicillin", 7, 3),
        ("vehicle", 5, 3),
        ("carside", 7, 4),
        ("zdt3", 6, 2),
        ("discbrake", 4, 2),
        ("mw7", 2, 2),
        ("weldedbeam", 4, 2),
        ("constrainedbc", 2, 2),
    ],
)
def test_registered_functions_smoke(name, dim, nobj):
    """Every registered function name must initialise and return the right output shape."""
    obj = Function(name=name, dim=dim, nobj=nobj)
    actual_dim = obj.bounds.shape[1]
    x = obj.bounds[0] + torch.rand(3, actual_dim, dtype=torch.float64) * (
        obj.bounds[1] - obj.bounds[0]
    )
    res = obj.evaluate(x)
    assert isinstance(res, torch.Tensor), f"{name}: result is not a Tensor"
    assert res.shape == torch.Size([3, nobj]), (
        f"{name}: expected shape [3, {nobj}], got {res.shape}"
    )


def test_evaluate_output_columns_match_nobj_not_dim():
    """evaluate() returns nobj columns even when nobj differs from dim."""
    obj = Function("dtlz1", dim=5, nobj=2)  # 5 input dims, 2 objectives
    x = obj.bounds[0] + torch.rand(4, 5, dtype=torch.float64) * (obj.bounds[1] - obj.bounds[0])
    res = obj.evaluate(x)
    assert res.shape == torch.Size([4, 2]), f"Expected [4, nobj=2], got {res.shape}"


def test_single_objective_branin():
    """branin wraps a single-objective function; evaluate() produces one output per point."""
    obj = Function("branin", dim=2, nobj=1)
    assert obj.get_cons() is None
    x = obj.bounds[0] + torch.rand(3, 2, dtype=torch.float64) * (obj.bounds[1] - obj.bounds[0])
    res = obj.evaluate(x)
    assert isinstance(res, torch.Tensor)
    assert res.shape[0] == 3, "Batch dimension must be preserved"


def test_constrained_function_evaluate_returns_only_objectives():
    """evaluate() yields objective values only; constraints are accessed via get_cons()."""
    obj = Function("weldedbeam", dim=4, nobj=2)
    assert obj.get_cons() is not None, "WeldedBeam must expose constraint callable"
    actual_dim = obj.bounds.shape[1]
    x = obj.bounds[0] + torch.rand(2, actual_dim, dtype=torch.float64) * (
        obj.bounds[1] - obj.bounds[0]
    )
    res = obj.evaluate(x)
    assert res.shape == torch.Size([2, 2]), (
        "evaluate() must return [n, nobj] only, not [n, nobj+ncons]"
    )
    cons_res = obj.get_cons()(x)
    assert isinstance(cons_res, torch.Tensor)


def test_registry_helpers():
    assert "branincurrin" in available_benchmarks()
    assert isinstance(create_benchmark("BraninCurrin", dim=2, nobj=2), BraninCurrin)
    with pytest.raises(ValueError, match="Unknown test function"):
        create_benchmark("missing", dim=2, nobj=2)


def test_bounds_shape_and_values_are_validated(custom_function):
    with pytest.raises(ValueError, match=r"shape \(2, 3\)"):
        Function(dim=3, nobj=2, custom_func=custom_function, bounds=torch.zeros(3, 2))
    with pytest.raises(ValueError, match="strictly less"):
        Function(dim=2, nobj=2, custom_func=custom_function, bounds=torch.ones(2, 2))
    invalid = torch.tensor([[0.0, 0.0], [1.0, float("inf")]])
    with pytest.raises(ValueError, match="finite"):
        Function(dim=2, nobj=2, custom_func=custom_function, bounds=invalid)


def test_function_can_be_subclassed():
    class QuadraticFunction(Function):
        def _evaluate(self, X):
            return torch.column_stack((X[:, 0] ** 2, X[:, 0] + 1))

    function = QuadraticFunction(
        dim=1,
        nobj=2,
        bounds=torch.tensor([[0.0], [2.0]]),
    )
    result = function.evaluate(torch.tensor([[0.5]]))
    assert torch.allclose(result, torch.tensor([[0.25, 1.5]], dtype=torch.float64))


def test_combined_evaluation_returns_objectives_and_constraints():
    calls = 0

    def combined(X):
        nonlocal calls
        calls += 1
        return EvaluationResult(
            objectives=torch.column_stack((X[:, 0], -X[:, 0])),
            constraints=(X[:, :1] - 0.5),
        )

    function = Function(
        dim=1,
        nobj=2,
        bounds=torch.tensor([[0.0], [1.0]]),
        combined_func=combined,
    )
    result = function.evaluate_all(torch.tensor([[0.25], [0.75]]))
    assert calls == 1
    assert result.objectives.shape == (2, 2)
    assert result.constraints.shape == (2, 1)


def test_separate_evaluation_returns_objectives_and_constraints():
    function = Function(
        dim=1,
        nobj=2,
        bounds=torch.tensor([[0.0], [1.0]]),
        custom_func=lambda X: torch.column_stack((X[:, 0], -X[:, 0])),
        cons=lambda X: X[:, :1] - 0.5,
    )
    result = function.evaluate_all(torch.tensor([[0.25], [0.75]]))
    assert result.objectives.shape == (2, 2)
    assert result.constraints.shape == (2, 1)


def test_evaluation_output_shape_is_validated():
    function = Function(
        dim=1,
        nobj=2,
        bounds=torch.tensor([[0.0], [1.0]]),
        custom_func=lambda X: X[:, :1],
    )
    with pytest.raises(ValueError, match="objective output must have shape"):
        function.evaluate(torch.tensor([[0.5]]))
