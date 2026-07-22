import torch

from qpots.acquisition import Acquisition
from qpots.config import DEFAULT_DEVICE, DEFAULT_DTYPE, get_device, get_dtype
from qpots.function import Function
from qpots.model_object import ModelObject
from qpots.runner import QPOTSConfig


class _ModelStub:
    nobj = 2
    ncons = 0
    device = torch.device("cpu")
    dtype = torch.float64


def _legacy_objectives(x):
    return torch.column_stack((x[:, 0], x[:, 0] ** 2))


def _legacy_constraints(x):
    return (1.0 - x[:, :1]).to(dtype=x.dtype)


def test_v2_function_constructor_and_separate_constraints_remain_supported():
    function = Function(
        "legacy-custom-name",
        1,
        2,
        _legacy_objectives,
        torch.tensor([[0.0], [1.0]]),
        _legacy_constraints,
        "cpu",
        torch.float64,
    )
    x = torch.tensor([[0.25], [0.75]])

    assert function.evaluate(x).shape == (2, 2)
    assert function.get_cons()(x).shape == (2, 1)
    assert function.evaluate_all(x).constraints.shape == (2, 1)


def test_v2_model_constructor_positional_arguments_remain_supported():
    model = ModelObject(
        torch.tensor([[0.25], [0.75]]),
        torch.tensor([[0.25, 0.0625], [0.75, 0.5625]]),
        torch.tensor([[0.0], [1.0]]),
        2,
        0,
        2,
        "cpu",
        1e-6,
        torch.float64,
    )

    assert model.nobj == 2
    assert model.ncons == 0
    assert model.ntrain == 2
    assert model.device == torch.device("cpu")


def test_v2_acquisition_constructor_positional_arguments_remain_supported():
    function = Function(
        custom_func=_legacy_objectives,
        bounds=torch.tensor([[0.0], [1.0]]),
        dim=1,
        nobj=2,
    )
    acquisition = Acquisition(function, _ModelStub(), None, "cpu", torch.float64, 2, 5, 128)

    assert acquisition.q == 2
    assert acquisition.NUM_RESTARTS == 5
    assert acquisition.RAW_SAMPLES == 128


def test_v2_runtime_constants_and_helpers_remain_available():
    assert get_device() == DEFAULT_DEVICE
    assert get_dtype() == DEFAULT_DTYPE


def test_runner_preserves_legacy_acquisition_keyword_contract():
    function = Function(
        custom_func=_legacy_objectives,
        bounds=torch.tensor([[0.0], [1.0]]),
        dim=1,
        nobj=2,
    )
    options = QPOTSConfig(iterations=3, batch_size=2).acquisition_options(function)

    assert options == {
        "nystrom": 0,
        "iters": 3,
        "nychoice": "pareto",
        "dim": 1,
        "ngen": 10,
        "q": 2,
        "mt": 0,
        "partial_info": 0,
        "threshold": None,
    }
